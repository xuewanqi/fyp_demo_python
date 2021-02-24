import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import networkx as nx
import copy
import numpy as np
from os.path import join
import random
import time
from itertools import product
from maps import Maps

device = torch.device('cpu')

class AgentEval(object):
    def __init__(self, difficulty):
        assert difficulty in ['easy', 'medium', 'hard']

        self.map=Maps(difficulty)

        node_embedding_path=join('./backend/saved_model',difficulty,'node_embedding.npy')
        model_path = join('./backend/saved_model', difficulty,
                            'avg_net.pt')
        
        self.avg_net = DRRN(self.map.num_nodes, self.map.time_horizon, node_embedding_path, self.map.embedding_size, self.map.hidden_size,
                            self.map.relevant_v_size, num_defender=self.map.num_defender).to(device)
        self.avg_net.load_state_dict(torch.load(model_path,map_location=device))

    def select_action(self, observation, legal_actions):
        assert len(observation) == 1
        assert len(legal_actions) == 1
        with torch.no_grad():
            prob = self.action_probs(
                observation, legal_actions, numpy=False)
            action_idx = torch.multinomial(
                prob, num_samples=1).item()
            action = legal_actions[0][action_idx]
            return action

    def action_probs(self, observation, legal_actions, numpy=False):
        with torch.no_grad():
            prob = F.softmax(self.avg_net(observation, legal_actions))
            if numpy:
                return prob.numpy()
            else:
                return prob




class DRRN(nn.Module):
    def __init__(self, num_nodes, time_horizon, pre_embedding_path, embedding_size,
                 hidden_size, relevant_v_size, num_defender=None, naive=False, out_mode='sl'):
        super(DRRN, self).__init__()

        assert out_mode in ['sl']
        assert naive is False

        weight = torch.FloatTensor(np.load(pre_embedding_path))
        assert weight.size(1) == embedding_size
        self.embedding = nn.Embedding.from_pretrained(
            weight, freeze=True, padding_idx=0)
        self.state_encoder = StateEncoder(num_nodes, time_horizon, embedding_size,
                                            hidden_size, relevant_v_size, num_defender, node_embedding=self.embedding)
            
        
        self.embedding_a = nn.Embedding(
            num_nodes+1, embedding_size, padding_idx=0)
        if num_defender:
            self.fc_a_1 = nn.Linear(embedding_size*num_defender, hidden_size)
            self.fc_a_2 = nn.Linear(hidden_size, relevant_v_size)
        else:
            self.fc_a_1 = nn.Linear(embedding_size, hidden_size)
            self.fc_a_2 = nn.Linear(hidden_size, relevant_v_size)
        if not naive:
            self.fc1 = nn.Linear(relevant_v_size*2, relevant_v_size)
            self.fc2 = nn.Linear(relevant_v_size, 1)

        self.num_nodes = num_nodes
        self.num_defender = num_defender
        self.naive = naive
        
        self.out_mode = out_mode
        self.to(device)

    def forward(self, states, actions):
        s_feature = self.state_encoder(states)
        a_in = [torch.LongTensor(k).to(device) for k in actions]
        a_in = pad_sequence(a_in, batch_first=True,
                            padding_value=0).detach()
        a_in.requires_grad = False  # shape:(batch, max_num_actions)

        a_feature = self.embedding_a(a_in)
        if self.num_defender > 1:
            a_feature = torch.flatten(a_feature, start_dim=2)
        a_feature = F.relu(self.fc_a_1(a_feature))
        # shape:(batch, max_num_actions, relevant_v_size)
        a_feature = self.fc_a_2(a_feature)
        if not self.naive:
            s_feature = s_feature.unsqueeze(1).repeat(1, a_feature.size(1), 1)

            # shape(batch, max_num_actions, full_features)
            feature = torch.cat((s_feature, a_feature), dim=2)
            output = F.relu(self.fc1(feature))
            output = self.fc2(output).squeeze(2)
        else:
            output = torch.matmul(a_feature, s_feature.unsqueeze(
                2)).squeeze(2)

        if output.size(0) > 1:
            mask = [torch.zeros(len(k)).to(device) for k in actions]
            mask = pad_sequence(mask, batch_first=True,
                                padding_value=-1e9).detach()
            mask.requires_grad = False
            output += mask
        if self.out_mode == 'sl':
            return output.squeeze(0)
        elif self.out_mode == 'rl':
            optimal_q, idx = torch.max(output, dim=1)
            optimal_a = a_in[torch.arange(a_in.size(0)), idx]
            return optimal_a, optimal_q, output.squeeze(0)
        else:
            ValueError


class StateEncoder(nn.Module):
    def __init__(self, num_nodes, time_horizon, embedding_size=16,
                 hidden_size=32, relevant_v_size=64, num_defender=None, node_embedding=None):
        super(StateEncoder, self).__init__()  
        self.seq_encoder = Gated_CNN(
            num_nodes, time_horizon, embedding_size, hidden_size, [3], True, node_embedding=node_embedding)

        self.fc_t_1 = nn.Linear(1, 8)
        self.fc_t_2 = nn.Linear(8, 8)
        if node_embedding:
            self.embedding_p = node_embedding
        else:
            self.embedding_p = nn.Embedding(
                num_nodes+1, embedding_size, padding_idx=0)
        if num_defender:
            self.fc_p_1 = nn.Linear(embedding_size*num_defender, hidden_size)
            self.fc_p_2 = nn.Linear(hidden_size, hidden_size)
            self.fc1 = nn.Linear(hidden_size*2+8, relevant_v_size)
        else:
            self.fc1 = nn.Linear(hidden_size+8, relevant_v_size)

        self.num_defender = num_defender
        self.time_horizon = time_horizon

    def forward(self, states, actions=None):
        attacker_history, position = zip(*states)
        norm_t = [[(len(h)-1)/self.time_horizon] for h in attacker_history]

        if self.num_defender:
            if self.num_defender > 1:
                assert len(position[0]) == self.num_defender
            elif self.num_defender == 1:
                assert isinstance(position[0], int)

        h_feature = F.relu(self.seq_encoder(attacker_history))

        norm_t = torch.Tensor(norm_t).to(device)
        t_feature = F.relu(self.fc_t_1(norm_t))
        t_feature = F.relu(self.fc_t_2(t_feature))

        if self.num_defender:
            position = torch.LongTensor(position).to(device)
            p_feature = self.embedding_p(position)
            if self.num_defender > 1:
                p_feature = torch.flatten(p_feature, start_dim=1)

            p_feature = F.relu(self.fc_p_1(p_feature))
            p_feature = F.relu(self.fc_p_2(p_feature))
            if h_feature.dim() == 1:
                h_feature.unsqueeze_(0)
            feature = torch.cat((h_feature, p_feature, t_feature), dim=1)
        else:
            if h_feature.dim() == 1:
                h_feature.unsqueeze_(0)
            feature = torch.cat((h_feature, t_feature), dim=1)
        output = self.fc1(feature)
        return output  # shape: (batch_size, relevant_v_sie)


class Gated_CNN(nn.Module):
    def __init__(self, num_nodes, time_horizon, embedding_size=32, num_kernels=64, kernel_size=[3], if_gate=True, node_embedding=None):
        # num_kernels is the dimension of features
        super(Gated_CNN, self).__init__()
        self.max_length = time_horizon+1  # to guide padding
        self.time_indicator = [[1 for idx in range(self.max_length)]]
        self.if_gate = if_gate
        if node_embedding:
            self.embedding = node_embedding
        else:
            self.embedding = nn.Embedding(
                num_nodes+1, embedding_size, padding_idx=0)  # +1 is for padding

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size]
        self.kernel_size = kernel_size
        self.conv = torch.nn.ModuleList()
        self.gate = torch.nn.ModuleList()
        for i in range(len(kernel_size)):
            self.conv.append(
                nn.Conv1d(embedding_size, num_kernels, kernel_size[i]))
            self.gate.append(
                nn.Conv1d(embedding_size, num_kernels, kernel_size[i]))
        self.to(device)
        # self.fc=nn.Linear(embedding_size,np.sum(num_kernels))

    def forward(self, inputs):
        # inputs:[[a,b,c,...],[c,d,a,...]]
        if not isinstance(inputs, list):
            inputs = list(inputs)
        inputs = inputs+self.time_indicator
        inputs = [torch.LongTensor(k).to(device) for k in inputs]
        inputs = pad_sequence(inputs, batch_first=True,
                              padding_value=0).detach()[:-1]
        inputs.requires_grad = False

        assert inputs.size(
            1) == self.max_length, 'max input sequence length must less than time horizon.'
        inputs = self.embedding(inputs).permute(0, 2, 1)

        outputs = []
        for i in range(len(self.kernel_size)):
            x = self.conv[i](inputs)
            if self.if_gate:
                gate = self.gate[i](inputs)
                outputs.append(x*F.sigmoid(gate))
            else:
                outputs.append(x)
        x = torch.cat(outputs, dim=2)
        x = F.max_pool1d(x, x.shape[-1])
        x = torch.squeeze(x)
        # x is the feature vector for the input sequence
        return x
