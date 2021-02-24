import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import random


class Maps(object):
    def __init__(self, difficulty):
        assert difficulty in ['easy', 'medium', 'hard']

        if difficulty is 'easy':
            random.seed(4)
            m = 7
            n = 7
            g = nx.grid_2d_graph(m, n)
            g = nx.convert_node_labels_to_integers(g, first_label=1)

            map_adjlist = nx.to_dict_of_lists(g)
            intra_nodes = [i for i in g.nodes() if len(map_adjlist[i]) == 4]

            p = 0.5
            to_remove = []
            for e in g.edges():
                if random.random() >= p:
                    to_remove.append(e)
            g.remove_edges_from(to_remove)

            q = 0.1

            def other_nodes(node, n=n):
                return node-n-1, node-n+1, node+n-1, node+n+1
            add_edges = []
            for node in intra_nodes:
                for other_node in other_nodes(node):
                    if random.random() <= q:
                        add_edges.append((node, other_node))
            g.add_edges_from(add_edges)

            g.add_edge(20, 21)
            map_adjlist = nx.to_dict_of_lists(g)
            max_actions = 0
            for node in map_adjlist:
                map_adjlist[node].append(node)
                map_adjlist[node].sort()
                if len(map_adjlist[node]) > max_actions:
                    max_actions = len(map_adjlist[node])
            self.num_nodes = len(map_adjlist)
            self.adjlist = map_adjlist
            self.defender_init = [(11, 23, 27, 39)]
            self.attacker_init = [25]
            self.exits = [4, 36, 45, 48, 7, 28, 8, 29, 1, 34]
            self.num_defender = len(self.defender_init[0])
            self.max_actions = pow(max_actions, self.num_defender)
            self.graph = g
            self.size = [m, n]

            self.embedding_size= 32
            self.hidden_size= 64
            self.relevant_v_size= 64

            self.time_horizon=7

        elif difficulty is 'medium':
            random.seed(100)
            m = 15
            n = 15
            g = nx.grid_2d_graph(m, n)
            g = nx.convert_node_labels_to_integers(g, first_label=1)

            map_adjlist = nx.to_dict_of_lists(g)
            intra_nodes = [i for i in g.nodes() if len(map_adjlist[i]) == 4]

            p = 0.4
            to_remove = []
            for e in g.edges():
                if random.random() >= p:
                    to_remove.append(e)
            g.remove_edges_from(to_remove)
            q = 0.1

            def other_nodes(node, n=n):
                return node-n-1, node-n+1, node+n-1, node+n+1
            add_edges = []
            for node in intra_nodes:
                for other_node in other_nodes(node):
                    if random.random() <= q:
                        add_edges.append((node, other_node))
            g.add_edges_from(add_edges)
            g.add_edge(162, 177)
            g.add_edge(21, 22)
            g.add_edge(217, 218)
            map_adjlist = nx.to_dict_of_lists(g)
            max_actions = 0
            for node in map_adjlist:
                map_adjlist[node].append(node)
                map_adjlist[node].sort()
                if len(map_adjlist[node]) > max_actions:
                    max_actions = len(map_adjlist[node])
            self.num_nodes = len(map_adjlist)
            self.adjlist = map_adjlist
            self.defender_init = [(53, 117, 173, 109)]
            self.attacker_init = [113]
            self.exits = [61, 151, 217, 223, 180, 75, 12, 6, 225, 31]
            self.num_defender = len(self.defender_init[0])
            self.max_actions = pow(max_actions, self.num_defender)
            self.graph = g
            self.size = [m, n]

            self.embedding_size= 32
            self.hidden_size= 64
            self.relevant_v_size= 64

            self.time_horizon=15

        else:  # hard
            random.seed(100)
            m = 30
            n = 30
            g = nx.grid_2d_graph(m, n)
            g = nx.convert_node_labels_to_integers(g, first_label=1)

            map_adjlist = nx.to_dict_of_lists(g)
            intra_nodes = [i for i in g.nodes() if len(map_adjlist[i]) == 4]
            q = 0.05

            def other_nodes(node, n=n):
                return node-n-1, node-n+1, node+n-1, node+n+1
            add_edges = []
            for node in intra_nodes:
                for other_node in other_nodes(node):
                    if random.random() <= q:
                        add_edges.append((node, other_node))
            g.add_edges_from(add_edges)

            p = 0.5
            num_remove_edge = 0
            while True:
                # for e in g.edges():
                e = random.choice(list(g.edges()))
                if e not in add_edges:
                    if random.random() >= p:
                        g.remove_edge(*e)
                        num_remove_edge += 1
                        if not nx.is_connected(g):
                            g.add_edge(*e)
                            num_remove_edge -= 1
                if num_remove_edge > (1-p)*((m-1)*n+m*(n-1)):
                    break
            g.remove_edge(312, 342)
            map_adjlist = nx.to_dict_of_lists(g)
            max_actions = 0
            for node in map_adjlist:
                map_adjlist[node].append(node)
                map_adjlist[node].sort()
                if len(map_adjlist[node]) > max_actions:
                    max_actions = len(map_adjlist[node])
            self.num_nodes = len(map_adjlist)
            self.adjlist = map_adjlist
            self.defender_init = [(256, 616, 430, 442)]
            self.attacker_init = [436]
            self.exits = [29, 882, 888, 895, 31, 781, 150, 300, 660, 780]
            self.num_defender = len(self.defender_init[0])
            self.max_actions = pow(max_actions, self.num_defender)
            self.graph = g
            self.size = [m, n]

            self.embedding_size= 32
            self.hidden_size= 96
            self.relevant_v_size= 64

            self.time_horizon=30


if __name__ == '__main__':
    Map = Maps('easy')
    EASY = Map.size
    EASY.append(Map.adjlist)
    print(EASY)
    
    # Output:
    # [7, 7, {1: [1, 2, 8], 2: [1, 2, 3, 9], 3: [2, 3, 4, 10], 4: [3, 4, 12], 5: [5, 6], 6: [5, 6, 7], 
    # 7: [6, 7, 13, 14], 8: [1, 8, 9, 15], 9: [2, 8, 9], 10: [3, 10, 16], 11: [11, 12, 18, 19], 
    # 12: [4, 11, 12], 13: [7, 13], 14: [7, 14, 21], 15: [8, 15], 16: [10, 16, 17, 22, 24], 
    # 17: [16, 17, 18, 23, 24], 18: [11, 17, 18], 19: [11, 19, 20], 20: [19, 20, 21, 26], 
    # 21: [14, 20, 21], 22: [16, 22], 23: [17, 23, 30], 24: [16, 17, 24, 25, 30, 31], 
    # 25: [24, 25, 32, 33], 26: [20, 26, 27, 33], 27: [26, 27, 28], 28: [27, 28, 34], 
    # 29: [29, 30, 36], 30: [23, 24, 29, 30, 31], 31: [24, 30, 31, 32, 37, 38], 
    # 32: [25, 31, 32, 33], 33: [25, 26, 32, 33, 40], 34: [28, 34, 40, 41], 
    # 35: [35], 36: [29, 36, 37, 43], 37: [31, 36, 37, 44], 38: [31, 38, 39, 44], 
    # 39: [38, 39, 40, 46], 40: [33, 34, 39, 40, 41, 47, 48], 41: [34, 40, 41], 
    # 42: [42], 43: [36, 43], 44: [37, 38, 44], 45: [45, 46], 46: [39, 45, 46, 47], 
    # 47: [40, 46, 47], 48: [40, 48], 49: [49]}]

    defender_init_location=Map.defender_init
    attacker_init_location=Map.attacker_init
    exits=Map.exits
    print('defender init location : {}, attacker init location : {}, exits location : {}'.format(defender_init_location,attacker_init_location,exits))

    # Output:
    # defender init location: [(11, 23, 27, 39)], attacker init location: [25], exits location: [4, 36, 45, 48, 7, 28, 8, 29, 1, 34]
