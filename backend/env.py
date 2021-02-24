import copy
from itertools import product
from maps import Maps

class Env(object):
    def __init__(self, difficulty):
        assert difficulty in ['easy', 'medium', 'hard']
        Map=Maps(difficulty)
        
        self.adjlist = Map.adjlist
        self.time_horizon = Map.time_horizon
        self.defender_init = Map.defender_init
        self.attacker_init = Map.attacker_init
        self.exits = Map.exits

        self.multi_defender = False
        if isinstance(self.defender_init[0], tuple) and len(self.defender_init[0]) > 1:
            self.multi_defender = True
            self.num_defender = len(self.defender_init[0])
        self.reset(self.defender_init, self.attacker_init)

    def simu_step(self, defender_a, attacker_a):

        assert defender_a in self.current_state.legal_action(0)
        assert attacker_a in self.current_state.legal_action(1)
        d_h = copy.deepcopy(self.current_state.defender_history)
        d_h.append(defender_a)
        a_h = copy.deepcopy(self.current_state.attacker_history)
        a_h.append(attacker_a)

        self.current_state = GameState(self, d_h, a_h)
        return self.current_state

    def reset(self, defender_init=None, attacker_init=None):
        if defender_init:
            self.defender_init = defender_init
        if attacker_init:
            self.attacker_init = attacker_init
        self.current_state = GameState(
            self, copy.deepcopy(self.defender_init), copy.deepcopy(self.attacker_init))
        return self.current_state


class GameState(object):
    # output of environ
    def __init__(self, env, defender_history, attacker_history):
        self.adjlist = env.adjlist
        self.time_horizon = env.time_horizon
        self.exits = env.exits
       
        self.defender_history = defender_history
        self.attacker_history = attacker_history
        self.multi_defender = env.multi_defender
        if self.multi_defender:
            self.num_defender = env.num_defender
        assert len(self.defender_history) == len(self.attacker_history)
        assert len(self.defender_history) >= 1 and len(
            self.defender_history) <= self.time_horizon+1, self.defender_history

    def is_end(self):
        # output bool
        assert len(self.defender_history) == len(self.attacker_history)
        if not self.multi_defender:
            if (len(self.defender_history) == self.time_horizon+1) or \
                    (self.defender_history[-1] == self.attacker_history[-1]) or (self.attacker_history[-1] in self.exits):
                return True
            else:
                return False
        else:
            if (len(self.defender_history) == self.time_horizon+1) or \
                    (self.attacker_history[-1] in self.defender_history[-1]) or (self.attacker_history[-1] in self.exits):
                return True
            else:
                return False

    def obs(self, play_id=None):
        # output format: ([1,2,3],4)/ [1,2,3]
        # for multi-defender: ([1,2,3],(4,5))/ [1,2,3]
        defender_obs = (self.attacker_history, self.defender_history[-1])
        attacker_obs = (self.attacker_history, self.defender_history[0])

        if play_id is None:
            return defender_obs, attacker_obs
        elif play_id == 0:
            return defender_obs
        elif play_id == 1:
            return attacker_obs
        else:
            ValueError('invalid player_id.')

    def reward(self, play_id=None):
        # reward for current state
     
        defender_reward = 0
        if self.attacker_history[-1] in self.defender_history[-1]:
            defender_reward += 1
        elif self.attacker_history[-1] in self.exits:
            defender_reward -= 1
        elif len(self.attacker_history) == self.time_horizon+1:
            defender_reward += 1
        attacker_reward = - defender_reward
        
        if play_id is None:
            return defender_reward, attacker_reward
        elif play_id == 0:
            return defender_reward
        elif play_id == 1:
            return attacker_reward
        else:
            ValueError('invalid player_id.')

    def legal_action(self, play_id=None):
        # output format: [1,2,3] , [4,5,6,7]
        if play_id is None:
            if self.is_end():
                if not self.multi_defender:
                    defender_legal_actions = [0]
                else:
                    defender_legal_actions = [(0,)*self.num_defender]
                attacker_legal_actions = [0]
            else:
                if not self.multi_defender:
                    defender_legal_actions = self.adjlist[self.defender_history[-1]]
                else:
                    defender_legal_actions = self._query_legal_defender_actions(
                        self.defender_history[-1])
                attacker_legal_actions = self.adjlist[self.attacker_history[-1]]
            return defender_legal_actions, attacker_legal_actions

        elif play_id == 0:  #return defender legal actions
            if self.is_end():
                if not self.multi_defender:
                    defender_legal_actions = [0]
                else:
                    defender_legal_actions = [(0,)*self.num_defender]
             
            else:
                if not self.multi_defender:
                    defender_legal_actions = self.adjlist[self.defender_history[-1]]
                else:
                    defender_legal_actions = self._query_legal_defender_actions(
                        self.defender_history[-1])
            return defender_legal_actions

        elif play_id == 1:  # return attacker legal actions
            if self.is_end():
                attacker_legal_actions = [0]
            else:
                attacker_legal_actions = self.adjlist[self.attacker_history[-1]]
            return attacker_legal_actions
        else:
            ValueError('invalid player_id.')

    def _query_legal_defender_actions(self, current_position):
        # current_position: (position0, position1,...)
        before_combination = []
        for i in range(len(current_position)):
            before_combination.append(self.adjlist[current_position[i]])
        return list(product(*before_combination))
