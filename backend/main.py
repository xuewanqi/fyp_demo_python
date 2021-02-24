import env
import agent
import maps
import random

if __name__ == '__main__':
    difficulty='easy'

    Environment=env.Env(difficulty)
    Defender=agent.AgentEval(difficulty)

    for e in range(3):
        game_state=Environment.reset()
        while not game_state.is_end():
        
            defender_obs, attacker_obs = game_state.obs()
            def_current_legal_action, att_current_legal_action = game_state.legal_action()

            defender_a = Defender.select_action(
                [defender_obs], [def_current_legal_action])
            attacker_a = random.choice(att_current_legal_action)
            
            game_state = Environment.simu_step(defender_a, attacker_a)
            def_reward, att_reward = game_state.reward()

            print('Attacker location : ', game_state.attacker_history[-1])
            print('Attacker legal actions', att_current_legal_action)
            print('Attacker history : ', game_state.attacker_history)

            print('Defender location : ', game_state.defender_history[-1])
            # print('Defender legal actions', def_current_legal_action)
            print('Defender history : ', game_state.defender_history)

            print('T : ', len(game_state.attacker_history)-1)
        print('Game is end, defender reward : {} \n\n\n'.format(def_reward))