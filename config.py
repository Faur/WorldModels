import numpy as np
import random

train_envs = ['car_racing']
test_envs = ['car_racing']

def generate_data_action(t, current_action):
    # a = env.action_space.sample()
    return np.array([random.random()*2-1,random.random(),random.random()])
    # if t < 10:
    #     return np.array([0,1,0])
    
    # if t % 5 > 0:
    #     return current_action

    # rn = random.randint(0,9)
    # if rn in [0]:
    #     return np.array([0,0,0])
    # if rn in [1,2,3,4]:
    #     return np.array([0,random.random(),0])
    # if rn in [5,6,7]:
    #     return np.array([-random.random(),0,0])
    # if rn in [8]:
    #     return np.array([random.random(),0,0])
    # if rn in [9]:
    #     return np.array([0,0,random.random()])


def adjust_obs(obs):
    return obs.astype('float32') / 255.


def select_action(env, obs, t, controller):
    if controller == 'random':
        action = env.action_space.sample()
        action[2] = action[2] / 10  # don't break to much.
        if t < np.random.randint(100):  # Be biased towards accelerating in the beginning.
            action[1] = 1
        return action
    elif controller == 'human':
        # TODO
        raise NotImplementedError
    elif controller == 'agent':
        # TODO
        raise NotImplementedError
    else:
        raise Exception('Controller "' + str(controller) + '" not understood.')
