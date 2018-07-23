#xvfb-run -s "-screen 0 1400x900x24" python 01_generate_data.py car_racing --total_episodes 200 --start_batch 0 --time_steps 300

import numpy as np
import time
import random
import config
#import matplotlib.pyplot as plt

from env import make_env

import argparse

def main(args):

    env_name = args.env_name
    path = args.path
    total_episodes = args.total_episodes
    start_batch = args.start_batch
    time_steps = args.time_steps
    render = args.render
    batch_size = args.batch_size
    run_all_envs = args.run_all_envs
    dont_save = args.dont_save

    if run_all_envs:
        envs_to_generate = config.train_envs
    else:
        envs_to_generate = [env_name]


    for current_env_name in envs_to_generate:
        print("Generating data for env {}".format(current_env_name))

        np.random.seed(int(time.time()))

        env = make_env(current_env_name)
        s = 0
        batch = start_batch

        batch_size = min(batch_size, total_episodes)

        while s < total_episodes:
            obs_data = []
            action_data = []
            reward_data = []
            done_data = []

            for i_episode in range(batch_size):
                    print('-----')
                    observation = env.reset()
                    observation = config.adjust_obs(observation)

                    # plt.imshow(observation)
                    # plt.show()

                    env.render()
                    # action = env.action_space.sample()
                    # action[1] = 1
                    # action = np.array([0, 1, 0])
                    t = 0
                    t_random = np.random.randint(0, 200)
                    print('t_random', t_random)

                    obs_sequence = []
                    action_sequence = []
                    reward_sequence = []
                    done_sequence = []

                    while t < time_steps: #and not done:
                        obs_sequence.append(observation)

                        # if t < t_random:
                        #     pass
                        # else:
                        #     action = config.generate_data_action(t, env, action)
                        action = config.select_action(env, observation, t, 'random')

                        action_sequence.append(action)

                        observation, reward, done, info = env.step(action)

                        reward_sequence.append(reward)
                        done_sequence.append(done)

                        observation = config.adjust_obs(observation)

                        t = t + 1

                        if render:
                            env.render()

                    obs_data.append(obs_sequence)
                    action_data.append(action_sequence)
                    reward_data.append(reward_sequence)
                    done_data.append(done_sequence)

                    print("Batch {} Episode {} finished after {} timesteps".format(batch, i_episode, t))
                    print("Current dataset contains {} observations".format(sum(map(len, obs_data))))

                    s = s + 1

            if not dont_save:
                print("Saving dataset for batch {}".format(batch))
                np.save(path+'/obs_data_' + current_env_name + '_' + str(batch), obs_data)
                np.save(path+'./action_data_' + current_env_name + '_' + str(batch), action_data)
                np.save(path+'./reward_data_' + current_env_name + '_' + str(batch), reward_data)
                np.save(path+'./done_data_' + current_env_name + '_' + str(batch), done_data)
            else:
                print("NOT saving batch {}".format(batch))

            batch = batch + 1

        env.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Create new training data'))
  parser.add_argument('env_name', type=str, help='name of environment')
  parser.add_argument('--path', type=str, default='./data', help='folder to save in')
  parser.add_argument('--total_episodes', type=int, default = 200, help='total number of episodes to generate')
  parser.add_argument('--start_batch', type=int, default = 0, help='start_batch number')
  parser.add_argument('--time_steps', type=int, default = 300, help='how many timesteps at start of episode?')
  parser.add_argument('--render', action='store_true', help='render the env as data is generated')
  parser.add_argument('--dont_save', action='store_true', help='dont save the data (for testing purpose).')
  parser.add_argument('--batch_size', type=int, default = 200, help='how many episodes in a batch (one file)')
  parser.add_argument('--run_all_envs', action='store_true', help='if true, will ignore env_name and loop over all envs in train_envs variables in config.py')

  args = parser.parse_args()
  print(args)
  print()

  main(args)
