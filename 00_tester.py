import argparse
import numpy as np
import matplotlib.pyplot as plt

import config
from env import make_env
from vae.arch import VAE
from rnn.arch import RNN


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

def main(args):
    controller = args.controller
    show = args.show

    plt.ion()

    if show == 'both':
        show_simulation = True
        show_vae = True

        fig, [ax_org, ax_vae, ax_sim] = plt.subplots(1, 3)

        vae = VAE()
        try:
            vae.set_weights('./vae/weights.h5')
        except:
            raise Exception("Either set --new_model or ensure ./vae/weights.h5 exists")

        rnn = RNN()  # learning_rate = LEARNING_RATE
        try:
            rnn.set_weights('./rnn/weights.h5')
        except:
            raise Exception("Ensure ./rnn/weights.h5 exists")

    current_env_name = config.train_envs[0]
    env = make_env(current_env_name)
    print('current_env_name', current_env_name)

    episode = 0
    while True:
        episode += 1
        done = False
        reward_sum = 0

        observation = env.reset()
        env.render()
        observation = config.adjust_obs(observation)

        if show_simulation:
            z_enc = vae.encode(np.expand_dims(observation, 0))[0]

            rnn_hidden = np.zeros([1, rnn.hidden_units])
            rnn_cell = np.zeros([1, rnn.hidden_units])

            # print(z_enc.shape)
            # print()
            # print(z_enc)

        t = 0
        while not done:
            t += 1
            action = select_action(env, observation, t, controller)

            if show_simulation:
                # rnn_input = np.concatenate([z_enc, np.expand_dims(action, 0)], axis=1)
                # rnn_input = np.expand_dims(rnn_input, 0)
                rnn_input = [
                    np.array([[np.concatenate([z_enc, action])]]),
                    rnn_hidden,
                    rnn_cell
                ]

                rnn_pred, rnn_hidden, rnn_cell = rnn.forward.predict(rnn_input)

                reward_pred = rnn_pred[0,-1]
                # reward = reward_pred

                mixture_coef = rnn_pred[0,:-1]
                mixture_coef = mixture_coef.reshape([-1, rnn.gaussian_mixtures*3])
                log_pi, mu, log_sigma = np.split(mixture_coef, 3, -1)
                log_pi = log_pi - np.log(np.sum(np.exp(log_pi), axis=1, keepdims=True))

                z_enc = np.sum(mu, -1)

                observation_pred = vae.decode(np.array([z_enc]))[0]

            observation, reward, done, info = env.step(action)
            observation = config.adjust_obs(observation)
            reward_sum += reward

            ## rendering
            ax_org.imshow(observation)
            # env.render()
            # todo: add title etc
            if show_vae:
                z_enc_vae = vae.encode(np.expand_dims(observation, 0))[0]
                if t < 10:
                    z_enc = z_enc_vae.copy()
                reconstruction = vae.decode(np.array([z_enc_vae]))[0]

                # reconstruction = vae.predict(np.expand_dims(observation, 0))[0]
                ax_vae.imshow(reconstruction)
                # todo: add title etc
            if show_simulation:
                ax_sim.imshow(observation_pred)

            plt.show()
            plt.pause(0.00001)
            # plt.show(False)
            # plt.draw()

            # fig.canvas.draw()
            # fig.canvas.flush_events()

            print('ep {:3d}, t {:5d} '.format(episode, t), action)
            if t >= 300:
                done = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Human interface')

    parser.add_argument('--show', type=str, default='both', choices=['vae', 'sim', 'both'], help='Show the output of the VAE?')
    parser.add_argument('--controller', type=str, default='random', choices=['human', 'agent', 'random'],
                        help="How should actions be selected? ['human', 'agent', 'random']")
    args = parser.parse_args()
    print(args, '\n')

    main(args)
