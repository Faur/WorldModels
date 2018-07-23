import argparse
import numpy as np
import matplotlib.pyplot as plt

import gym

import config
from env import make_env
from vae.arch import VAE
from rnn.arch import RNN


GAUSSIAN_MIXTURES = 5
Z_DIM = 32
NP_RANDOM, SEED = gym.utils.seeding.np_random()


def get_mixture_coef(z_pred):
    log_pi, mu, log_sigma = np.split(z_pred, 3, 1)
    log_pi = log_pi - np.log(np.sum(np.exp(log_pi), axis=1, keepdims=True))
    return log_pi, mu, log_sigma


def sample_z(mu, log_sigma):
    z = mu + (np.exp(log_sigma)) * NP_RANDOM.randn(*log_sigma.shape) * 0
    return z


def get_pi_idx(x, pdf):
    # samples from a categorial distribution
    N = pdf.size
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
            return i
    random_value = np.random.randint(N)
    # print('error with sampling ensemble, returning random', random_value)
    return random_value


def sample_next_mdn_output(rnn, obs):
    ## TODO: What is obs dim? is it step wise?

    d = GAUSSIAN_MIXTURES * Z_DIM
    # y_pred = rnn.model.predict(np.array([[obs]]))[0][0]
    y_pred, h, c = rnn.forward.predict([np.array([[obs]])] + rnn.lstm.states)
    rnn.lstm.states = [h, c]

    z_pred = y_pred[:, :3 * d]
    rew_pred = y_pred[:, -1]
    z_pred = np.reshape(z_pred, [-1, GAUSSIAN_MIXTURES * 3])

    log_pi, mu, log_sigma = get_mixture_coef(z_pred)

    chosen_log_pi = np.zeros(Z_DIM)
    chosen_mu = np.zeros(Z_DIM)
    chosen_log_sigma = np.zeros(Z_DIM)

    # adjust temperatures
    logmix2 = np.copy(log_pi)
    logmix2 -= logmix2.max()
    logmix2 = np.exp(logmix2)
    logmix2 /= logmix2.sum(axis=1).reshape(Z_DIM, 1)

    for j in range(Z_DIM):
        idx = get_pi_idx(NP_RANDOM.rand(), logmix2[j])
        idx = 0
        chosen_log_pi[j] = idx
        chosen_mu[j] = mu[j, idx]
        chosen_log_sigma[j] = log_sigma[j, idx]

    next_z = sample_z(chosen_mu, chosen_log_sigma)

    return next_z, chosen_mu, chosen_log_sigma, chosen_log_pi


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

            # rnn_hidden = np.zeros([1, rnn.hidden_units])
            # rnn_cell = np.zeros([1, rnn.hidden_units])
            rnn.lstm.states = [np.zeros([1, 256]), np.zeros([1, 256])]

            # print(z_enc.shape)
            # print()
            # print(z_enc)

        t = 0
        while not done:
            t += 1
            action = config.select_action(env, observation, t, controller)

            if show_simulation:
                rnn_input = np.concatenate([z_enc, np.array(action)])


                next_z, chosen_mu, chosen_log_sigma, chosen_log_pi = sample_next_mdn_output(rnn, rnn_input)
                observation_pred = vae.decode(np.array([next_z]))[0]

                # rnn_input = np.concatenate([z_enc, np.expand_dims(action, 0)], axis=1)
                # rnn_input = np.expand_dims(rnn_input, 0)
                # rnn_input = [
                #     np.array([[np.concatenate([z_enc, action])]]),
                #     rnn_hidden,
                #     rnn_cell
                # ]
                # rnn_pred, rnn_hidden, rnn_cell = rnn.forward.predict(rnn_input)
                # reward_pred = rnn_pred[0,-1]
                # reward = reward_pred
                # mixture_coef = rnn_pred[0,:-1]
                # mixture_coef = mixture_coef.reshape([-1, rnn.gaussian_mixtures*3])
                # log_pi, mu, log_sigma = np.split(mixture_coef, 3, -1)
                # log_pi = log_pi - np.log(np.sum(np.exp(log_pi), axis=1, keepdims=True))
                # # z_enc = np.sum(mu, -1)
                # z_enc = np.random.normal(mu, np.exp(log_sigma))
                # z_enc = np.sum(z_enc, -1)
                # observation_pred = vae.decode(np.array([z_enc]))[0]

            observation, reward, done, info = env.step(action)
            observation = config.adjust_obs(observation)
            reward_sum += reward

            ## rendering
            fig.suptitle("{:3d}".format(t), fontsize=14, fontweight='bold')
            ax_org.imshow(observation)
            # env.render()
            # todo: add title etc
            if show_vae:
                z_enc_vae = vae.encode(np.expand_dims(observation, 0))[0]
                if t < 100:
                    z_enc = z_enc_vae.copy()
                reconstruction = vae.decode(np.array([z_enc_vae]))[0]

                # reconstruction = vae.predict(np.expand_dims(observation, 0))[0]
                ax_vae.imshow(reconstruction)
                # todo: add title etc
            if show_simulation:
                ax_sim.imshow(observation_pred)

            plt.tight_layout()
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
