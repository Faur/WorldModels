#python 04_train_rnn.py --new_model
#python 04_train_rnn.py --path ./data/new --max_batch 9

from rnn.arch import RNN
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.ion()


# LEARNING_RATE = 0.001
# MIN_LEARNING_RATE = 0.001


def random_batch(data_mu, data_logvar, data_action, data_rew, data_done, batch_size):
    N_data = len(data_mu)
    indices = np.random.permutation(N_data)[0:batch_size]

    mu = data_mu[indices]
    logvar = data_logvar[indices]
    action = data_action[indices]
    rew = data_rew[indices]
    done = data_done[indices]

    s = logvar.shape
    z = mu + np.exp(logvar/2.0) * np.random.randn(*s)

    rew = np.expand_dims(rew, axis=2)
    done = np.expand_dims(done, axis=2)

    return z, action, rew, done


def main(args):
    start_batch = args.start_batch
    max_batch = args.max_batch
    new_model = args.new_model
    path = args.path

    rnn = RNN() #learning_rate = LEARNING_RATE
    import keras
    keras.utils.vis_utils.plot_model(rnn.model, to_file='vis_rnn_model.png', show_shapes=True)

    if not new_model:
        try:
            rnn.set_weights('./rnn/weights.h5')
        except:
            print("Either set --new_model or ensure ./rnn/weights.h5 exists")
            raise

    for batch_num in range(start_batch, max_batch + 1):
        print('Building batch {}...'.format(batch_num))
        new_mu = np.load(path + '/mu_' + str(batch_num) + '.npy')
        new_log_var = np.load(path + '/log_var_' + str(batch_num) + '.npy')
        new_action = np.load(path + '/action_' + str(batch_num) + '.npy')
        new_reward = np.load(path + '/reward_' + str(batch_num) + '.npy')
        new_done = np.load(path + '/done_' + str(batch_num) + '.npy')

        if batch_num > start_batch:
            mu_data = np.concatenate([mu_data, new_mu])
            log_var_data = np.concatenate([log_var_data, new_log_var])
            action_data = np.concatenate([action_data, new_action])
            rew_data = np.concatenate([rew_data, new_reward])
            done_data = np.concatenate([done_data, new_done])
        else:
            # Run first time
            mu_data = new_mu
            log_var_data = new_log_var
            action_data = new_action
            rew_data = new_reward
            done_data = new_done

    for epoch in range(1, 1+rnn.epochs):
        print('EPOCH ' + str(epoch))

        z, action, rew,done = random_batch(mu_data, log_var_data, action_data, rew_data, done_data, rnn.batch_size)

        rnn_input = np.concatenate([z[:, :-1, :], action[:, :-1, :]], axis = 2)
        rnn_output = np.concatenate([z[:, 1:, :], rew[:, 1:, :]], axis = 2) #, done[:, 1:, :]

        if 1:
            from vae.arch import VAE
            vae = VAE()
            try:
                vae.set_weights('./vae/weights.h5')
            except:
                raise Exception("Either set --new_model or ensure ./vae/weights.h5 exists")
            fig, [ax_inp, ax_tgt] = plt.subplots(1, 2)
            for i in range(len(rnn_input)):
                fig.suptitle("{:3d}".format(i), fontsize=14, fontweight='bold')

                ax_inp.imshow(vae.decode([[rnn_input[epoch, i, :32]]])[0])
                ax_inp.set_title('input')
                ax_tgt.imshow(vae.decode([[rnn_output[epoch, i, :32]]])[0])
                ax_tgt.set_title('target')

                plt.tight_layout()
                plt.show()
                plt.pause(0.00001)

        if epoch == 0:
            np.save(path + '/rnn_input.npy', rnn_input)
            np.save(path + '/rnn_output.npy', rnn_output)

        rnn.train(rnn_input, rnn_output)

        if epoch % 10 == 0:
            rnn.model.save_weights('./rnn/weights.h5')

    rnn.model.save_weights('./rnn/weights.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train RNN'))
    parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
    parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')
    parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
    parser.add_argument('--path', type=str, default='./data', help='folder to save in')

    args = parser.parse_args()
    print(args)

    main(args)
