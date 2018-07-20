#python 03_generate_rnn_data.py

from vae.arch import VAE
import argparse
import config
import numpy as np

def encode_batch(vae, obs_data, action_data, rew_data, done_data):

    # rnn_input = []
    # rnn_output = []
    mu_out = []
    log_var_out = []
    action_out = []
    reward_out = []
    done_out = []

    initial_mu = []
    initial_log_var = []

    for obs, act, rew, done in zip(obs_data, action_data, rew_data, done_data):   

        rew = np.where(rew>0, 1, 0)
        done = done.astype(int)  

        mu, log_var = vae.encoder_mu_log_var.predict(np.array(obs))
        
        initial_mu.append(mu[0, :])
        initial_log_var.append(log_var[0, :])

        mu_out.append(mu)
        log_var_out.append(log_var)
        action_out.append(act)
        reward_out.append(rew)
        done_out.append(done)

    initial_mu = np.array(initial_mu)
    initial_log_var = np.array(initial_log_var)

    mu_out = np.array(mu_out)
    log_var_out = np.array(log_var_out)
    action_out = np.array(action_out)
    reward_out = np.array(reward_out)
    done_out = np.array(done_out)

    return (mu_out, log_var_out, action_out, reward_out, done_out, initial_mu, initial_log_var)



def main(args):

    path = args.path
    start_batch = args.start_batch
    max_batch = args.max_batch

    vae = VAE()

    try:
      vae.set_weights('./vae/weights.h5')
    except:
      print("./vae/weights.h5 does not exist - ensure you have run 02_train_vae.py first")
      raise

    for batch_num in range(start_batch, max_batch + 1):
      first_item = True
      print('Generating batch {}...'.format(batch_num))

      for env_name in config.train_envs:
        try:
          new_obs_data = np.load(path + '/obs_data_' + env_name + '_'  + str(batch_num) + '.npy')
          new_action_data= np.load(path + '/action_data_' + env_name + '_'  + str(batch_num) + '.npy')
          new_reward_data = np.load(path + '/reward_data_' + env_name + '_'  + str(batch_num) + '.npy')
          new_done_data = np.load(path + '/done_data_' + env_name + '_'  + str(batch_num) + '.npy')

          if first_item:
            obs_data = new_obs_data
            action_data = new_action_data
            rew_data = new_reward_data
            done_data = new_done_data
            first_item = False
          else:
            obs_data = np.concatenate([obs_data, new_obs_data])
            action_data = np.concatenate([action_data, new_action_data])
            rew_data = np.concatenate([rew_data, new_reward_data])
            done_data = np.concatenate([done_data, new_done_data])

          print('Found {}...current data size = {} episodes'.format(env_name, len(obs_data)))
        except:
          pass
      
      if first_item == False:
        mu, log_var, action, reward, done, initial_mu, initial_log_var = encode_batch(vae, obs_data, action_data, rew_data, done_data)
        
        np.save(path + '/mu_' + str(batch_num), mu)
        np.save(path + '/log_var_' + str(batch_num), log_var)
        np.save(path + '/action_' + str(batch_num), action)
        np.save(path + '/reward_' + str(batch_num), reward)
        np.save(path + '/done_' + str(batch_num), done)

        np.save(path + '/initial_mu_' + str(batch_num), initial_mu)
        np.save(path + '/initial_log_var_' + str(batch_num), initial_log_var)
      else:
        print('no data found for batch number {}'.format(batch_num))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Generate RNN data'))
  parser.add_argument('--path', type=str, default='./data', help='folder to save in')
  parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
  parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')

  args = parser.parse_args()

  main(args)
