#python 02_train_vae.py --new_model

from vae.arch import VAE
import argparse
import numpy as np
import config

def main(args):
  batch_num = args.batch_num

  vae = VAE()

  try:
    vae.set_weights('./vae/weights.h5')
  except:
    print("Either set --new_model or ensure ./vae/weights.h5 exists")
    raise

  """ Train on batch 0, then on batch {0, 1}, then on batches {0,1,2} ..."""
  print('Building batch {}...'.format(batch_num))
  first_item = True

  for env_name in config.train_envs:
    try:
      new_data = np.load('./data/obs_data_' + env_name + '_' + str(batch_num) + '.npy')
      data = new_data
      first_item = False
      print('Found {}...current data size = {} episodes'.format(env_name, len(data)))
    except:
      pass

  if first_item == False: # i.e. data has been found for this batch number
    data = np.array([item for obs in data for item in obs])
    # vae.train(data)

    # import keras
    # keras.utils.vis_utils.plot_model(vae.model, to_file='vis_vae_model.png', show_shapes=True)
    vae.test(data)
  else:
    print('no data found for batch number {}'.format(batch_num))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--batch_num', type=int, default = 9, help='The start batch number')
  args = parser.parse_args()

  main(args)
