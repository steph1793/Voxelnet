import tensorflow as tf
import argparse
import os

from model_helper.training_helper import train_epochs
from data import Data_helper
from model import Model
from config import cfg

"""gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6136),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6136)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

"""
def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--strategy", default="all", help="Distributed or centralized training (options : all for all available gpus or string of gpus numbers separated by commas like '0,1')", type=str)

  parser.add_argument("--batch_size", default=2, help="Total batch_size", type=int)
  parser.add_argument("--n_epochs", default=160, help="Total Number of epochs to train the model", type=int)

  parser.add_argument("--learning_rate", default=1e-3, help="Learning rate", type=float)
  parser.add_argument("--small_addon_for_BCE", default=1e-6, help="Small addon to add to the binary asymetric cross entropy for the loss", type=float)
  parser.add_argument("--max_gradient_norm", default=5.0, help="Maximum gradient norm to clip into", type=float)
  parser.add_argument("--alpha_bce", default=1.5, help="Alpha BCE", type=float)
  parser.add_argument("--beta_bce", default=1.0, help="Beta BCE", type=float)
  parser.add_argument("--huber_delta", default=3.0, help="Huber loss epsilon", type=float)

  parser.add_argument("--dump_vis", default="no", help="Boolean to save the viz (images, heatmaps, birdviews) of the dump test (yes or no)", type=str2bool)
  parser.add_argument("--data_root_dir", default="", help="Data root directory", type=str)
  parser.add_argument("--model_dir", default="", help="Directory to save the models, the viz and the logs", type=str)
  parser.add_argument("--model_name", default="", help="Model Name", type=str)
  parser.add_argument("--dump_test_interval", default=-1, help="Launch a dump test every n epochs", type=int)
  parser.add_argument("--summary_interval",default=-1, help="Save the training summary every n steps", type=int)
  parser.add_argument("--summary_val_interval", default=-1, help="Run an evaluation of the model and save the summary  every n steps and", type=int)
  parser.add_argument("--summary_flush_interval", default=-1, help="Flush the summaries every n steps", type=int)
  parser.add_argument("--ckpt_max_keep", default=11, help="Max checkpoints to keep", type=int) 


  args = parser.parse_args()
  params = vars(args)
  params["mode"] = "train"
  cfg["DATA_DIR"] = params["data_root_dir"]
  cfg["CALIB_DIR"] = os.path.join(cfg["DATA_DIR"], "training/calib")

  # Strategy Management         #####################################################
  logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  n_gpus = len(logical_gpus)
  if n_gpus==1:
    strategy = tf.distribute.OneDeviceStrategy("gpu:0")
    print("Using One Device Strategy")
  else:
    if params["strategy"].lower() == "all":
      strategy = tf.distribute.MirroredStrategy()
      print("Using mirrored strategy with gpus", logical_gpus)
    else:
      l = ["gpu:{}".format(int(i)) for i in params["strategy"].split(",") if i.isdigit()]
      assert l, "There is no gpus {} available".format(params["strategy"].split(","))
      if len(l)==1:
        strategy = tf.distribute.OneDeviceStrategy("gpu:0")
        print("Using One Device Strategy")
      else:
        strategy = tf.distribute.MirroredStrategy(l)
        print("Using mirrored strategy with gpus", l)

  
  ############ Datasets #################################################################
  print("Datasets creation (training dataset, sample_test dataset, validation and dump_test dataset)")
  with strategy.scope():
    train_batcher = Data_helper(cfg, params, 16, "train", is_aug_data=True, create_anchors=True, strategy=strategy)
    rand_test_batcher = Data_helper(cfg, params, 1, "sample_test", is_aug_data=False, create_anchors=False, strategy=strategy)
    val_batcher = Data_helper(cfg, params, 16, "eval", is_aug_data=False, create_anchors=False, strategy=strategy)


  ###### Model #################################################################################
  print("Model creation ...")
  with strategy.scope():
    model = Model(cfg, params, strategy)
    model.add_loss_()
    
  
  ####### Checkpoint manager #######################################################
  print("Building the checkpoint Manager ...")
  with strategy.scope():
    checkpoint_dir = os.path.join(params["model_dir"], params["model_name"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0, trainable=False), voxelnet=model, epoch=tf.Variable(0, trainable=False))
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir , max_to_keep=params["ckpt_max_keep"])

    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
      print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
      print("Initialized from scratch.")


  print("Start training : ")
  with strategy.scope():
    model.add_optimizer_(ckpt.epoch)
    train_epochs( model, train_batcher, rand_test_batcher, val_batcher,  params, cfg, ckpt, ckpt_manager, strategy)

if __name__ =="__main__":
  main()