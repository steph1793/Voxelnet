import tensorflow as tf
import argparse
import os

from model_helper.test_helper import predict
from data import Data_helper
from model import Model
from config import cfg

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

  parser.add_argument("--dump_vis", default="no", help="Boolean to save the viz (images, heatmaps, birdviews) of the dump test (yes or no)", type=str2bool)
  parser.add_argument("--data_root_dir", default="", help="Data root directory", type=str)
  parser.add_argument("--dataset_to_test", default="", help="Dataset to use for the predictions (validation or testing)", type=str)
  parser.add_argument("--model_dir", default="", help="Directory to save the models, the viz and the logs", type=str)
  parser.add_argument("--model_name", default="", help="Model Name", type=str)
  parser.add_argument("--ckpt_name", default="", help="Checkpoint to evaluate name, if empty uses the latest checkpoint", type=str) 

  args = parser.parse_args()
  params = vars(args)
  assert params["dataset_to_test"] in ["testing", "validation"]
  params["mode"] = "test"
  cfg["DATA_DIR"] = params["data_root_dir"]
  cfg["CALIB_DIR"] = os.path.join(cfg["DATA_DIR"], "training" if params["dataset_to_test"]=="validation" else "testing", "calib")

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
    test_batcher = Data_helper(cfg, params, 16, ("test"  if params["dataset_to_test"]=="testing" else "eval"), is_aug_data=False, create_anchors=True, strategy=strategy)
    


  ###### Model #################################################################################
  print("Model creation ...")
  with strategy.scope():
    model = Model(cfg, params, strategy)
    #model.add_loss_()
    
  
  ####### Checkpoint manager #######################################################
  print("Building the checkpoint Manager ...")
  with strategy.scope():
    checkpoint_dir = os.path.join(params["model_dir"], params["model_name"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0, trainable=False), voxelnet=model, epoch=tf.Variable(0, trainable=False))
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir , max_to_keep=1)


  assert ckpt_manager.latest_checkpoint, "there is no model. Launch the training first"
  if not params["ckpt_name"]:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
  else:
    p = os.path.join(params["model_dir"], params["model_name"], "checkpoints", params["ckpt_name"])
    ckpt.restore(p)
    print("Restored from {}".format(p))

  predict(strategy, model, test_batcher,  params, cfg)

if __name__ =="__main__":
  main()