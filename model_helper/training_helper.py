import time
from utils.utils import box3d_to_label
import numpy as np
import tensorflow as tf
from termcolor import colored
import os
import cv2

from model_helper.test_helper import predict_step

def train_summary(writer, metrics):
  with writer.as_default():
    tf.summary.scalar('train/loss', metrics[0])
    tf.summary.scalar('train/reg_loss', metrics[1])
    tf.summary.scalar('train/cls_loss', metrics[2])
    tf.summary.scalar('train/cls_pos_loss', metrics[3])
    tf.summary.scalar('train/cls_neg_loss', metrics[4])
    [tf.summary.histogram(each.name, each) for each in metrics[5]]

def val_summary(writer, metrics):
  with writer.as_default():
    tf.summary.scalar('validate/loss', metrics[0])
    tf.summary.scalar('validate/reg_loss', metrics[1])
    tf.summary.scalar('validate/cls_loss', metrics[2])
    tf.summary.scalar('validate/cls_pos_loss', metrics[3])
    tf.summary.scalar('validate/cls_neg_loss', metrics[4])

def pred_summary(writer, metrics):
  with writer.as_default():
    tf.summary.image("predict/bird_view_lidar", metrics["bird_view"] )
    tf.summary.image("predict/bird_view_heatmap",metrics["heatmap"])
    tf.summary.image("predict/front_view_rgb",metrics["front_image"])

def epoch_counter(current_step, num_batches):
  return int(current_step//num_batches) +1

def train_epochs( model, train_batcher, rand_test_batcher, val_batcher,  params, cfg,
                 ckpt, ckpt_manager, strategy):
  
  @tf.function
  def distributed_train_step():

    batch = next(train_batcher)  
    per_replica_losses = strategy.experimental_run_v2(model.train_step,
                                                      args=(batch["feature_buffer"], 
                                                            batch["coordinate_buffer"],
                                                            batch["targets"], 
                                                            batch["pos_equal_one"], 
                                                            batch["pos_equal_one_reg"],
                                                            batch["pos_equal_one_sum"],
                                                            batch["neg_equal_one"], 
                                                            batch["neg_equal_one_sum"]))
    return [strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss,
                          axis=None) for per_replica_loss in per_replica_losses]

  @tf.function
  def distributed_validate_step():
    batch = next(rand_test_batcher)
    per_replica_losses = strategy.experimental_run_v2(model.train_step,
                                                    args=(batch["feature_buffer"], 
                                                          batch["coordinate_buffer"],
                                                          batch["targets"], 
                                                          batch["pos_equal_one"], 
                                                          batch["pos_equal_one_reg"],
                                                          batch["pos_equal_one_sum"],
                                                          batch["neg_equal_one"], 
                                                          batch["neg_equal_one_sum"]))
    return [strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss,
                          axis=None) for per_replica_loss in per_replica_losses], batch

  dump_vis = params["dump_vis"] # bool
  kitti_eval_script = cfg.KITTY_EVAL_SCRIPT

  sum_logdir = os.path.join(params["model_dir"], params["model_name"], "train_log/summary_logdir") 
  logdir = os.path.join(params["model_dir"], params["model_name"], "train_log/logdir")
  dump_test_logdir = os.path.join(params["model_dir"], params["model_name"], "train_log/dump_test_logdir") 

  os.makedirs(sum_logdir, exist_ok=True)
  os.makedirs(logdir, exist_ok=True)
  os.makedirs(dump_test_logdir, exist_ok=True)

  step = 1

  dump_interval = params["dump_test_interval"] # 10
  summary_interval = params["summary_interval"] # 5
  summary_val_interval = params["summary_val_interval"]  # 10
  summary_flush_interval = params["summary_flush_interval"]
  summary_writer = tf.summary.create_file_writer(sum_logdir)

  epoch = ckpt.epoch
  epoch.assign(epoch_counter(ckpt.step.numpy(), train_batcher.num_examples))
  
  try:
    while epoch.numpy() <= params["n_epochs"]:
      num_batches = train_batcher.num_examples//params["batch_size"]+(1 if train_batcher.num_examples%params["batch_size"]==1 else 0)
      for step in range(num_batches):

        epoch.assign(epoch_counter(ckpt.step.numpy(), num_batches))
        if epoch.numpy() > params["n_epochs"]:
          break
        
        global_step = ckpt.step.numpy()
        tf.summary.experimental.set_step(global_step)

        t0 = time.time()
        losses = distributed_train_step()
        t1 = time.time() - t0

        print('train: {} @ epoch:{}/{} global_step:{} loss: {} reg_loss: {} cls_loss: {} cls_pos_loss: {} cls_neg_loss: {} batch time: {:.4f}'.format(step+1, epoch.numpy(), params["n_epochs"], ckpt.step.numpy(), colored('{:.4f}'.format(losses[0]), "red"), colored('{:.4f}'.format(losses[1]), "magenta"), colored('{:.4f}'.format(losses[2]), "yellow"), colored('{:.4f}'.format(losses[3]), "blue"), colored('{:.4f}'.format(losses[4]), "cyan"),  t1))
        with open('{}/train.txt'.format(logdir), 'a') as f:
          f.write( 'train: {} @ epoch:{}/{} global_step:{} loss: {:.4f} reg_loss: {:.4f} cls_loss: {:.4f} cls_pos_loss: {:.4f} cls_neg_loss: {:.4f} batch time: {:.4f} \n'.format(step+1, epoch.numpy(), params["n_epochs"], ckpt.step.numpy(), losses[0], losses[1], losses[2], losses[3], losses[4], t1) )
                        
        if (step+1) % summary_interval == 0:
          train_summary(summary_writer, list(losses)+[model.trainable_variables])

        if (step+1) % summary_val_interval == 0:
          print("summary_val_interval now")

          ret, batch = distributed_validate_step()
          val_summary(summary_writer, ret)
          try:
            ret = predict_step( model, batch, train_batcher.anchors, cfg, params, summary=True)
            pred_summary(summary_writer, ret)
          except:
            print("prediction skipped due to error")

        if (step+1) % summary_flush_interval==0:
          summary_writer.flush()
        
        if global_step%train_batcher.num_examples==0:
          ckpt_manager.save(checkpoint_number=ckpt.step.numpy())
          print("Saved checkpoint for step {}".format(ckpt.step.numpy()))
          summary_writer.flush()

        ckpt.step.assign_add(1)

      # dump test data every 10 epochs
      
      if ( epoch.numpy()  ) % dump_interval == 0 :
        print("dump_test")
        # create output folder
        os.makedirs(os.path.join(dump_test_logdir, str(epoch.numpy())), exist_ok=True)
        os.makedirs(os.path.join(dump_test_logdir, str(epoch.numpy()), 'data'), exist_ok=True)
        if dump_vis:
          os.makedirs(os.path.join(dump_test_logdir, str(epoch.numpy()), 'vis'), exist_ok=True)
                      
        for eval_step, batch in enumerate(val_batcher.batcher):
          if dump_vis:
            res = predict_step(model, batch,  train_batcher.anchors, cfg, params, summary=False, vis=True)
            tags, results, front_images, bird_views, heatmaps = res["tag"], res["scores"], res["front_image"], res["bird_view"], res["heatmap"]
          else:
            res = predict_step( model, batch,  train_batcher.anchors, cfg, params, summary=False, vis=False)
            tags, results = res["tag"], res["scores"]
          for tag, result in zip(tags, results):
            of_path = os.path.join(dump_test_logdir, str(epoch.numpy()), 'data', tag + '.txt')
            with open(of_path, 'w+') as f:
              labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar')[0]
              for line in labels:
                f.write(line)
              print('write out {} objects to {}'.format(len(labels), tag))
          # dump visualizations
          if dump_vis:
            for tag, front_image, bird_view, heatmap in zip(tags, front_images, bird_views, heatmaps):
              front_img_path = os.path.join( dump_test_logdir, str(epoch.numpy()),'vis', tag + '_front.jpg'  )
              bird_view_path = os.path.join( dump_test_logdir, str(epoch.numpy()), 'vis', tag + '_bv.jpg'  )
              heatmap_path = os.path.join( dump_test_logdir, str(epoch.numpy()), 'vis', tag + '_heatmap.jpg'  )
              cv2.imwrite( front_img_path, front_image )
              cv2.imwrite( bird_view_path, bird_view )
              cv2.imwrite( heatmap_path, heatmap )
          
        
        # execute evaluation code
        #cmd_1 = "./"+kitti_eval_script
        #cmd_2 = os.path.join(cfg.DATA_DIR, "validation", "label_2")
        #cmd_3 = os.path.join( dump_test_logdir, str(epoch.numpy()) )
        #cmd_4 = os.path.join( dump_test_logdir, str(epoch.numpy()), 'log' )
        #os.system( " ".join( [cmd_1, cmd_2, cmd_3, cmd_4] ) ).read()
        
  except KeyboardInterrupt:
    ckpt_manager.save(checkpoint_number=ckpt.step.numpy())
    print("Saved checkpoint for step {}".format(ckpt.step.numpy()))
    summary_writer.flush()