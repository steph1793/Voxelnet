from utils.utils import cal_anchors, process_pointcloud, cal_rpn_target
import tensorflow as tf
import glob
import random
import numpy as np
import time
from queue import Queue
from threading import Thread
import threading
import os
from utils.aug_data import aug_data

class thread_safe_generator(object):
  def __init__(self, gen):
      self.gen = gen
      self.lock = threading.Lock()

  def __next__(self):
      with self.lock:
          return next(self.gen)

class Data_helper:

  def __init__(self, cfg, params, buffer_size, mode, is_aug_data, create_anchors=False,  strategy=None):
    self.cfg = cfg
    self.params = params
    self.mode = mode
    data_d = "training" if mode == "train" else "testing" if mode =="test" else "validation"
    if mode != "test":
      label_tags = [os.path.basename(a).split(".")[0] for a in glob.glob(os.path.join(cfg.DATA_DIR, data_d, "label_2/*.txt"))]
    img_tags = [os.path.basename(a).split(".")[0] for a in glob.glob(os.path.join(cfg.DATA_DIR, data_d, "image_2/*.png"))]
    lidar_tags = [os.path.basename(a).split(".")[0] for a in glob.glob(os.path.join(cfg.DATA_DIR, data_d, "velodyne/*.bin"))]

    if mode != "test":
      assert label_tags and img_tags and lidar_tags, "One of the three (label_2, image_2, velodyne) folders is empty, Data folder must not be empty"
      assert not set(label_tags).symmetric_difference(set(img_tags)) and not set(img_tags).symmetric_difference(set(lidar_tags)),\
      "Must have equivalent tags in image_2, label_2 and velodyne dirs, check those files"
    else:
      assert  img_tags and lidar_tags, "One of the three (image_2, velodyne) folders is empty, Data folder must not be empty"
      assert  not set(img_tags).symmetric_difference(set(lidar_tags)),\
      "Must have equivalent tags in image_2, velodyne dirs, check those files"
    
    self.tags = lidar_tags
    self.num_examples = len(lidar_tags)
    if create_anchors:
      pass
    self.anchors = cal_anchors(cfg)

    #self.tag_gen = thread_safe_generator(self.tag_generator(params["n_epochs"]))

    """self.ex_queue = Queue(params["ex_buffer_size"])
    self.launch_fillers(params["num_threads"], mode, is_aug_data)
    while self.ex_queue.qsize()<params["ex_buffer_size"]:
      time.sleep(2)"""

    self.batcher = self.batch_dataset( params["batch_size"], mode,is_aug_data, buffer_size, cfg, strategy)
    self.batch_iter = iter(self.batcher)
    
  """def tag_generator(self):
    random.shuffle(self.tags)
    for ind in self.tags:
      yield ind"""


  def fill_examples_queue(self, cfg, mode, is_aug_data=False):
    data_d = "training" if mode == "train" else "testing" if mode =="test" else "validation"
    img_dir = "{}/{}/image_2".format(cfg.DATA_DIR, data_d)
    labels_dir = "{}/{}/label_2".format(cfg.DATA_DIR, data_d)
    pc_dir = "{}/{}/velodyne".format(cfg.DATA_DIR, data_d)
    
    if mode in ["train", "sample_test"]:
      random.shuffle(self.tags)
    else:
      print("sort data")
      sorted(self.tags)
    for index in self.tags:
      #index = next(self.tag_gen)
      dic = {}
      if is_aug_data:
        dic = aug_data(index, os.path.join(cfg.DATA_DIR, data_d))
      else:
        pc = np.fromfile("%s/%06d.bin" % (pc_dir, int(index)), dtype=np.float32).reshape(-1,4) 
        if mode == "test":
          dic["lidar"] = pc
          dic["labels"] = []
        elif mode == "sample_test" or mode== "eval":
          dic["lidar"] = pc
          dic["labels"] = np.array([line.strip() for line in open("%s/%06d.txt" % (labels_dir, int(index)) , 'r').readlines()])
        else:
          dic["lidar"] = 0
          dic["labels"] = np.array([line.strip() for line in open("%s/%06d.txt" % (labels_dir, int(index)) , 'r').readlines()])
        dic["num_points"] = len(pc)

        if mode == "train":
          dic["img"] = 0
        else:
          img = tf.io.read_file("%s/%06d.png" % (img_dir, int(index)))
          img = tf.image.decode_png(img, channels=cfg.IMG_CHANNEL)
          img = tf.image.convert_image_dtype(img, tf.float32)
          dic["img"] = tf.image.resize(img, [cfg.IMG_HEIGHT, cfg.IMG_WIDTH])

        dic["tag"] = "%06d" % int(index)
    
        
        dic.update(process_pointcloud(pc, cfg))

      if mode in ["train", "eval", "sample_test"]:
        dic["pos_equal_one"], dic["neg_equal_one"], dic["targets"]= cal_rpn_target(dic["labels"][np.newaxis, ...].astype(str), 
                                                                                      cfg.MAP_SHAPE , 
                                                                                      self.anchors, 
                                                                                      cfg.DETECT_OBJECT, 
                                                                                      'lidar')
      
        dic["pos_equal_one_reg"] = np.concatenate(
                [np.tile( dic["pos_equal_one"][..., [0]], 7), np.tile( dic["pos_equal_one"][..., [1]], 7)], axis=-1)[0] #we index to 0 because we added a batch dimension
        dic["pos_equal_one_sum"] = np.clip(np.sum(dic["pos_equal_one"], axis=(
                1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)[0]
        dic["neg_equal_one_sum"] = np.clip(np.sum(dic["neg_equal_one"], axis=(
                1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)[0]

        dic["pos_equal_one"], dic["neg_equal_one"], dic["targets"] = dic["pos_equal_one"][0], dic["neg_equal_one"][0], dic["targets"][0]
      else:
        dic["pos_equal_one"], dic["neg_equal_one"], dic["targets"] = 0,0,0
        dic["pos_equal_one_reg"], dic["pos_equal_one_sum"], dic["neg_equal_one_sum"] = 0,0,0
      
      yield dic
      #self.ex_queue.put(dic)

  """
  def launch_fillers(self, num_threads, mode, is_aug_data):
    self.elements_queue_threads = []
    for i in range(num_threads):
      self.elements_queue_threads.append(Thread(target=self.fill_examples_queue, args=(self.cfg, mode,is_aug_data,)))
      self.elements_queue_threads[-1].setDaemon(True)
      self.elements_queue_threads[-1].start()

  
  def example_generator(self):
    once_empty = False
    while True:
      if self.ex_queue.qsize() == 0 and not once_empty:
        time.sleep(10)
        once_empty = True
        continue
      elif self.ex_queue.qsize() != 0:
        once_empty=False
        ex = self.ex_queue.get()
        yield ex
      else:
        break
  """


  def batch_dataset(self, batch_size, mode , is_aug_data, buffer_size, cfg, strategy):
    dataset = tf.data.Dataset.from_generator(lambda: self.fill_examples_queue(self.cfg, mode,is_aug_data), 
                                            output_types={
                                                "img" : tf.float32,
                                                "labels" : tf.string,
                                                "tag" : tf.string,
                                                "feature_buffer" : tf.float32,
                                                "coordinate_buffer" : tf.int32,
                                                "number_buffer" : tf.int32,
                                                "lidar" : tf.float32,
                                                "num_points" : tf.int32,
                                                "pos_equal_one": tf.float32,
                                                "neg_equal_one" : tf.float32,
                                                "targets" : tf.float32,
                                                "pos_equal_one_reg" : tf.float32,
                                                "pos_equal_one_sum" : tf.float32,
                                                "neg_equal_one_sum" : tf.float32
                                            },
                                            output_shapes={
                                                "img" : [cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNEL] if mode!="train" else [],
                                                "labels" : [None],
                                                "tag" : [],
                                                "feature_buffer" : [None, cfg.MAX_POINT_NUMBER, 7],
                                                "coordinate_buffer" : [None, 3],
                                                "number_buffer" : [None],
                                                "lidar" : [None, 4] if "test" in mode or mode == "eval" else [],
                                                "num_points" : [],
                                                "pos_equal_one": [*cfg.MAP_SHAPE, cfg.NUM_ANCHORS_PER_CELL] if mode in ["train", "eval", "sample_test"] else [] ,
                                                "neg_equal_one" : [*cfg.MAP_SHAPE, cfg.NUM_ANCHORS_PER_CELL] if mode in ["train", "eval", "sample_test"] else [],
                                                "targets" : [*cfg.MAP_SHAPE, 7*cfg.NUM_ANCHORS_PER_CELL] if mode in ["train", "eval", "sample_test"] else [],
                                                "pos_equal_one_reg" : [*cfg.MAP_SHAPE, 7*cfg.NUM_ANCHORS_PER_CELL] if mode in ["train", "eval", "sample_test"] else [],
                                                "pos_equal_one_sum" : [1,1,1] if mode in ["train", "eval", "sample_test"] else [],
                                                "neg_equal_one_sum" : [1,1,1] if mode in ["train", "eval", "sample_test"] else []
                                            })
    
    dataset = dataset.padded_batch(batch_size, 
                                  padded_shapes = {
                                      "img" : [cfg.IMG_HEIGHT,cfg.IMG_WIDTH, cfg.IMG_CHANNEL] if mode!="train" else [],
                                      "labels" : [None],
                                      "tag" : [],
                                      "feature_buffer" : [None, cfg.MAX_POINT_NUMBER, 7],
                                      "coordinate_buffer" : [None, 3],
                                      "number_buffer" : [None],
                                      "lidar" : [None, 4] if "test" in mode or mode == "eval" else [],
                                      "num_points": [],
                                      "pos_equal_one": [*cfg.MAP_SHAPE, cfg.NUM_ANCHORS_PER_CELL] if mode in ["train", "eval", "sample_test"] else [] ,
                                      "neg_equal_one" : [*cfg.MAP_SHAPE, cfg.NUM_ANCHORS_PER_CELL] if mode in ["train", "eval", "sample_test"] else [],
                                      "targets" : [*cfg.MAP_SHAPE, 7*cfg.NUM_ANCHORS_PER_CELL] if mode in ["train", "eval", "sample_test"] else [],
                                      "pos_equal_one_reg" : [*cfg.MAP_SHAPE, 7*cfg.NUM_ANCHORS_PER_CELL] if mode in ["train", "eval", "sample_test"] else [],
                                      "pos_equal_one_sum" : [1,1,1] if mode in ["train", "eval", "sample_test"] else [],
                                      "neg_equal_one_sum" : [1,1,1] if mode in ["train", "eval", "sample_test"] else []
                                  }, padding_values = {
                                      'img' : 0.0,
                                      'labels' : b"",
                                      'tag' : b"",
                                      "feature_buffer" : 0.0,
                                      "coordinate_buffer" : 0,
                                      "number_buffer" : 0,
                                      "lidar":0.,
                                      "num_points":0,
                                      "pos_equal_one": 0. ,
                                      "neg_equal_one" : 0.,
                                      "targets" : 0.,
                                      "pos_equal_one_reg" : 0.,
                                      "pos_equal_one_sum" : 0.,
                                      "neg_equal_one_sum" : 0.
                                  })
    
    def update_dataset(batch):
      batch_idx = tf.range(0, tf.shape(batch["coordinate_buffer"])[0], 1)
      batch_idx = tf.expand_dims(tf.expand_dims(batch_idx, axis=-1), axis=-1)
      batch_idx = tf.tile(batch_idx, [1, tf.shape(batch["coordinate_buffer"])[1], 1])
      batch["coordinate_buffer"] = tf.concat([batch_idx, batch["coordinate_buffer"]], axis=-1)
      return batch

    dataset = dataset.map(update_dataset)
    if mode in ["train", "sample_test"]:
      dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size)
    if type(strategy) != type(None):
      print("Distributed dataset !")
      dataset = strategy.experimental_distribute_dataset(dataset)
    
    return dataset


  def __iter__(self):
    return self.batch_iter
  def __next__(self):
    return next(self.batch_iter)
      