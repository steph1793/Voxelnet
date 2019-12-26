import tensorflow as tf
import numpy as np
import os
from model_helper.loss_optimizer_helper import Loss, Optimizer
from utils.utils import delta_to_boxes3d, corner_to_standup_box2d, center_to_corner_box2d, load_calib, draw_lidar_box3d_on_image
from utils.utils import lidar_to_bird_view_img, draw_lidar_box3d_on_birdview, label_to_gt_box3d
from utils.colorize import colorize

class VFE_Layer(tf.keras.layers.Layer):
  """
    A VFE layer class
    Args : 
      c_out : int, the dimension of the output after VFE, must be even
  """
  def __init__(self, c_out):
    super(VFE_Layer, self).__init__()
    self.units = c_out//2
    self.fcn = tf.keras.layers.Dense(self.units, activation="relu")
    self.bn = tf.keras.layers.BatchNormalization(trainable=True)


  def call(self, input, mask, training=False):
    """
      Call method of the class
      Args:
        input : Tensor (4D tensor in our case), [Batch_size, max_num_voxels, max_num_pts, out_dim]
        (out_dim = 7, at the beginning of the network)
      Returns:
        output : Tensor with the same shape as input, except the last dim which is c_out
    """
    fcn_out = self.bn(self.fcn(input), training=training)
    max_pool = tf.reduce_max(fcn_out, axis=2,  keepdims=True) # [batch_size, max_num_voxels, 1, out_dim//2]
    tiled_max_pool = tf.tile(max_pool, [1,1,tf.shape(fcn_out)[2],1]) # [batch_size, max_num_voxels, max_num_pts, out_dim//2]
    output = tf.concat([fcn_out, tiled_max_pool], axis=-1) # [batch_size, max_num_voxels, max_num_pts, out_dim//2]
    mask = tf.tile(mask, [1,1,1, 2*self.units])
    return tf.multiply(output, tf.cast(mask, tf.float32))


class VFE_Block(tf.keras.layers.Layer):
  """
    VFE_block class, made of VFE layers
    
    Args:
      vfe_out_dims : n-integer list made of the output dimensions of VFEs, each dimension must be even
      final_dim : int32, dimension of the last Dense layer after VFEs
      sparse_shape : 3-list, int32, dimensions of the sparse voxels space // ex : [10, 400,352] 
  """
  def __init__(self, vfe_out_dims, final_dim, sparse_shape):
    super(VFE_Block, self).__init__()

    self.vfe_out_dims = vfe_out_dims
    self.final_dim = final_dim
    self.sparse_shape = sparse_shape

    self.VFEs = [VFE_Layer(dim) for dim in vfe_out_dims]
    self.final_fcn = tf.keras.layers.Dense(self.final_dim, activation="relu")



  def call(self, input, voxel_coor_buffer, shape, training=False):
    """
      call Method
      Args:
        input : 4D tensor, of type float32, [batch_size, K, T, 7]
        voxel_coor_buffer : 2D tensor , int32 of dimension [batch_size, 4]
        training : (optional), boolean 
      Returns:
        output : 5-D tensor, [batch_size, channels, Depth, Height, Width]
    """

    vfe_out = input

    # create a mask for the sparce space
    mask = tf.not_equal(tf.reduce_max(input, axis=-1, keepdims=True), 0) # [batch_size, max_num_voxels, max_num_pts, 1]
    
    for i, vfe in enumerate(self.VFEs):
      vfe_out = vfe(vfe_out, mask, training=training) # [batch_size, max_num_voxels, max_num_pts, vfe_out_dims[i] ]
    
    output = self.final_fcn(vfe_out) # [batch_size, max_num_voxels, max_num_pts, final_dim]
    output = tf.reduce_max(output, axis=2) # [batch_size, max_num_voxels, final_dim]

    # Voxels Sparse representation [batch_size, Depth, Height, Width, channels]
    output = tf.scatter_nd(indices=voxel_coor_buffer, updates=output, shape=shape)

    return tf.transpose(output, perm=[0,4,1,2,3]) #[batch_size, channels, Depth, Height, Width]

    

class ConvMiddleLayer(tf.keras.layers.Layer):
  """
    Convolutional Middle Layer class
    Args:
      out_shape : 4-list, int32, dimensions of the output (batch_size, new_chnnles, height, widht)
  """
  def __init__(self, out_shape):
    super(ConvMiddleLayer, self).__init__()
    self.out_shape = out_shape

    self.conv1 = tf.keras.layers.Conv3D(64, (3,3,3), (2,1,1), data_format="channels_first", padding="VALID")
    self.conv2 = tf.keras.layers.Conv3D(64, (3,3,3), (1,1,1), data_format="channels_first", padding="VALID")
    self.conv3 = tf.keras.layers.Conv3D(64, (3,3,3), (2,1,1), data_format="channels_first", padding="VALID")

    self.bn1 = tf.keras.layers.BatchNormalization(trainable=True)
    self.bn2 = tf.keras.layers.BatchNormalization(trainable=True)
    self.bn3 = tf.keras.layers.BatchNormalization(trainable=True)

  def call(self, input):
    """
      Call Method
      Args:
        input : 5D Tensor, float32, shape=[batch_size, channels(128), Depth(10), Height(400), Width(352)]
      returns:
        
    """
    # Refer to the paper, section 3 for details 
    out = tf.pad(input, [(0,0)]*2 + [(1,1)]*3)
    out = tf.nn.relu(self.bn1(self.conv1(out)))

    out = tf.pad(out, [(0,0)]*3 + [(1,1)]*2)
    out = tf.nn.relu(self.bn2(self.conv2(out)))

    out = tf.pad(out, [(0,0)]*2 + [(1,1)]*3)
    out = tf.nn.relu(self.bn3(self.conv3(out)))
    return tf.reshape(out, self.out_shape)



class RPN(tf.keras.layers.Layer):
    
  def __init__(self, num_anchors_per_cell):
    super(RPN, self).__init__()
    self.num_anchors_per_cell = num_anchors_per_cell
    self.num_anchors_per_cell = num_anchors_per_cell 
    BN = tf.keras.layers.BatchNormalization

    # block 1
    self.conv1_block1, self.bn1_block1 = self.conv_layer(128, (3,3),(2,2)), BN(trainable=True)
    self.conv2_block1, self.bn2_block1 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
    self.conv3_block1, self.bn3_block1 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
    self.conv4_block1, self.bn4_block1 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
    
    # block 2
    self.conv1_block2, self.bn1_block2 = self.conv_layer(128, (3,3),(2,2)), BN(trainable=True)
    self.conv2_block2, self.bn2_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
    self.conv3_block2, self.bn3_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
    self.conv4_block2, self.bn4_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
    self.conv5_block2, self.bn5_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
    self.conv6_block2, self.bn6_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)

    # block 3
    self.conv1_block3, self.bn1_block3 = self.conv_layer(256, (3,3),(2,2)), BN(trainable=True)
    self.conv2_block3, self.bn2_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)
    self.conv3_block3, self.bn3_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)
    self.conv4_block3, self.bn4_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)
    self.conv5_block3, self.bn5_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)
    self.conv6_block3, self.bn6_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)
    
    # deconvolutions
    self.deconv_1, self.deconv_bn1 = self.deconv_layer(256, (3,3), (1,1)), BN(trainable=True)
    self.deconv_2, self.deconv_bn2 = self.deconv_layer(256, (2,2), (2,2)), BN(trainable=True)
    self.deconv_3, self.deconv_bn3 = self.deconv_layer(256, (4,4), (4,4)), BN(trainable=True)
    
    # probability and regression maps
    self.prob_map_conv = self.conv_layer(self.num_anchors_per_cell,(1,1),(1,1))
    self.reg_map_conv = self.conv_layer(7*self.num_anchors_per_cell, (1,1),(1,1))
      
  def conv_layer(self, out_channels, kernel_size, stride_size):
    return tf.keras.layers.Conv2D(out_channels, 
                                  kernel_size, 
                                  stride_size, 
                                  padding="SAME", 
                                  data_format="channels_first")

  def deconv_layer(self, out_channels, kernel_size, stride_size):
    return tf.keras.layers.Conv2DTranspose(out_channels, 
                                           kernel_size, 
                                           stride_size, 
                                           padding="SAME", 
                                           data_format="channels_first")

  def block_conv_op(self, block_id, input):
    i = 1
    out = input
    while True:
      try:
        c = getattr(self, "conv{}_block{}".format(i, block_id))
        b = getattr(self, "bn{}_block{}".format(i, block_id))
      except:
        break
      out = tf.nn.relu(b(c(out)))
      i+=1
    return out

  def deconv_op(self, i, input):
    out = input
    c = getattr(self, "deconv_{}".format(i))
    b = getattr(self, "deconv_bn{}".format(i))
    out = tf.nn.relu(b(c(out)))
    return out


  def call(self, input):
    input_shape = input.shape
    assert len(input_shape)==4 and input_shape[-1]%8==0 and input_shape[-2]%8==0, "The input must be of shape [Batch_size, channels, map_height, map_width] with map_height and map_width multiple of 8, got {}".format(input_shape)
    
    output = self.block_conv_op(1, input) 
    deconv1 = self.deconv_op(1, output)

    output = self.block_conv_op(2, output)
    deconv2 = self.deconv_op(2, output)

    output = self.block_conv_op(3, output)
    deconv3 = self.deconv_op(3, output)

    output = tf.concat([deconv3, deconv2, deconv1], axis=1)
    prob_map = self.prob_map_conv((output))
    reg_map = self.reg_map_conv((output))
    prob_map = tf.transpose(prob_map, (0,2,3,1))
    reg_map = tf.transpose(reg_map, (0,2,3,1))

    prob_map = tf.nn.sigmoid(prob_map)

    return prob_map, reg_map



class Model(tf.keras.Model):
  def __init__(self, cfg, params, strategy, *args, **kwargs):
    super(Model, self).__init__()
    self.strategy = strategy
    n_replicas = self.strategy.num_replicas_in_sync
    self.params = params
    self.cfg = cfg
    self.vfe_block = VFE_Block(cfg.VFE_OUT_DIMS, cfg.VFE_FINAl_OUT_DIM, cfg.GRID_SIZE )
    self.convMiddle = ConvMiddleLayer((params["batch_size"]//n_replicas, -1, *cfg.GRID_SIZE[1:]))
    self.rpn = RPN(cfg.NUM_ANCHORS_PER_CELL)

  def add_loss_(self):
    self.loss_object = Loss(self.params)

  def add_optimizer_(self, n_epoch):
    if self.params["mode"]=="train":
      self.optimizer = Optimizer(self.params, n_epoch, optimizer="adam")


  def call(self, training, batch=None, *args, **kwargs):
    if not batch:
      assert "feature_buffer"  in kwargs and "coordinate_buffer" in kwargs, "you must provide a batch object or feature_buffer and coordiante_buffer tensors"
      batch = kwargs
    
    n_replicas = self.strategy.num_replicas_in_sync
    shape = [self.params["batch_size"]//n_replicas]+self.vfe_block.sparse_shape + [self.vfe_block.final_dim]
    output = self.vfe_block(batch["feature_buffer"], batch["coordinate_buffer"], shape, training)
    output = self.convMiddle(output)
    prob_map, reg_map = self.rpn(output)
    return prob_map, reg_map


  def train_step(self, feature_buffer, 
                 coordinate_buffer,
                 targets, 
                 pos_equal_one, 
                 pos_equal_one_reg,
                 pos_equal_one_sum,
                 neg_equal_one, 
                 neg_equal_one_sum):
    with tf.GradientTape() as tape:
      p_map, r_map = self.call(training=True, 
                                feature_buffer=feature_buffer, 
                                coordinate_buffer=coordinate_buffer)
      loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_loss = self.loss_object(r_map, p_map,
                                                                        targets,
                                                                        pos_equal_one, 
                                                                        pos_equal_one_reg,
                                                                        pos_equal_one_sum,
                                                                        neg_equal_one, 
                                                                        neg_equal_one_sum)

    grads = tape.gradient(loss, self.trainable_variables)
    normed_grads, norm = tf.clip_by_global_norm(grads, self.params["max_gradient_norm"])
    self.optimizer.optimizer.apply_gradients(zip(normed_grads, self.trainable_variables))
    return loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_loss

  def dist_train_step(self, feature_buffer,
                             coordinate_buffer,
                             targets,
                             pos_equal_one,
                             pos_equal_one_reg,
                             pos_equal_one_sum,
                             neg_equal_one,
                             neg_equal_one_sum):
    per_replica_losses = self.strategy.experimental_run_v2(self.train_step,
                                                      args=(feature_buffer, 
                                                            coordinate_buffer,
                                                            targets, 
                                                            pos_equal_one, 
                                                            pos_equal_one_reg,
                                                            pos_equal_one_sum,
                                                            neg_equal_one, 
                                                            neg_equal_one_sum))
    return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)

  def validate_step(self, feature_buffer,
                             coordinate_buffer,
                             targets,
                             pos_equal_one,
                             pos_equal_one_reg,
                             pos_equal_one_sum,
                             neg_equal_one,
                             neg_equal_one_sum):
    p_map, r_map = self.call(training=False, 
                              feature_buffer=feature_buffer, 
                              coordinate_buffer=coordinate_buffer)
    loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_los = self.loss_object(r_map, p_map,
                                                      targets,
                                                      pos_equal_one, 
                                                      pos_equal_one_reg,
                                                      pos_equal_one_sum,
                                                      neg_equal_one, 
                                                      neg_equal_one_sum)
    return loss, reg_loss, cls_loss,  cls_pos_loss, cls_neg_los

  def dist_validate_step(self, feature_buffer,
                            coordinate_buffer,
                            targets,
                            pos_equal_one,
                            pos_equal_one_reg,
                            pos_equal_one_sum,
                            neg_equal_one,
                            neg_equal_one_sum):
    per_replica_losses = self.strategy.experimental_run_v2(self.train_step,
                                                    args=(feature_buffer, 
                                                          coordinate_buffer,
                                                          targets, 
                                                          pos_equal_one, 
                                                          pos_equal_one_reg,
                                                          pos_equal_one_sum,
                                                          neg_equal_one, 
                                                          neg_equal_one_sum))
    return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                          axis=None)
      
  def _predict_step(self, feature_buffer, coordinate_buffer):
    p_map, r_map = self.call(training=False, 
                              feature_buffer=feature_buffer, 
                              coordinate_buffer=coordinate_buffer)
    return p_map, r_map
