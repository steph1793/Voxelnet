import tensorflow as tf

class Loss:

  def __init__(self, params):
    self.global_batch_size = params["batch_size"]# self.strategy.num_replicas_in_sync * self.params["batch_size"]
    self.small_addon_for_BCE = params["small_addon_for_BCE"]
    self.alpha_bce = params["alpha_bce"]
    self.beta_bce = params["beta_bce"]
    self.smooth_l1 = tf.losses.Huber(delta=params["huber_delta"], reduction=tf.losses.Reduction.NONE)


  def reg_loss_fn(self, reg_target, 
                  reg_pred, 
                  pos_equal_one_reg, 
                  pos_equal_one_sum):
    
    loss = self.smooth_l1(reg_target*pos_equal_one_reg, reg_pred*pos_equal_one_reg)/pos_equal_one_sum
    return tf.math.reduce_sum(loss) * (1./self.global_batch_size)

  def prob_loss_fn(self,  prob_pred, 
                   pos_equal_one, 
                   pos_equal_one_sum, 
                   neg_equal_one, 
                   neg_equal_one_sum):
    
    cls_pos_loss = (-pos_equal_one * tf.math.log(prob_pred + self.small_addon_for_BCE)) / pos_equal_one_sum
    cls_neg_loss = (-neg_equal_one * tf.math.log(1 - prob_pred + self.small_addon_for_BCE)) / neg_equal_one_sum
    cls_loss = tf.reduce_sum(self.alpha_bce*cls_pos_loss + self.beta_bce*cls_neg_loss)*(1./self.global_batch_size)
    return cls_loss, tf.reduce_sum(cls_pos_loss)*(1./self.global_batch_size), tf.reduce_sum(cls_neg_loss)*(1./self.global_batch_size)

  def __call__(self, reg_pred, prob_pred,
               targets, 
               pos_equal_one, 
               pos_equal_one_reg,
               pos_equal_one_sum,
               neg_equal_one, 
               neg_equal_one_sum):
    
    reg_loss = self.reg_loss_fn(targets, reg_pred, pos_equal_one_reg, pos_equal_one_sum)
    cls_loss, cls_pos_loss, cls_neg_loss = self.prob_loss_fn(prob_pred, 
                                                             pos_equal_one, 
                                                             pos_equal_one_sum, 
                                                             neg_equal_one, 
                                                             neg_equal_one_sum)
    loss = reg_loss + cls_loss
    return loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_loss


class Optimizer:

  def __init__(self, params, epoch_var, optimizer="adam"):
    boundaries = [80, 120]
    self.lr_cst = params["learning_rate"]
    values = [ self.lr_cst, self.lr_cst * 0.1, self.lr_cst * 0.01 ]
    self.lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)(epoch_var)
    self.optimizer = tf.keras.optimizers.Adam(self.lr)