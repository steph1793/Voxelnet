import tensorflow as tf
import os
from config import cfg
from utils.utils import *
from utils.colorize import colorize

  

def predict_step(model, batch, anchors, cfg, params, summary=False, vis=False):

  @tf.function
  def distributed_predict_step():
    return model.strategy.experimental_run_v2(model._predict_step, args=(batch["feature_buffer"], batch["coordinate_buffer"]))

  tag = batch["tag"].numpy().astype(str)
  if summary or vis:
    batch_gt_boxes3d = label_to_gt_box3d(
    batch["labels"].numpy().astype(str), cls=cfg.DETECT_OBJECT, coordinate='lidar')
  print('predict', tag)

  res = distributed_predict_step()
  if model.strategy.num_replicas_in_sync > 1:
    probs, deltas = tf.concat(res[0].values, axis=0).numpy(), tf.concat(res[1].values, axis=0).numpy()
  else:
    probs, deltas = res[0].numpy(), res[1].numpy()
  batch_boxes3d = delta_to_boxes3d(
        deltas, anchors, coordinate='lidar')
  batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
  batch_probs = probs.reshape((params["batch_size"], -1))

  # NMS
  ret_box3d = []
  ret_score = []
  for batch_id in range(params["batch_size"]):
    # remove box with low score
    ind = np.where(batch_probs[batch_id, :] >= cfg.RPN_SCORE_THRESH)[0]
    tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
    tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
    tmp_scores = batch_probs[batch_id, ind].astype(np.float32)
    # TODO: if possible, use rotate NMS
    boxes2d = corner_to_standup_box2d(
        center_to_corner_box2d(tmp_boxes2d, coordinate='lidar')).astype(np.float32)
    ind = tf.image.non_max_suppression(boxes2d, tmp_scores,max_output_size=cfg.RPN_NMS_POST_TOPK, iou_threshold=cfg.RPN_NMS_THRESH )
    ind = ind.numpy()
    tmp_boxes3d = tmp_boxes3d[ind, ...]
    tmp_scores = tmp_scores[ind]
    ret_box3d.append(tmp_boxes3d)
    ret_score.append(tmp_scores)

  ret_box3d_score = []
  for boxes3d, scores in zip(ret_box3d, ret_score):
    ret_box3d_score.append(np.concatenate([np.tile(cfg.DETECT_OBJECT, len(boxes3d))[:, np.newaxis],
                                                boxes3d, scores[:, np.newaxis]], axis=-1))
  
  img = 255. * batch["img"].numpy() #tensorflow scales the image between 0 and 1 when reading it, we need to rescale it between 0 and 255
  if summary:
    # only summry 1 in a batch
    cur_tag = tag[0]
    P, Tr, R = load_calib( os.path.join( cfg.CALIB_DIR, cur_tag + '.txt' ) )

    front_image = draw_lidar_box3d_on_image(img[0], ret_box3d[0], ret_score[0],
                                                  batch_gt_boxes3d[0], P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
          
    n_points = batch["num_points"][0].numpy()
    lidar = batch["lidar"][0][0:n_points,].numpy()
    bird_view = lidar_to_bird_view_img(lidar, factor=cfg.BV_LOG_FACTOR)
              
    bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[0], ret_score[0],
                                                    batch_gt_boxes3d[0], factor=cfg.BV_LOG_FACTOR, P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)

    heatmap = colorize(probs[0, ...], cfg.BV_LOG_FACTOR)
    return {"tag":tag, "scores":ret_box3d_score, "front_image":tf.expand_dims(front_image, axis=0), 
            "bird_view":tf.expand_dims(bird_view, axis=0), "heatmap":tf.expand_dims(heatmap, axis=0)}

  if vis:
    front_images, bird_views, heatmaps = [], [], []
    for i in range(len(img)):
      cur_tag = tag[i]
      n_points = batch["num_points"][i].numpy()
      lidar = batch["lidar"][i][0:n_points,].numpy()
      P, Tr, R = load_calib( os.path.join( cfg.CALIB_DIR, cur_tag + '.txt' ) )
              
      front_image = draw_lidar_box3d_on_image(img[i], ret_box3d[i], ret_score[i],
                                        batch_gt_boxes3d[i], P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
                                        
      bird_view = lidar_to_bird_view_img(lidar, factor=cfg.BV_LOG_FACTOR)
                                        
      bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[i], ret_score[i],
                                        batch_gt_boxes3d[i], factor=cfg.BV_LOG_FACTOR, P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)

      heatmap = colorize(probs[i, ...], cfg.BV_LOG_FACTOR)

      front_images.append(front_image)
      bird_views.append(bird_view)
      heatmaps.append(heatmap)
          
    return {"tag":tag, "scores":ret_box3d_score, "front_image":front_images, "bird_view":bird_views, "heatmap":heatmaps}

  return { "tag":tag, "scores":ret_box3d_score}



def predict(strategy, model, test_batcher, params, cfg):

  predictions_path = os.path.join(params["model_dir"], params["model_name"], "predictions")

  os.makedirs(predictions_path, exist_ok=True)
  os.makedirs(os.path.join(predictions_path, 'data'), exist_ok=True)
  os.makedirs(os.path.join(predictions_path, 'vis'), exist_ok=True)

  for batch in test_batcher:
    if params["dump_vis"]:
      #res = model.predict_step(batch, test_batcher.anchors, summary=False, vis=True)
      res = predict_step(model, batch,  test_batcher.anchors, cfg, params, summary=False, vis=True)
      tags, results, front_images, bird_views, heatmaps = res["tag"], res["scores"], res["front_image"], res["bird_view"], res["heatmap"]
    else:
      #res = model.predict_step(batch, test_batcher.anchors, summary=False, vis=False)
      res =predict_step(model, batch,  test_batcher.anchors, cfg, params,  summary=False, vis=False)
      tags, results = res["tag"], res["scores"]
    # ret: A, B
    # A: (N) tag
    # B: (N, N') (class, x, y, z, h, w, l, rz, score)
    for tag, result in zip(tags, results):
      of_path = os.path.join(predictions_path, 'data', tag + '.txt')
      with open(of_path, 'w+') as f:
        labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar')[0]
        for line in labels:
          f.write(line)
        print('write out {} objects to {}'.format(len(labels), tag))
      # dump visualizations
      if params["dump_vis"]:
        for tag, front_image, bird_view, heatmap in zip(tags, front_images, bird_views, heatmaps):
          front_img_path = os.path.join( predictions_path, 'vis', tag + '_front.jpg'  )
          bird_view_path = os.path.join( predictions_path, 'vis', tag + '_bv.jpg'  )
          heatmap_path = os.path.join( predictions_path, 'vis', tag + '_heatmap.jpg'  )
        cv2.imwrite( front_img_path, front_image )
        cv2.imwrite( bird_view_path, bird_view )
        cv2.imwrite( heatmap_path, heatmap )