from easydict import EasyDict as edict

__cfg__ = edict()

# for dataset dir
__cfg__.DATA_DIR = 'DATA_DIR/T_DATA'
__cfg__.KITTY_EVAL_SCRIPT = "kitti_eval/launch_test.sh"
__cfg__.CALIB_DIR = ''

# selected object
__cfg__.DETECT_OBJECT = 'Car'  # Pedestrian/Cyclist
__cfg__.NUM_ANCHORS_PER_CELL = 2

if __cfg__.DETECT_OBJECT == 'Car':
    __cfg__.MAX_POINT_NUMBER = 35
    __cfg__.Z_MIN = -3
    __cfg__.Z_MAX = 1
    __cfg__.Y_MIN = -40
    __cfg__.Y_MAX = 40
    __cfg__.X_MIN = 0
    __cfg__.X_MAX = 70.4
    __cfg__.VOXEL_X_SIZE = 0.2
    __cfg__.VOXEL_Y_SIZE = 0.2
    __cfg__.VOXEL_Z_SIZE = 0.4
    __cfg__.VOXEL_POINT_COUNT = 35
    __cfg__.INPUT_WIDTH = int((__cfg__.X_MAX - __cfg__.X_MIN) / __cfg__.VOXEL_X_SIZE)
    __cfg__.INPUT_HEIGHT = int((__cfg__.Y_MAX - __cfg__.Y_MIN) / __cfg__.VOXEL_Y_SIZE)
    __cfg__.INPUT_DEPTH = int((__cfg__.Z_MAX - __cfg__.Z_MIN) / __cfg__.VOXEL_Z_SIZE)
    __cfg__.LIDAR_COORD = [0, 40, 3]
    __cfg__.FEATURE_RATIO = 2
    __cfg__.FEATURE_WIDTH = int(__cfg__.INPUT_WIDTH / __cfg__.FEATURE_RATIO)
    __cfg__.FEATURE_HEIGHT = int(__cfg__.INPUT_HEIGHT / __cfg__.FEATURE_RATIO)
else:
    __cfg__.MAX_POINT_NUMBER = 45
    __cfg__.Z_MIN = -3
    __cfg__.Z_MAX = 1
    __cfg__.Y_MIN = -20
    __cfg__.Y_MAX = 20
    __cfg__.X_MIN = 0
    __cfg__.X_MAX = 48
    __cfg__.VOXEL_X_SIZE = 0.2
    __cfg__.VOXEL_Y_SIZE = 0.2
    __cfg__.VOXEL_POINT_COUNT = 45
    __cfg__.INPUT_WIDTH = int((__cfg__.X_MAX - __cfg__.X_MIN) / __cfg__.VOXEL_X_SIZE)
    __cfg__.INPUT_HEIGHT = int((__cfg__.Y_MAX - __cfg__.Y_MIN) / __cfg__.VOXEL_Y_SIZE)
    __cfg__.INPUT_DEPTH = int((__cfg__.Z_MAX - __cfg__.Z_MIN) / __cfg__.VOXEL_Z_SIZE)
    __cfg__.FEATURE_RATIO = 2
    __cfg__.LIDAR_COORD = [0, 20, 3]
    __cfg__.FEATURE_WIDTH = int(__cfg__.INPUT_WIDTH / __cfg__.FEATURE_RATIO)
    __cfg__.FEATURE_HEIGHT = int(__cfg__.INPUT_HEIGHT / __cfg__.FEATURE_RATIO)


__cfg__.SCENE_SIZE = [__cfg__.Z_MAX - __cfg__.Z_MIN, __cfg__.Y_MAX- __cfg__.Y_MIN, __cfg__.X_MAX - __cfg__.X_MIN]
__cfg__.VOXEL_SIZE = [__cfg__.VOXEL_Z_SIZE, __cfg__.VOXEL_Y_SIZE, __cfg__.VOXEL_X_SIZE]
__cfg__.GRID_SIZE = [int(A/B) for A,B in zip(__cfg__.SCENE_SIZE, __cfg__.VOXEL_SIZE)]
__cfg__.MAP_SHAPE = [__cfg__.FEATURE_HEIGHT, __cfg__.FEATURE_WIDTH]

__cfg__.IMG_WIDTH = 1242
__cfg__.IMG_HEIGHT = 375
__cfg__.IMG_CHANNEL = 3


# set the log image scale factor
__cfg__.BV_LOG_FACTOR = 4

# For the VFE layer
__cfg__.VFE_OUT_DIMS = [32,128]
__cfg__.VFE_FINAl_OUT_DIM = 128

# cal mean from train set
__cfg__.MATRIX_P2 = ([[719.787081,    0.,            608.463003, 44.9538775],
                  [0.,            719.787081,    174.545111, 0.1066855],
                  [0.,            0.,            1.,         3.0106472e-03],
                  [0.,            0.,            0.,         0]])

# cal mean from train set
__cfg__.MATRIX_T_VELO_2_CAM = ([
    [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
    [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
    [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
    [0, 0, 0, 1]
])
# cal mean from train set
__cfg__.MATRIX_R_RECT_0 = ([
    [0.99992475, 0.00975976, -0.00734152, 0],
    [-0.0097913, 0.99994262, -0.00430371, 0],
    [0.00729911, 0.0043753, 0.99996319, 0],
    [0, 0, 0, 1]
])


# Faster-RCNN/SSD Hyper params
if __cfg__.DETECT_OBJECT == 'Car':
    # car anchor
    __cfg__.ANCHOR_L = 3.9
    __cfg__.ANCHOR_W = 1.6
    __cfg__.ANCHOR_H = 1.56
    __cfg__.ANCHOR_Z = -1.0 - __cfg__.ANCHOR_H/2
    __cfg__.RPN_POS_IOU = 0.6
    __cfg__.RPN_NEG_IOU = 0.45

elif __cfg__.DETECT_OBJECT == 'Pedestrian':
    # pedestrian anchor
    __cfg__.ANCHOR_L = 0.8
    __cfg__.ANCHOR_W = 0.6
    __cfg__.ANCHOR_H = 1.73
    __cfg__.ANCHOR_Z = -0.6 - __cfg__.ANCHOR_H/2
    __cfg__.RPN_POS_IOU = 0.5
    __cfg__.RPN_NEG_IOU = 0.35

if __cfg__.DETECT_OBJECT == 'Cyclist':
    # cyclist anchor
    __cfg__.ANCHOR_L = 1.76
    __cfg__.ANCHOR_W = 0.6
    __cfg__.ANCHOR_H = 1.73
    __cfg__.ANCHOR_Z = -0.6 - __cfg__.ANCHOR_H/2
    __cfg__.RPN_POS_IOU = 0.5
    __cfg__.RPN_NEG_IOU = 0.35

# for rpn nms
__cfg__.RPN_NMS_POST_TOPK = 20
__cfg__.RPN_NMS_THRESH = 0.1
__cfg__.RPN_SCORE_THRESH = 0.96


__cfg__.CORNER2CENTER_AVG = True  # average version or max version


cfg = __cfg__