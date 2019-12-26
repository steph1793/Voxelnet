# Voxelnet (tensorflow 2.0.0)
![Image of Voxelnet Architecture](https://github.com/steph1793/Voxelnet/blob/master/images/pre.png)

Implementation of [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/abs/1711.06396) in tensorflow 2.0.0. <br>
This project is based on the work of [Qiangui Huang](https://github.com/qianguih), ([project](https://github.com/qianguih/voxelnet)) and [Xiaojian Ma](https://github.com/jeasinema). Thanks to them for their tremendous job that allowed me to rebuild this architecture and understand the non explicit parts of the paper.<br><br>

# Dependencies
* Python 3.6
* Tensorflow 2.0.0
* opencv
* numba

# Installation
1. Clone this repository
2. Compile the Cython module
```bash
$ python3 setup build_ext --inplace
```
3. Compile the evaluation code (Optional)
This will compile the Kitti_eval project which I decided not to run during the training (during the dump tests). But at the end, of the training, you may run this evaluation code with the model of your choice. In the training script, it is commented. You may un-comment to run the kitti_eval during the training.
```bash
$ cd kitti_eval
$ g++ -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp
```
4. grant the execution permission to evaluation script
```bash
$ cd kitti_eval
$ chmod +x launch_test.sh
```
