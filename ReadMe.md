# The Devil is in the Pose: Ambiguity-free 3D Rotation-invariant Learning via Pose-aware Convolution (CVPR, 2022)"
**The Devil is in the Pose: Ambiguity-free 3D Rotation-invariant Learning via Pose-aware Convolution**<br>
by [Ronghan Chen](https://scholar.google.com/citations?user=NH4NtmMAAAAJ&hl=zh-CN&oi=ao), [Yang Cong](https://scholar.google.com/citations?user=iUUu8PkAAAAJ&hl=zh-CN&oi=ao)<br>
in CVPR, 2022. [Arxiv](https://arxiv.org/abs/2205.15210)

**If you have any question about the code or the paper, don't hesitate and open an issuse!😉**

## Dependencies
 
Python=3.7, 
CUDA=10.0,
PyTorch=1.4.0, 
torch_geometric=1.6.0,
torch_cluster=1.5.4,
torch_sparse=0.6.1,
torch_scatter=2.0.4,
tensorboardX, 
scikit-learn, 
numpy,
termcolor

## Data

First, please download the ModelNet40 dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)(1.6G), 
and place it at `dataset/modelnet40_normal_resampled`. 

## Point Cloud Classification on ModelNet40


To train a model under `SO(3)` or `z` rotations:

    sh scripts/PaRINet_so3.sh 
    sh scripts/PaRINet_rot_z.sh 

Best model will be saved to `log/PaRINet_***/best`.
And you can evaluate them by running:

    sh scripts/test_PaRINet_rot_z.sh
    sh scripts/test_PaRINet_so3.sh

To visualize the training process, please run:

    tensorboard --logdir log
    
## Cite this work
If you find this work useful, please citing the paper:

```
@inproceedings{chen2022devil,
  title={The Devil is in the Pose: Ambiguity-free 3D Rotation-invariant Learning via Pose-aware Convolution},
  author={Chen, Ronghan and Cong, Yang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7472--7481},
  year={2022}
}
```
 
## Acknowledgement
- The code framework is borrowed from [3D-CODED](https://github.com/ThibaultGROUEIX/3D-CODED)
- The code for classification architecture is borrowed from [DGCNN](https://github.com/WangYueFt/dgcnn)
- Thanks for 

## TODO
Code on ScanObjectNN and ShapeNetPart.

##

