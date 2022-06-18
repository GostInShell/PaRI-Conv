## Point Cloud Classification on ModelNet40

### Dependencies
 
Linux (tested on Ubuntu 16.04)

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


### Data

First, please download the ModelNet40 dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)(1.6G), 
and place it at `dataset/modelnet40_normal_resampled`. 
### Usage

To train a model under `SO(3)` or `z` rotations:

    sh scripts/PaRINet_so3.sh 
    sh scripts/PaRINet_rot_z.sh 

Best model will be saved to `log/PaRINet_***/best`. We already put pre-trained model there.
And you can directly evaluate them by running:

    sh scripts/test_PaRINet_rot_z.sh
    sh scripts/test_PaRINet_so3.sh

To visualize the training process, please run:

    tensorboard --logdir log
