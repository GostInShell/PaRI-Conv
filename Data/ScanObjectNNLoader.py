import torch
import torch.utils.data as data
import numpy as np
import os, sys, h5py

from torch_geometric.data import Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class ScanObjectNNCls(data.Dataset):

    def __init__(
            self, transforms=None, train=True, self_supervision=False, split='main'
    ):
        super().__init__()

        self.transforms = transforms

        self.self_supervision = self_supervision

        self.train = train

        split = 'main_split' if split=='main' else f'split_{split}'

        root = f'/data1/crh/Data/ScanObjectNN/h5_files/{split}/'
        if self.self_supervision:
            h5 = h5py.File(root + 'training_objectdataset.h5', 'r')
            points_train = np.array(h5['data']).astype(np.float32)
            h5.close()
            self.points = points_train
            self.labels = None
        elif train:
            h5 = h5py.File(root + 'training_objectdataset.h5', 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            h5 = h5py.File(root + 'test_objectdataset.h5', 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()

        self.class_num = self.labels.max() + 1
        print(self.class_num)

        print('Successfully load ScanObjectNN with', len(self.labels), 'instances')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.train:
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs][:1024].copy()
        current_points[:, 0:3] = pc_normalize(current_points[:, 0:3])
        label = self.labels[idx]

        data = Data(pos=torch.from_numpy(current_points).float(), y=torch.tensor(int(label)).long(),
                    norm=torch.from_numpy(current_points).float())
        
        if self.transforms is not None:
            data = self.transforms(data)

        # if self.self_supervision:
        #     return current_points
        # else:
        # label = self.labels[idx]
        return data

        ###############################################################
        # pc = self.data[index][:self.npoints].numpy()
        # # pc = self.data[index].numpy()
        # cls = np.asarray(self.label[index])
        #
        # pc[:, 0:3] = pc_normalize(pc[:, 0:3])
        #
        # points = self._augment_data(pc)  # only shuffle by default

        # print(points)

        # print((self.cache[index] - points).mean())

        # data = Data(pos=torch.from_numpy(points[:, :3]).float(), y=torch.from_numpy(cls).long(),
        #             norm=torch.from_numpy(points[:, 3:]).float())
        # data.idx = index
        #
        # if self.transform is not None:
        #     data = self.transform(data)

        # return data

    def __len__(self):
        return self.points.shape[0]

