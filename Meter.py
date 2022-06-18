import sklearn.metrics as metrics
import numpy as np
import torch
from torch_geometric.utils import mean_iou, intersection_and_union as i_and_u

seg_classes = {
        'Airplane': [0, 1, 2, 3],
        'Bag': [4, 5],
        'Cap': [6, 7],
        'Car': [8, 9, 10, 11],
        'Chair': [12, 13, 14, 15],
        'Earphone': [16, 17, 18],
        'Guitar': [19, 20, 21],
        'Knife': [22, 23],
        'Lamp': [24, 25, 26, 27],
        'Laptop': [28, 29],
        'Motorbike': [30, 31, 32, 33, 34, 35],
        'Mug': [36, 37],
        'Pistol': [38, 39, 40],
        'Rocket': [41, 42, 43],
        'Skateboard': [44, 45, 46],
        'Table': [47, 48, 49],
    }

class Meters(object):
    def __init__(self, opt):
        super(Meters, self).__init__()
        self.opt = opt
        self.losses = {}
        self.updata_freq = {}
        if self.opt.task=='seg':
            self.m_iou = True
        else:
            self.m_iou = False


    def collect(self, pred, label, loss, cat=None):
        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()

        self.update('preds', pred)
        self.update('label', label)
        self.update('loss', loss)
        self.update('cat', cat)

        # part_iou = []
        # for part in range(label.min(), label.max() + 1):
        #     I = ((pred == part) & (label == part)).sum()
        #     U = ((pred == part) | (label == part)).sum()
        #     if U == 0:
        #         iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
        #     else:
        #         iou = I / float(U)
        #     part_iou.append(iou)

        # print(torch.tensor(part_iou).mean())

    def collect_dict(self, **kwargs):
        for key, value in kwargs.items():
            self.update(key, value)

    def update(self, name, val):
        if not name in self.losses:
            self.losses[name] = [val]
            self.updata_freq[name] = 0
        else:
            self.losses[name].append(val)
            self.updata_freq[name] += 1

    def output(self, train=True):
        # if self.opt.task=='cls':
        train_true = np.concatenate(self.losses['label'])
        train_pred = np.concatenate(self.losses['preds'])
        try:
        #     print(self.losses['cat'])
            train_cats = torch.cat(self.losses['cat'])
        except:
            pass
        # else:
        # train_true = self.losses['label']
        # train_pred = self.losses['preds']
        loss = torch.stack(self.losses['loss']).detach().cpu().numpy()

        result = {}

        if self.m_iou:
            shape_ious = []
            for shape_idx in range(len(train_true)):
                true = torch.from_numpy(train_true[shape_idx])
                pred = torch.from_numpy(train_pred[shape_idx])

                part_iou = []
                part_list = list(seg_classes.values())[train_cats[shape_idx]]
                for part in part_list:
                    I = ((pred == part) & (true == part)).sum()
                    U = ((pred == part) | (true == part)).sum()
                    if U == 0:
                        iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
                    else:
                        iou = I / U.float()
                    part_iou.append(iou)

                part_iou = torch.tensor(part_iou).mean()
                shape_ious.append(part_iou)

            shape_ious = torch.tensor(shape_ious)

            result['mins_iou'] = shape_ious.mean()

            cat_num = train_cats.max() + 1
            cat_iou = torch.zeros(cat_num)
            count = torch.zeros(cat_num)
            for i, cat in enumerate(train_cats):
                cat_iou[cat] += shape_ious[i]
                count[cat] += 1
            for _, name in enumerate(seg_classes.keys()):
                # print("% *s"% (len, A))
                print("%15s"%name, end='')
            print()
            for iou in cat_iou/count:
                print("%15.3f"%iou, end='')
            result['mcat_iou'] = (cat_iou / count).mean()
        else:
            if train:
                result['train_loss'] = loss.sum() / self.updata_freq['loss']
                result['train_acc'] = metrics.accuracy_score(train_true.reshape(-1), train_pred.reshape(-1))
                result['train_avg_acc'] = metrics.balanced_accuracy_score(train_true.reshape(-1),
                                                                          train_pred.reshape(-1))
            else:
                result['test_acc'] = metrics.accuracy_score(train_true.reshape(-1), train_pred.reshape(-1))
                result['test_avg_acc'] = metrics.balanced_accuracy_score(train_true.reshape(-1), train_pred.reshape(-1))

        print(result)

        self.clear()

        return result

    def mean(self):
        for key, value in self.losses.items():
            self.losses[key] = np.asarray(self.losses[key])
            self.losses[key] = self.losses[key].mean()

        return self.losses


    def clear(self):
        self.losses = {}
        self.updata_freq = {}
