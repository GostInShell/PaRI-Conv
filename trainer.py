import os
import numpy as np
import glob
import torch
from termcolor import colored
import time
from os.path import join, exists
import torch.nn.functional as F
from torch.optim import Adam, SGD

from Data import ModelNetNormal

from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import Networks
import utils

from Meter import Meters
from tensorboardX import SummaryWriter

class LazyScheduler(object):
    def __init__(self):
        pass

    def step(self, epoch=None):
        pass

class Trainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.epoch = 0
        self.writer = SummaryWriter(logdir=opt.dir_name)
        self.meter = Meters(opt)
        self.get_log_paths()
        self.reset_epoch()
        self.result = Meters(opt)
        self.best_acc = 0.

    def build_dataset(self):
        ####################### dataset #########################

        # root = 'dataset/modelnet40_normal_resampled'
        root = '/data1/crh/Data/modelnet40_normal_resampled'
        '''
        Please download ModelNet40 at https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip (1.6G).
        '''

        if self.opt.rand_rot:
            train_transform = T.Compose([utils.RandRotSO3([[360, 0], [360, 1], [360, 2]]),
                                         utils.RandomShiver(),
                                         utils.RandomTranslate(),
                                         utils.LRF(self.opt.o3d_normal,
                                                  self.opt.o3d_normal_orien,
                                                  axes=self.opt.axes),
                                         utils.InitialAttributes(self.opt.init_attr)])
            test_transform = T.Compose([utils.RandRotSO3([[360, 0], [360, 1], [360, 2]]),
                                        utils.LRF(self.opt.o3d_normal,
                                                  self.opt.o3d_normal_orien,
                                                  axes=self.opt.axes),
                                        utils.InitialAttributes(self.opt.init_attr)])
            # pass

        elif self.opt.rot_z:
            train_transform = T.Compose([utils.RandRotSO3([[180, 2]]),
                                         utils.RandomShiver(),
                                         utils.RandomTranslate(),           # comment this when using --glob bary for a
                                                                            # better performance
                                         utils.LRF(self.opt.o3d_normal,
                                                   self.opt.o3d_normal_orien,
                                                   axes=self.opt.axes),
                                         utils.InitialAttributes(self.opt.init_attr)])
            test_transform = T.Compose([
                                        utils.RandRotSO3([[180, 0], [180, 1], [180, 2]]),
                                        utils.LRF(self.opt.o3d_normal,
                                                  self.opt.o3d_normal_orien,
                                                  axes=self.opt.axes),
                                        utils.InitialAttributes(self.opt.init_attr)])


        if self.opt.voting:
            self.rand_rot_so3 = utils.RandRotSO3([[360, 0], [360, 1], [360, 2]])

        if self.opt.dataset=="ModelNetNormal":
            self.train_dataset = ModelNetNormal(root, self.opt.sample_points, split='train',
                                                normal_channel=True,
                                                transform=train_transform,
                                                drop_out=self.opt.density)
            self.test_dataset = ModelNetNormal(root, self.opt.sample_points, split='test',
                                               normal_channel=True,
                                               transform=test_transform)
        elif self.opt.dataset=="ScanObjectNNCls":
            from Data import ScanObjectNNCls
            self.train_dataset = ScanObjectNNCls(transforms=train_transform, train=True)
            self.test_dataset = ScanObjectNNCls(transforms=test_transform, train=False)
        else:
            print("No such dataset.")
            assert 1==0


        self.train_dataset.class_num = self.train_dataset.class_num

        self.train_loader = DataLoader(dataset=self.train_dataset,
                                         batch_size=self.opt.batch_size,
                                         shuffle=True,
                                         num_workers=int(self.opt.workers),
                                         drop_last=False)

        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.opt.batch_size_test,
                                      shuffle=False,
                                      num_workers=int(self.opt.workers),
                                      drop_last=False)
        self.opt.output_channels = int(self.train_dataset.class_num)


    def build_network(self):
        network = Networks.PaRINet(self.opt)

        print(network)
        utils.count_parameters(network)

        network.cuda()
        self.network = network


    def build_optimizer(self):
        self.opt.lrate = self.opt.lrate / 32 * self.opt.batch_size

        if self.opt.use_sgd:


            self.optimizer = SGD(self.network.parameters(), lr=self.opt.lrate * 100, momentum=self.opt.sgd_momentum,
                                 weight_decay=self.opt.weight_decay)
        else:
            self.optimizer = Adam(self.network.parameters(), lr=self.opt.lrate, weight_decay=1e-4)

        self.optimizer.zero_grad()
        self.load_a_model_dict(self.network, self.optimizer, self.opt.model_path)

        if self.opt.use_annl:
            print('CosineAnnealingLR')
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.opt.nepoch, self.opt.lrate,
                                                                        last_epoch=self.epoch - 1)
        elif self.opt.use_step:
            print('step_LR')
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.step_size,
                                                             gamma=self.opt.gamma)
        else:
            print('LazyScheduler')
            self.scheduler = LazyScheduler()


    def build_losses(self):
        def cal_loss(pred, gold, smoothing=self.opt.smooth):
            gold = gold.contiguous().view(-1)

            if smoothing:
                eps = 0.2
                n_class = pred.size(1)

                one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
                one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
                log_prb = F.log_softmax(pred, dim=1)

                loss = -(one_hot * log_prb).sum(dim=1).mean()
            else:
                loss = F.cross_entropy(pred, gold, reduction='mean')

            return loss

        self.criterion = cal_loss


    def train_iteration(self):
        pred = self.network(self.data)
        loss = self.get_loss(pred, self.label)
        loss.backward()
        self.optimizer.step()  # gradient update

        self.print_iteration_stats(loss)

        self.optimizer.zero_grad()
        preds = pred.max(dim=1)[1]
        self.meter.collect(preds, self.label, loss)

    def train_epoch(self):
        self.network.train()
        self.reset_iteration()
        for data in self.train_loader:
            self.data = data.to('cuda')
            self.label = data.y.cuda()
            self.train_iteration()
            self.increment_iteration()

        self.scheduler.step(epoch=self.epoch)
        self.writer.add_scalars('train_results', self.meter.output(), self.epoch)

    def test_iteration(self):

        pred = self.network(self.data)
        loss = self.get_loss(pred, self.label)
        preds = pred.max(dim=1)[1]

        self.meter.collect(preds, self.label, loss)

        print(
            '\r' + colored(
                '[%d: %d/%d]' % (
                    self.epoch, self.iteration, len(self.test_dataset) / (self.opt.batch_size_test)),
                'red') +
            colored('gt_L2:  %f' % loss.item(), 'yellow'),
            end='')

    def test_epoch(self):
        self.network.eval()
        self.reset_iteration()
        for data in self.test_loader:
            self.data = data.to('cuda')
            self.label = data.y.cuda()
            self.test_iteration()
            self.increment_iteration()

        result = self.meter.output()
        if self.opt.test_final:
            self.result.collect_dict(**result)

        if result['train_acc'] > self.best_acc and self.opt.training:
            self.best_acc = result['train_acc']
            print(f'The best acc is {self.best_acc}')
            self.save_best_network()

        self.writer.add_scalars('test_results', result, self.epoch)
        return result

    def get_loss(self, pred, label):
        loss = self.criterion(pred, label)
        return loss

    def print_iteration_stats(self, loss):
        """
        print stats at each iteration
        """
        current_time = time.time()
        self.len_dataset = len(self.train_dataset)
        ellpased_time = current_time - self.start_train_time
        total_time_estimated = self.opt.nepoch * (self.len_dataset / self.opt.batch_size) * ellpased_time / (
                0.00001 + self.iteration + 1.0 * self.epoch * self.len_dataset / self.opt.batch_size)  # regle de 3
        ETL = total_time_estimated - ellpased_time
        print(
            f"\r["
            + colored(f"{self.epoch}", "cyan")
            + f": "
            + colored(f"{self.iteration}", "red")
            + "/"
            + colored(f"{int(self.len_dataset / self.opt.batch_size)}", "red")
            + "] train loss:  "
            + colored(f"{loss.item()} ", "yellow")
            + colored(f"Ellapsed Time: {ellpased_time / 60 / 60}h ", "cyan")
            + colored(f"ETL: {ETL / 60 / 60}h", "red"),
            end="",
        )

    def load_a_model_dict(self, network, optimizer, model_path):
        if model_path != "":
            try:
                print(model_path)
                state = torch.load(model_path)
                if state.get('net', False):
                    network.load_state_dict(state['net'])
                    optimizer.load_state_dict(state['optim'])
                else:
                    network.load_state_dict(state)
                print(" Previous network weights loaded! From ", model_path)
            except:
                try:
                    state = torch.load(model_path)
                    # pre_dict = torch.load(model_path)
                    MP_dict = network.state_dict()
                    if state.get('net', False):
                        pre_dict = state['net']
                        optimizer.load_state_dict(state['optim'])
                    else:
                        pre_dict = state
                    pre_dict = {k: v for k, v in pre_dict.items() if k in MP_dict.keys()}
                    MP_dict.update(pre_dict)
                    network.load_state_dict(MP_dict)
                    print(" Part of previous network weights loaded! From ", model_path)
                except:
                    print("Failed to reload ", model_path)
        else:
            print(f'self.opt.model_path is {model_path}')


    def get_log_paths(self):
        """
        Define paths to save and reload networks from parsed options
        :return:
        """
        net_name = 'network'
        self.opt.model_path = self.get_paths(net_name)

    def get_paths(self, net_name):
        # if not self.opt.not_load_dict:
        model_path = join(self.opt.dir_name, net_name+'_*.pth')

        # # If a network is already created in the directory
        found_m = self.find_logs(model_path)
        if found_m is not None:
            model_path, epoch_m = found_m
            self.opt.start_epoch = epoch_m + 1

            utils.yellow_print(f'Reload {net_name} from epoch {epoch_m}.')
        else:
            model_path = join(self.opt.dir_name, net_name+'.pth')
            utils.yellow_print('Training from start!!!!!')

        return model_path

    def find_logs(self, path):
        dirs = glob.glob(path)
        epochs = []
        if len(dirs) == 0:
            return None
        else:
            for dir in dirs:
                _, file = os.path.split(dir)
                de_ext_file, _ = os.path.splitext(file)
                try:
                    epochs.append(int(de_ext_file.split('_')[-1]))
                except:
                    epochs.append(-1)

            ind = np.asarray(epochs).argmax()
            return dirs[ind], epochs[ind]

    def increment_epoch(self):
        self.epoch += 1

    def increment_iteration(self):
        self.iteration += 1

    def reset_iteration(self):
        self.iteration = 0

    def save_network(self):
        print("saving net...")
        state = {'net': self.network.state_dict(),
                 'optim': self.optimizer.state_dict(),
                 'epoch': self.epoch}
        torch.save(state, f"{self.opt.dir_name}/network_{self.epoch}.pth")
        print("network saved")

    def save_best_network(self):
        print("saving best net...")
        state = {'net': self.network.state_dict(),
                 'optim': self.optimizer.state_dict(),
                 'epoch': self.epoch}
        if not os.path.exists(f"{self.opt.dir_name}/best"):
            os.mkdir(f"{self.opt.dir_name}/best")
        torch.save(state, f"{self.opt.dir_name}/best/network_{0}.pth")
        print("network saved")

    def reset_epoch(self):
        self.epoch = self.opt.start_epoch

