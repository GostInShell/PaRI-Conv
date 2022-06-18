import torch
# import open3d as o3d
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn import knn
import torch.nn.functional as F
from termcolor import colored

import open3d as o3d

def count_parameters(network):
    num_params = 0
    for param in network.parameters():
        num_params += param.numel()
    print(f'para_num is: {num_params / 1e6} MB')

class RandomShuffle(object):
    def __init__(self, keys=None):
        self.keys = keys

    def __call__(self, data):
        indices = list(range(data.pos.size(0)))
        np.random.shuffle(indices)

        if self.keys is not None:
            for key in self.keys:
                data[key] = data[key][indices]

        else:
            # print(data.__dict__)
            for key in data.__dict__:
                # print(key)
                if data[key] is not None:
                    if data[key].size(0)==data.pos.size(0):
                        data[key] = data[key][indices]
        return data


class RandomTranslate(object):
    def __init__(self):
        pass

    def __call__(self, data):
        # xyz1 = torch.zeros(3).float().uniform_(2. / 3., 3. / 2.)
        xyz2 = torch.zeros(3).float().uniform_(-0.2, 0.2)

        data.pos = data.pos + xyz2.unsqueeze(0)

        return data

class RandomShiver(object):
    def __init__(self):
        pass

    def __call__(self, data):
        xyz1 = torch.zeros(3).float().uniform_(2. / 3., 3. / 2.)

        data.pos = data.pos*xyz1.unsqueeze(0)
        data.norm = data.norm / xyz1.unsqueeze(0)

        return data

class Jitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        if self.sigma>0.001:
            N, C = data.pos.size()
            data.pos += np.clip(self.sigma * torch.randn(N, C), -1 * self.clip, self.clip)
            return data
        else:
            return data


# class RenderModel(object):
#     def __init__(self):
#         pass
#
#     def __call__(self, data):
#         assert data.norm is not None
#         assert data.barycenter is not None
#
#         center = data.pos.mean(keepdim=True, dim=0)
#         ray = data.pos - center
#         dist = torch.norm(ray, dim=-1)
#         ray = F.normalize(ray)
#         data.norm = F.normalize(data.norm)
#
#         # angel between normal and center vector
#         normalized_ray = ray / dist.unsqueeze(-1)
#         cos_norm_ray = torch.einsum("ij,ij->i", data.norm, normalized_ray)
#         sin_norm_ray = torch.norm(data.norm.cross(normalized_ray, dim=1), dim=-1)
#
#         # angel between normal and barycenter
#         data.x = torch.stack((dist, cos_norm_ray, sin_norm_ray), dim=1)
#
#         assert data.x.size(1)==3
#
#         return data


class InitialAttributes(object):
    def __init__(self, init_attr):
        self.init_attr = init_attr
        pass

    def __call__(self, data):
        if self.init_attr=='default':
            center = data.pos.mean(keepdim=True, dim=0)
            ray = data.pos - center
            dist = torch.norm(ray, dim=-1)
            # ray = F.normalize(ray)                         # key here!!!!!!!!!!!!!!!!!!!!!!!

            # angel between axis 0 and center vector
            normalized_ray = ray / dist.unsqueeze(-1)

            cos_l0_ray = torch.einsum("ij,ij->i", data.l0, normalized_ray)
            sin_l0_ray = torch.norm(data.l0.cross(normalized_ray, dim=1), dim=-1)

            # cos_l0_ray = torch.einsum("ij,ij->i", data.l1, normalized_ray)
            # sin_l0_ray = torch.norm(data.l1.cross(normalized_ray, dim=1), dim=-1)

            # angel between normal and barycenter
            data.x = torch.stack((dist, cos_l0_ray, sin_l0_ray), dim=1)
        elif self.init_attr=='dist':
            center = data.pos.mean(keepdim=True, dim=0)
            ray = data.pos - center
            dist = torch.norm(ray, dim=-1)
            data.x = torch.stack((dist, dist, dist), dim=1)
        else:
            data.x = torch.ones_like(data.pos)

        assert data.x.size(1)==3

        return data


# class LocalBarycenter(object):
#     def __init__(self, k):
#         self.k = k
#         pass
#
#     def __call__(self, data):
#         row, col = knn(data.pos, data.pos, self.k)
#         data.barycenter = data.pos[col].view(-1, self.k, 3).mean(dim=1)
#
#         return data


class GetBarycenter(object):
    def __init__(self, k):
        self.k = k
        pass

    def __call__(self, data):
        row, col = knn(data.pos, data.pos, self.k)
        return data.pos[col].view(-1, self.k, 3).mean(dim=1) - data.pos

class GetO3dNormal(object):
    def __init__(self, k, orien=False):
        self.k = k
        self.orien = orien
        print('************************Using Open3D Normal************************')

    def __call__(self, data):
        # row, col = knn(data.pos, data.pos, self.k)
        # return data.pos[col].view(-1, self.k, 3).mean(dim=1) - data.pos

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data.pos.numpy())
        # pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=self.k))
        # pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=0.15))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=10))

        # norm = torch.from_numpy(np.asarray(pcd.normals)).float()
        # mean_norm = []
        #
        # kd = o3d.geometry.KDTreeFlann(pcd)
        # for i in range(len(pcd.points)):
        #     k, idx, _ = kd.search_radius_vector_3d(pcd.points[i], radius=0.1)
        #     mean_norm.append(norm[idx].mean(dim=0))
        #
        # norm = torch.stack(mean_norm, dim=0)
        # assert list(norm.size())==[1024, 3]

        if not self.orien:
            pcd.orient_normals_towards_camera_location()
            pcd.normals = o3d.utility.Vector3dVector(-1. * np.asarray(pcd.normals))
            pcd.orient_normals_consistent_tangent_plane(20)

        norm = torch.from_numpy(np.asarray(pcd.normals)).float()

        # norm = torch.from_numpy(np.asarray(pcd.normals)).float()
        # print(f'norm:{torch.norm(data.norm, dim=1)}')
        norm = norm / torch.norm(norm, keepdim=True, dim=1)

        if self.orien:
            mask = (norm*data.norm).sum(dim=1, keepdim=False) < 0
            norm[mask] = -norm[mask]

        # pcd.normals = o3d.utility.Vector3dVector(norm.numpy())
        # o3d.visualization.draw_geometries([pcd])

        data.norm = norm

        return norm

class LRF(object):
    def __init__(self, o3d_normal=False, orien=False, axes=None):
        if axes==None:
            axes = ["normal", "bary"]
        else:
            print(axes)
            assert isinstance(axes, list) and len(axes)==2

        get_lrf=[]
        for i, name in enumerate(axes):
            if name=="normal":
                if not o3d_normal:
                    get_lrf.append(lambda data: data.norm)
                else:
                    get_lrf.append(GetO3dNormal(48, orien))

            elif name=="bary":
                get_lrf.append(GetBarycenter(48))
            elif name=="glob":
                # get_lrf.append(lambda data: data.pos - torch.mean(data.pos, dim=0))
                get_lrf.append(lambda data: data.pos)
        self.get_lrf = get_lrf


    def __call__(self, data):
        data.l0 = F.normalize(self.get_lrf[0](data))
        data.l1 = F.normalize(self.get_lrf[1](data))

        return data


class RandRotSO3(object):
    def __init__(self, rotates):
        rotates_list = []
        for rot in rotates:
            rotates_list.append(T.RandomRotate(*rot))
        self.rand_rot_so3 = T.Compose(rotates_list)

    def __call__(self, data):
        if hasattr(data, 'norm'):
            key = 'norm'
            data.pos = torch.stack((data.pos, data[key]), dim=0)
        elif hasattr(data, 'normal'):
            key = 'normal'
            data.pos = torch.stack((data.pos, data[key]), dim=0)
        else:
            print('norm key error')

        data.pos = self.rand_rot_so3(data).pos

        # try:
        data[key] = data.pos[1]
        data.pos = data.pos[0]
        # except:
        #     pass

        return data

def grey_print(x):
    print(colored(x, "grey"))


def red_print(x):
    print(colored(x, "red"))


def green_print(x):
    print(colored(x, "green"))


def yellow_print(x):
    print(colored(x, "yellow"))


def blue_print(x):
    print(colored(x, "blue"))


def magenta_print(x):
    print(colored(x, "magenta"))


def cyan_print(x):
    print(colored(x, "cyan"))


def white_print(x):
    print(colored(x, "white"))


def print_arg(opt):
    cyan_print("PARAMETER: ")
    for a in opt.__dict__:
        print(
            "         "
            + colored(a, "yellow")
            + " : "
            + colored(str(opt.__dict__[a]), "cyan")
        )
