import torch
import torch.optim as optim
import os
from termcolor import colored
import pandas as pd
import time
from os.path import join, exists
from os import mkdir
import glob
import numpy as np

# from visualize import Visualize, Viz

class AbstractTrainer(object):
    def __init__(self, opt):
        super(AbstractTrainer, self).__init__(opt)
        self.start_time = time.time()
        self.opt = opt
        self.git_repo_path = ""
        self.get_log_paths()
        # self.init_meters()
        self.reset_epoch()
        # self.Viz = Viz()
        self.start_visdom()

        self.save_examples = False
        # my_utils.print_arg(self.opt)

    def start_visdom(self):
        self.visualizer = Visualize(self.opt)

    def commit_experiment(self):
        pass

    def get_current_commit(self):
        """
        This helps reproduce results as all results will be associated with a commit
        :return:
        """
        with open("commit.txt", 'r') as f:
            current_commit = f.read()
            print("git repo path : ", self.git_repo_path)
        return self.git_repo_path + current_commit[:-1]

    def init_save_dict(self, opt):
        self.local_dict_to_save_experiment = opt.__dict__
        self.local_dict_to_save_experiment["commit"] = self.get_current_commit()

    def save_new_experiments_results(self):
        """
        This fonction should be called exactly once per experiment and avoid conflicts with other experiments
        :return:
        """
        if os.path.exists('results.csv'):
            self.results = pd.read_csv('results.csv', header=0)
        else:
            columns = []
            self.results = pd.DataFrame(columns=columns)
        self.update_results()
        self.results.to_csv('results.csv', index=False)  # Index=False avoids the proliferation of indexes

    def update_results(self):
        self.end_time = time.time()
        self.local_dict_to_save_experiment["timing"] = self.end_time - self.start_time
        self.results = self.results.append(self.local_dict_to_save_experiment, ignore_index=True)
        # self.results.drop(self.results.columns[self.results.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

        # Code snippet from Yana Hasson
        # results est un dataframe pandas, dest_folder est un dossier qui contient index.html
        # le replace est au cas où tu as des tags html dans le contenu de tes colonnes (sinon il ne fait rien)
        html_str = self.results.to_html(table_id="example").replace("&lt;", "<").replace("&gt;", ">")
        # html_str contient le html 'brut'
        with open(os.path.join(self.opt.dest_folder, "raw.html"), "w") as fo:

            fo.write(html_str)

    def get_log_paths(self):
        """
        Define paths to save and reload networks from parsed options
        :return:
        """

        if not self.opt.demo:
            if not exists("log"):
                print("Creating log folder")
                mkdir("log")
            if not exists(self.opt.dir_name):
                print("creating folder  ", self.opt.dir_name)
                mkdir(self.opt.dir_name)

        opt_name, net_name = 'optimizer_last', 'network'
        self.opt.optimizer_path, self.opt.model_path = self.get_paths(opt_name, net_name)

        if self.opt.refine:
            opt_name, net_name = 'refinenet_optimizer_last', 'refinenet'
            self.opt.refinenet_optimizer_path, self.opt.refinenet_path = self.get_paths(opt_name, net_name)


    def get_paths(self, opt_name, net_name):
        if not self.opt.not_load_dict:
            self.opt.log_path = join(self.opt.dir_name, "log.txt")
            optimizer_path = join(self.opt.dir_name, opt_name+'_*.pth')
            model_path = join(self.opt.dir_name, net_name+'_*.pth')
            # self.opt.reload_optimizer_path = ""

            # # If a network is already created in the directory
            found_m, found_o = self.find_logs(model_path), self.find_logs(optimizer_path)
            if found_m is not None:
                model_path, epoch_m = found_m
                # optimizer_path, epoch_o = found_o
                if found_o is None or epoch_m != found_o[1]:
                    optimizer_path = ''
                else:
                    optimizer_path = found_o[0]
                self.opt.start_epoch = epoch_m + 1

                my_utils.yellow_print(f'Reload {net_name} from epoch {epoch_m}.')
            # elif found_m is not None:
            #     self.opt.model_path, epoch_m = found_m
            #     self.opt.start_epoch = epoch_m + 1
            #     my_utils.yellow_print(f'Reload from epoch {epoch_m}.')
            #     self.opt.optimizer_path = join(self.opt.dir_name, 'optimizer_last.pth')
            else:
                optimizer_path = join(self.opt.dir_name, opt_name+'.pth')
                model_path = join(self.opt.dir_name, net_name+'.pth')
                my_utils.yellow_print('Training from start!!!!!')
        else:
            # self.opt.optimizer_path = join(self.opt.dir_name, 'optimizer_last.pth')
            # self.opt.model_path = join(self.opt.dir_name, "network.pth")
            optimizer_path, model_path = '', ''
            my_utils.yellow_print('Training from start!!!!!')

        return optimizer_path, model_path


        # if exists(self.opt.model_path):
        #     self.opt.reload_model_path = self.opt.model_path
        #     self.opt.reload_optimizer_path = self.opt.optimizer_path

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

    # def save_network(self):
    #     print("saving net...")
    #     torch.save(self.network.state_dict(), self.opt.model_path)
    #     torch.save(self.optimizer.state_dict(), self.opt.optimizer_path)
    #     print("network saved")

    def init_meters(self):
        self.log = meter.Logs()

    def print_loss_info(self):
        pass

    def build_network(self):
        """
        Create network architecture. Refer to auxiliary.model
        :return:
        """
        pass

    def build_optimizer(self):
        """
        Create optimizer
        """

        pass

    def build_dataset_train(self):
        """
        Create training dataset
        """
        pass

    def build_dataset_test(self):
        """
        Create testing dataset
        """
        pass

    def build_losses(self):
        pass

    def save_network(self):
        print("saving net...")
        if self.opt.refine:
            torch.save(self.refinenet.state_dict(), f"{self.opt.dir_name}/refinenet_{self.epoch}.pth")
            torch.save(self.optimizer.state_dict(), f"{self.opt.dir_name}/refinenet_optimizer_last_{self.epoch}.pth")
        else:
            torch.save(self.network.state_dict(), f"{self.opt.dir_name}/network_{self.epoch}.pth")
            torch.save(self.optimizer.state_dict(), f"{self.opt.dir_name}/optimizer_last_{self.epoch}.pth")
        print("network saved")

    # def dump_stats(self):
    #     """
    #     Save stats at each epoch
    #     """
    #     log_table = {
    #         "epoch": self.epoch + 1,
    #         "lr": self.opt.lrate,
    #         "env": self.opt.env,
    #     }
    #     log_table.update(self.log.current_epoch)
    #     print(log_table)
    #     with open(self.opt.logname, "a") as f:  # open and append
    #         f.write("json_stats: " + json.dumps(log_table) + "\n")
    #     self.local_dict_to_save_experiment.update(self.log.current_epoch)
    #
    #     self.opt.start_epoch = self.epoch
    #     with open(os.path.join(self.opt.dir_name, "options.json"), "w") as f:  # open and append
    #         f.write(json.dumps(self.opt.__dict__))

    # def print_iteration_stats(self, loss):
    #     """
    #     print stats at each iteration
    #     """
    #     current_time = time.time()
    #     ellpased_time = current_time - self.start_train_time
    #     total_time_estimated = self.opt.nepoch * (self.len_dataset) / self.opt.batch_size) * ellpased_time / (
    #             0.00001 + self.iteration + 1.0 * self.epoch * self.len_dataset / self.opt.batch_size)  # regle de 3
    #     ETL = total_time_estimated - ellpased_time
    #     print(
    #         f"\r["
    #         + colored(f"{self.epoch}", "cyan")
    #         + f": "
    #         + colored(f"{self.iteration}", "red")
    #         + "/"
    #         + colored(f"{int(self.len_dataset / self.opt.batch_size)}", "red")
    #         + "] train loss:  "
    #         + colored(f"{loss.item()} ", "yellow")
    #         + colored(f"Ellapsed Time: {ellpased_time / 60 / 60}h ", "cyan")
    #         + colored(f"ETL: {ETL / 60 / 60}h", "red"),
    #         end="",
    #     )

    def print_iteration_stats(self, loss):
        """
        print stats at each iteration
        """
        current_time = time.time()
        self.len_dataset = len(self.dataset.train_dataset)
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

    def train_iteration(self):
        pass

    def learning_rate_scheduler(self):
        """
        Defines the learning rate schedule
        """
        # if self.epoch == self.opt.lr_decay_1:
        #     self.opt.lrate = self.opt.lrate / 10.0
        #     self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        # if self.epoch == self.opt.lr_decay_2:
        #     self.opt.lrate = self.opt.lrate / 10.0
        #     self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        if len(self.opt.lr_schedule) == 0:
            pass
        elif self.opt.lr_schedule[0] == 'exp':
            if self.epoch > self.opt.lr_schedule[1]:
                self.opt.lrate = self.opt.lrate * self.opt.lr_schedule[-1]
                self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        elif self.opt.lr_schedule[0] == 'step':
            if self.epoch == self.opt.lr_schedule[1]:
                self.opt.lrate = self.opt.lrate * self.opt.lr_schedule[-1]
                self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)

    def train_epoch(self):
        pass

    def test_iteration(self):
        pass

    def test_epoch(self):
        pass

    def increment_epoch(self):
        self.epoch = self.epoch + 1
        self.opt.epoch = self.epoch

    def increment_iteration(self):
        self.iteration = self.iteration + 1

    def update_curve(self):
        if self.iteration % 20 == 0:
            self.visualizer.update_curves()

    def reset_iteration(self):
        self.iteration = 0

    def reset_epoch(self):
        self.epoch = self.opt.start_epoch
