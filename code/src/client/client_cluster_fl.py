import numpy as np
import copy
import datetime
import os
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from src.utils import *
# from src.noise import *

# now = datetime.datetime.now()
# timestamp = now.strftime("%Y%m%d_%H%M%S")
# file_name = 'gradient_L2{}.txt'.format(timestamp)
#
# # 指定文件路径和文件名
# file_path = './results.txt/'
#
# # 如果文件路径不存在，则创建它
# if not os.path.exists(file_path):
#     os.makedirs(file_path)
#
# # 将文件路径和文件名合并起来
# file_path_name = os.path.join(file_path, file_name)

class Client_ClusterFL(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device, dp_clip, dp_epsilon, dp_delta, rounds,
                 train_dl_local=None, test_dl_local=None):
        self.name = name
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr
        self.momentum = momentum
        self.device = device
        self.dp_clip = dp_clip
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.rounds = rounds
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.acc_best = 0
        self.count = 0
        self.save_best = True

    def set_learning_rate(self, second_learning_rate):
        self.lr = second_learning_rate

    def set_dp_clip(self, second_dp_clip):
        self.dp_clip = second_dp_clip

    def train(self,is_print=False):
        self.net.to(self.device)
        self.net.train()

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)
        epoch_loss = []
        noise_scale = self.get_noise()
        #print("noise_scale",noise_scale,"learning_rate", self.lr)
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx,(images,labels) in enumerate(self.ldr_train):
                per_loss = []
                batch_data_parameters_grad_dict = {}
                batch_data_num = len(images)
                cout_sum = 0
                for iid,(X_microbatch, y_microbatch) in enumerate(TensorDataset(images, labels)):# 这里相当于逐样本
                    X_microbatch, y_microbatch = X_microbatch.to(self.device), y_microbatch.to(self.device)
                    per_data_parameters_grad_dict = {}  # 用于存储每个样本计算得到的参数梯度
                    output = self.net(torch.unsqueeze(X_microbatch.to(torch.float32), 0))  # 这要是这里要做升维
                    loss = F.cross_entropy(output, torch.unsqueeze(y_microbatch.to(torch.long), 0))
                    loss.backward()# 梯度求导，这边求出梯度
                    per_loss.append(loss.item())
                    model_parameter_grad_norm = 0.0
                    with torch.no_grad():
                        for name, param in self.net.named_parameters():
                            model_parameter_grad_norm += (torch.norm(param.grad) ** 2).item()
                            per_data_parameters_grad_dict[name] = param.grad.clone().detach()
                        model_parameter_grad_norm = np.sqrt(model_parameter_grad_norm)
                    for name in per_data_parameters_grad_dict:
                        per_data_parameters_grad_dict[name] /= max(1, model_parameter_grad_norm / self.dp_clip)  # 梯度裁剪
                        if name not in batch_data_parameters_grad_dict:
                            batch_data_parameters_grad_dict[name] = per_data_parameters_grad_dict[name]
                        else:
                            batch_data_parameters_grad_dict[name] += per_data_parameters_grad_dict[name]
                    for param in self.net.parameters():
                        param.grad.zero_()
                    cout_sum += 1
                batch_loss.append(sum(per_loss) / len(per_loss))
                for name in batch_data_parameters_grad_dict:  # 为批数据计算得到的参数梯度加噪，并求平均
                    batch_data_parameters_grad_dict[name] += torch.normal(0,noise_scale,size=batch_data_parameters_grad_dict[name].shape, device=self.device)
                    batch_data_parameters_grad_dict[name] /= cout_sum
                for name, param in self.net.named_parameters():
                    param.grad = batch_data_parameters_grad_dict[name]
                optimizer.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return sum(epoch_loss) / len(epoch_loss)


    def get_state_dict(self):
        return self.net.state_dict()

    def get_best_acc(self):
        return self.acc_best

    def get_count(self):
        return self.count

    def get_net(self):
        return self.net

    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target.long(), reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy

    def eval_test_glob(self, glob_dl):
        # pacfl没用的这个函数
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in glob_dl:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(glob_dl.dataset)
        accuracy = 100. * correct / len(glob_dl.dataset)
        return test_loss, accuracy

    def eval_train(self):
        # “eval_train”方法在不更新模型参数的情况下评估神经网络在训练数据上的性能
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                train_loss += F.cross_entropy(output, target.long(), reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy