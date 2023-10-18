import numpy as np
import json
import os
import argparse

import torch
from torch import nn

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate

from fedlab.models.cnn import CNN_MNIST
from fedlab.contrib.algorithm.basic_server import SyncServerHandler

from fedlab.utils.functional import evaluate, setup_seed
from fedlab.contrib.algorithm.fedavg import FedAvgSerialClientTrainer
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from sampler import UniformSampler, KVibSampler
from dataset import UnbalancedFEMNIST

from model import CNN_MNIST
from scipy.special import softmax
import time

from torch.utils.tensorboard import SummaryWriter

class SamplerServer(SyncServerHandler):
    def setup_optim(self, sampler, weights):  
        self.n = self.num_clients
        self.num_to_sample = int(self.sample_ratio*self.n)
        self.round_clients = int(self.sample_ratio*self.n)
        self.sampler = sampler
        self.weights = weights
    
    @property
    def num_clients_per_round(self):
        return self.round_clients
           
    def sample_clients(self, random=False):
        clients = self.sampler.sample(self.num_to_sample)
        self.round_clients = len(clients)
        assert self.num_clients_per_round == len(clients)
        return clients
        
    def global_update(self, buffer):
        # print("Theta {:.4f}, Ws {}".format(self.theta, self.ws))
        gradient_list = [torch.sub(self.model_parameters, ele[0]) for ele in buffer]
        # gradient_list = [ele[0] for ele in buffer]
        norms = np.array([torch.norm(grad, p=2, dim=0).item() for grad in gradient_list])
        
        if self.sampler.name in ['uniform']:
            indices, _ = self.sampler.last_sampled
            weights = self.weights[indices]
        elif self.sampler.name in ['kvib']:
            indices, probs = self.sampler.last_sampled
            weights = self.weights[indices]
            # print(weights*norms)
            self.sampler.update(weights*norms)
        else:
            assert False
        
        if self.sampler.name in ["uniform"]:
            estimates = Aggregators.fedavg_aggregate(gradient_list, weights)
        elif self.sampler.name in ["kvib"]:
            ws = weights/probs
            estimates = sum([w*grad for grad, w in zip(gradient_list, ws)])
        else:
            assert False

        serialized_parameters = self.model_parameters - estimates
        SerializationTool.deserialize_model(self._model, serialized_parameters)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-num_clients', type=int)
    parser.add_argument('-com_round', type=int)
    parser.add_argument('-k', type=int)
    
    # kvib
    parser.add_argument('-theta', type=float)
    parser.add_argument('-reg', type=float)
  
    # local solver
    parser.add_argument('-batch_size', type=int)
    parser.add_argument('-epochs', type=int)
    parser.add_argument('-lr', type=float)
    
    # setting
    parser.add_argument('-dataset', type=str, default="v1") # v1, v2, v3
    parser.add_argument('-sampler', type=str)
    parser.add_argument('-solver', type=str, default="fedavg")
    parser.add_argument('-freq', type=int, default=5)
    parser.add_argument('-seed', type=int, default=42)
    return parser.parse_args()
args = parse_args()


# basic
model = CNN_MNIST()
criterion = nn.CrossEntropyLoss()
dataset = UnbalancedFEMNIST(args.dataset)
args.num_clients = dataset.num_clients
args.L = args.reg

run_time = time.strftime("%m-%d-%H:%M")
dir = "./{}_logs/seed_{}/{}_NUM{}_BS{}_LR{}_EP{}_K{}_R{}".format("femnist", args.seed, args.dataset, args.num_clients, args.batch_size, args.lr, args.epochs, args.k,
            args.com_round)
log = "{}_{}".format(args.sampler, run_time)

path = os.path.join(dir, log)
writer = SummaryWriter(path)
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))
writer.add_scalar('Test_Accuracy/{}'.format(args.dataset), 0, 0)
setup_seed(args.seed)

if args.solver == "fedavg":    
    trainer = FedAvgSerialClientTrainer(model, args.num_clients, cuda=True)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr, criterion)
    trainer.setup_dataset(dataset)

if args.sampler == "kvib":
    sampler = KVibSampler(args.num_clients, args.k, args.reg, args.com_round)
    args.reg = sampler.reg[0]
    args.theta = sampler.theta

elif args.sampler == "uniform":
    probs = np.ones(args.num_clients)/args.num_clients
    sampler = UniformSampler(args.num_clients, probs)

else:
    assert False

# server-sampler
handler = SamplerServer(model=model,
                        global_round=args.com_round,
                        sample_ratio=float(args.k)/args.num_clients)
    
handler.num_clients = trainer.num_clients
weights = np.array([len(dataset.get_dataset(i)) for i in range(args.num_clients)]) # lambda
weights = weights/weights.sum()
handler.setup_optim(sampler, weights)

t = 0
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))


train_loss, _ = evaluate(handler._model, criterion, dataset.trainloader)
writer.add_scalar('Train_Loss/{}'.format(args.dataset), train_loss, t)
writer.add_scalar('Test_Accuracy/{}'.format(args.dataset), 0, t)
# writer.add_scalar('Test_Loss/{}'.format(args.dataset), 0, t)


history = []
while handler.if_stop is False:
    # server side    
    sampled_clients = handler.sample_clients()
    history.append(len(sampled_clients))
    print("Round {} - Running - Client selection [{}]".format(t, len(sampled_clients)))
    broadcast = handler.downlink_package

    # client side
    trainer.local_process(broadcast, sampled_clients)
    uploads = trainer.uplink_package
    
    for pack in uploads:
        handler.load(pack)
    
    t += 1
    # overall record
    if t % args.freq == 0:
        _, eval_acc = evaluate(handler._model, criterion, dataset.testloader)
        train_loss, _ = evaluate(handler._model, criterion, dataset.trainloader)
        writer.add_scalar('Train_Loss/{}'.format(args.dataset), train_loss, t)
        writer.add_scalar('Test_Accuracy/{}'.format(args.dataset), eval_acc, t)

        print("Round {}, Train Loss {:.4f}, Test Accuracy: {:.4f}".format(
            t, train_loss, eval_acc))


 