import numpy as np
import json
import os
import argparse
from copy import deepcopy

import torch
from torch import nn, softmax
from torch.utils.data import DataLoader, ConcatDataset
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.utils.functional import evaluate, setup_seed
from fedlab.contrib.algorithm.fedavg import FedAvgSerialClientTrainer
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from synthetic_dataset import SyntheticDataset

from sampler import KVibSampler, UniformSampler
from model import LinearReg
import time

from torch.utils.tensorboard import SummaryWriter


def solver(weights, k, n):
        norms = np.sqrt(weights)
        idx = np.argsort(norms)
        probs = np.zeros(len(norms))
        l=0
        for l, id in enumerate(idx):
            l = l + 1
            if k+l-n > sum(norms[idx[0:l]])/norms[id]:
                l -= 1
                break
        
        m = sum(norms[idx[0:l]])
        for i in range(len(idx)):
            if i <= l:
                probs[idx[i]] = (k+l-n)*norms[idx[i]]/m
            else:
                probs[idx[i]] = 1
        return np.array(probs)

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
        
        # weights = np.ones(len(buffer))/self.num_clients
        
        if self.sampler.name in ['uniform']:
            indices, _ = self.sampler.last_sampled
            weights = self.weights[indices]
        elif self.sampler.name in ['optimal']:
            indices, probs = self.sampler.last_sampled
            weights = self.weights[indices]
        elif self.sampler.name in ['arbi']:
            indices, probs = self.sampler.last_sampled
            weights = self.weights[indices]
            self.sampler.update(weights*norms)
        else:
            assert False
        
        if self.sampler.name in ["uniform"]:
            # fedavg
            estimates = Aggregators.fedavg_aggregate(gradient_list, weights)
            # estimates = sum([w*grad for grad, w in zip(gradient_list, weights)])
        elif self.sampler.name in ["arbi"]:
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
    parser.add_argument('-sample_ratio', type=float)

    # vrb, arbi
    parser.add_argument('-theta', type=float, default=0.3)
    parser.add_argument('-reg', type=float, default=1)
    
    # local solver
    parser.add_argument('-batch_size', type=int)
    parser.add_argument('-epochs', type=int)
    parser.add_argument('-lr', type=float)

    # data & reproduction
    parser.add_argument('-alpha', type=float, default=0.1)
    parser.add_argument('-preprocess', type=bool, default=True)
    parser.add_argument('-seed', type=int, default=0) # run seed
    
    # setting
    parser.add_argument('-dataset', type=str, default="synthetic")
    parser.add_argument('-sampler', type=str)
    parser.add_argument('-solver', type=str, default="fedavg")
    parser.add_argument('-freq', type=int, default=10)
    parser.add_argument('-dseed', type=int, default=0) # data seed

    parser.add_argument('-a', type=float, default=0.0)
    parser.add_argument('-b', type=float, default=0.0)
    return parser.parse_args()
args = parse_args()

args.k = int(args.num_clients*args.sample_ratio)

# format
dataset = args.dataset
dataset = "synthetic_{}_{}".format(args.a, args.b)

run_time = time.strftime("%m-%d-%H:%M")
base_dir = "online_exps/"
dir = "./{}/{}_seed_{}/Run{}_NUM{}_BS{}_LR{}_EP{}_K{}_R{}".format(base_dir, dataset, args.dseed, args.seed, args.num_clients, args.batch_size, args.lr, args.epochs, args.k,
            args.com_round)
log = "{}_{}".format(args.sampler, run_time)

path = os.path.join(dir, log)
writer = SummaryWriter(path)
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

setup_seed(args.seed)

model = LinearReg(60, 10)
synthetic_path = "./synthetic/data_{}_{}_num{}_seed{}".format(args.a, args.b, args.num_clients, args.dseed)
dataset = SyntheticDataset(synthetic_path, synthetic_path + "/feddata/", False)

test_data = ConcatDataset([dataset.get_dataset(i, "test") for i in range(args.num_clients)])
test_loader = DataLoader(test_data, batch_size=1024)

if args.sampler == "kvib":
    sampler = KVibSampler(args.num_clients, args.k, args.reg, args.com_round)

elif args.sampler == "uniform":
    probs = np.ones(args.num_clients)/args.num_clients
    sampler = UniformSampler(args.num_clients, probs)
else:
    assert False

trainer = FedAvgSerialClientTrainer(model, args.num_clients, cuda=True)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)
trainer.setup_dataset(dataset)

# server-sampler
handler = SamplerServer(model=model,
                        global_round=args.com_round,
                        sample_ratio=args.sample_ratio)
    
handler.num_clients = trainer.num_clients
weights = np.array([len(dataset.get_dataset(i)) for i in range(args.num_clients)]) # lambda
weights = weights/weights.sum()
handler.setup_optim(sampler, weights)

t = 0

loss, acc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)

writer.add_scalar('Test/Loss/{}'.format(args.dataset), loss, t)
writer.add_scalar('Test/Accuracy/{}'.format(args.dataset), acc, t)

# regret
dyrgt = 0
json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

while handler.if_stop is False:
    print("running..")
    # server side
    all_clients = np.arange(args.num_clients)

    broadcast = handler.downlink_package

    # client side
    trainer.local_process(broadcast, all_clients)
    full_info = trainer.uplink_package
    
    grad_list = [torch.sub(handler.model_parameters, ele[0]) for ele in full_info]
    norms = np.array([torch.norm(grad, p=2, dim=0).item() for grad in grad_list])*weights

    if args.sampler in ["optimal"]:
        handler.sampler.update(norms)

    sampled_clients = handler.sample_clients()
    uploads = [full_info[i] for i in sampled_clients]
    
    indices, p = handler.sampler.last_sampled
    full_gradient = Aggregators.fedavg_aggregate(grad_list, weights)

    print("sampled {}".format(str(indices)))

    # sampling variance
    part_grads = [grad_list[i] for i in indices]
    part_weights = weights[indices]/p
    estimates = sum([w*grad for grad, w in zip(part_grads, part_weights)])
    if args.sampler in ["uniform"]:
        estimates = estimates/len(part_weights)

    # variance
    vt = np.abs(torch.norm(estimates, p=2, dim=0) - torch.norm(full_gradient, p=2, dim=0))
    # vt = torch.norm(estimates - full_gradient, p=2, dim=0)

    # regret 
    if args.sampler in ["uniform"]:
        optimal_p = np.sqrt(norms)/np.sqrt(norms).sum()
        optimal = (norms/optimal_p).sum()/len(norms)
        sampler_values = (norms[indices]/p).sum()/len(indices)
    else:
        # optimal k
        optimal_p = solver(norms, args.k, args.num_clients)
        optimal = (norms/optimal_p).sum()

        sampler_values = (norms[indices]/p).sum()
    
    dyrgt += np.abs(sampler_values - optimal)/optimal

    for pack in uploads:
        handler.load(pack)

    t += 1

    writer.add_scalar('Online/Regret/{}'.format(args.dataset), dyrgt, t)
    writer.add_scalar('Online/Variance/{}'.format(args.dataset), vt, t)

    tloss, tacc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)
    writer.add_scalar('Test/Loss/{}'.format(args.dataset), tloss, t)
    writer.add_scalar('Test/Accuracy/{}'.format(args.dataset), tacc, t)

    print("Round {}, Loss {:.4f}, Test Accuracy: {:.4f}, Variance: {:.4f}".format(
        t, tloss, tacc, vt))
    