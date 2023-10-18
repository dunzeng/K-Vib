from fedlab.utils.dataset.functional import lognormal_unbalance_split, homo_partition
from fedlab.utils.dataset.partition import DataPartitioner
from fedlab.contrib.dataset import FedDataset, Subset
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch
import os
import h5py
from torch.utils.data import DataLoader, ConcatDataset
from fedlab.contrib.dataset.basic_dataset import FedDataset, BaseDataset
import numpy as np

class UnbanlancedPartitioner(DataPartitioner):
    def __init__(self, num_clients, num_samples, unbalance_sgm):
        self.num_clients = num_clients
        self.num_samples = num_samples
        self.unbalance_sgm = unbalance_sgm
        self.client_sample_nums = lognormal_unbalance_split(num_clients, num_samples, unbalance_sgm)
        self.client_dict = homo_partition(self.client_sample_nums, num_samples)

    def _perform_partition(self):
        raise NotImplementedError()
        
    def __getitem__(self, index):
        return self.client_dict[index]

    def __len__(self):
        return len(self.client_dict)

class UnbanlancedMNIST(FedDataset):
    def __init__(self, path, partitioner, transform=None, target_transform=None, preprocess=False) -> None:
        trainset = torchvision.datasets.MNIST(root="/data/zengdun/mnist/",
                                                train=True,
                                                download=False)

        self.partitioner = partitioner
        self.num_clients = partitioner.num_clients
        self.path = path

        if preprocess is True:
            if os.path.exists(os.path.join(self.path)) is not True:
                os.makedirs(os.path.join(self.path, "train"))
                os.makedirs(os.path.join(self.path, "var"))
                os.makedirs(os.path.join(self.path, "test"))

            # partition
            subsets = {
                cid: Subset(trainset,
                            partitioner.client_dict[cid],
                            transform=transform,
                            target_transform=target_transform)
                for cid in range(self.num_clients)
            }
            for cid in subsets:
                torch.save(
                    subsets[cid],
                    os.path.join(self.path, "train", "data{}.pkl".format(cid)))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(
            os.path.join(self.path, type, "data{}.pkl".format(id)))
        return dataset

    def get_dataloader(self, id, batch_size, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
    
class UnbalancedFEMNIST(FedDataset):
    def __init__(self, version="v1") -> None:
        super().__init__()
        data_dir = {"v1": "datasets/cookup_train_1.h5", "v2": "datasets/cookup_train_2.h5", "v3":"datasets/cookup_train_3.h5"}
        path = data_dir[version]
        
        femnist = h5py.File("{}".format(path),"r")
        self.num_clients = len(femnist['examples'])
        trans = transforms.ToTensor()

        self.datasets = []
        for key in femnist['examples'].keys():
            x, y = femnist['examples'][key]['pixels'], femnist['examples'][key]['label']
            ty = torch.Tensor(np.array(y)).type(torch.LongTensor)
            tx = [trans(ele).view(1, 28, 28) for ele in x]
            dataset = BaseDataset(tx, ty)
            self.datasets.append(dataset)
        
        testset = h5py.File("datasets/test.h5","r")
        self.test_dataset = []
        for key in testset['examples'].keys():
            x, y = testset['examples'][key]['pixels'], testset['examples'][key]['label']
            ty = torch.Tensor(np.array(y)).type(torch.LongTensor)
            tx = [trans(ele).view(1, 28, 28) for ele in x]
            dataset = BaseDataset(tx, ty)
            self.test_dataset.append(dataset)
        self.test_dataset = ConcatDataset(self.test_dataset)
        self.testloader = DataLoader(self.test_dataset, 2048)
        train_dataset = ConcatDataset(self.datasets)
        self.trainloader = DataLoader(train_dataset, 2048)

    def get_dataset(self, id, type="train"):
        assert id < self.num_clients, "the size of datasize is {} > id {}".format(self.num_clients, id)
        return self.datasets[id]

    def get_dataloader(self, id, batch_size, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader