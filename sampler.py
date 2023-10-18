import numpy as np
import math 
from abc import ABCMeta, abstractmethod

class SamplerBase:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, n, random_state=0):
        if random_state is None or isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
            pass
        self.random_state = random_state
        self.n = n
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass

    @abstractmethod
    def update(self, loss):
        pass

    @abstractmethod
    def reset(self):
        pass

class UniformSampler(SamplerBase):
    def __init__(self, n, probs):
        self.name = "uniform"
        self.n = n
        self.p = probs

    def sample(self, k):
        sampled = np.random.choice(self.n, k, p=self.p, replace=False)
        self.last_sampled = sampled, self.p[sampled]
        return np.sort(sampled)
        
    def update(self, weights):
        pass

class KVibSampler():
    def __init__(self, n, k, reg, T):
        self.name = "kvib"
        self.n, self.k  = n, k
        self.theta = math.pow(n/(T*k), 1/3)
        self.reg = (reg*n)/(self.theta*k)*np.ones(n)

        self.w = np.zeros(n)
        self.p = np.ones(n)*k/n
        
        print("theta {} reg {}".format(self.theta, self.reg))
        self.last_sampled = None
        
    def solver(self, norms):
        # norms = np.sqrt(weights)
        idx = np.argsort(norms)
        probs = np.zeros(len(norms))
        l=0
        for l, id in enumerate(idx):
            l = l + 1
            if self.k+l-self.n > sum(norms[idx[0:l]])/norms[id]:
                l -= 1
                break
        
        m = sum(norms[idx[0:l]])
        for i in range(len(idx)):
            if i <= l:
                probs[idx[i]] = (self.k+l-self.n)*norms[idx[i]]/m
            else:
                probs[idx[i]] = 1
        return np.array(probs)
    
    def sample(self, batch_size=None):
        probs = self.solver(self.w+self.reg)
        assert np.abs(probs.sum() - self.k) <= 1

        mixed_probs = (1-self.theta)*probs + self.theta*self.k/self.n
        sampled = np.arange((self.n))[np.random.random_sample(self.n) <= mixed_probs]
        self.last_sampled = (sampled, mixed_probs[sampled])
        
        # print("Original Probs {}".format(str(probs[sampled])))
        #print("Sampler. \n Theta {} Probs {} \n Sampled {} - [{}]".format(self.theta, mixed_probs, sampled, len(sampled)))

        return sampled
        
    def update(self, weights):
        indices, probs = self.last_sampled
        assert len(weights) == len(indices)
        self.w[indices] += weights**2/probs
        # print("Feedback. \n Weights {}".format(self.w))
    
    def full_update(self, weights):
        assert len(weights) == self.n
        self.w += weights**2/self.p

class OptimalSampler(SamplerBase):
    def __init__(self, n, k):
        super().__init__(n)
        self.name = "optimal"
        self.k = k
        self.p = None
    
    def sample(self, batch_size=None):
        indices = np.arange((self.n))[np.random.random_sample(self.n) <= self.p]
        self.last_sampled = indices, self.p[indices]
        return indices
        
    def update(self, loss):
        self.p = self.optim_solver(loss)

    def estimate(self):
        indices = np.arange((self.n))[np.random.random_sample(self.n) <= self.p]
        return indices, self.p[indices]

    def optim_solver(self, norms):
        norms = np.array(norms)
        idx = np.argsort(norms)
        probs = np.zeros(len(norms))
        l=0
        for l, id in enumerate(idx):
            l = l + 1
            if self.k+l-self.n > sum(norms[idx[0:l]])/norms[id]:
                l -= 1
                break
        
        m = sum(norms[idx[0:l]])
        for i in range(len(idx)):
            if i <= l:
                probs[idx[i]] = (self.k+l-self.n)*norms[idx[i]]/m
            else:
                probs[idx[i]] = 1
                
        return np.array(probs)
    