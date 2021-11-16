import torch
import torch.nn as nn
import hashlib


class SeededBatchNorm2d(nn.Module):
    
    def __init__(self, num_features, seed=0):
        super(SeededBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.hashed_seed = {}
        m = hashlib.sha256()
        m.update(str(seed).encode('utf-8'))
        self.hash = m.digest()
        self.seed_num = 0
        self.hashed_seed[self.hash] = "bn_" + str(self.seed_num)
        self.seed_num += 1
        if torch.cuda.is_available():
            self.__setattr__(self.hashed_seed[self.hash], nn.BatchNorm2d(num_features).cuda())
        else:
            self.__setattr__(self.hashed_seed[self.hash], nn.BatchNorm2d(num_features))
        
    def forward(self, x):
        self.__getattr__(self.hashed_seed[self.hash]).training = self.training
        return self.__getattr__(self.hashed_seed[self.hash]).forward(x)
        
    def reset_seed(self, seed):
        m = hashlib.sha256()
        m.update(str(seed).encode('utf-8'))
        h = m.digest()
        if not h in self.hashed_seed:
            self.hashed_seed[h] = "bn_" + str(self.seed_num)
            self.seed_num += 1
            if torch.cuda.is_available():
                self.__setattr__(self.hashed_seed[h], nn.BatchNorm2d(self.num_features).cuda())
            else:
                self.__setattr__(self.hashed_seed[h], nn.BatchNorm2d(self.num_features))
        self.hash = h