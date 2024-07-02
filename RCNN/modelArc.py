import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as op

class resblk (nn.Module):
    def __init__(self, ichannel, ochannel, stride = 1, dropout = 0.3, skip = True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.c1 = nn.Sequential(nn.Conv2d(ichannel, ochannel, 3, stride, 1),
                                nn.BatchNorm2d(ochannel), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(ichannel, ochannel, 3, stride, 1),
                                nn.BatchNorm2d(ochannel), nn.ReLU())
        self.skipConnection = None

        if skip:
            if stride != 1 or ichannel != ochannel:
                self.skipConnection = nn.conv2d(ichannel, ochannel, 1, 1)
    def forward(self, x):
        input = x

        output = F.relu(self.c1(x))
        output = self.c2(output)
        
        if self.skipConnection != None:
            output += self.skipConnection(input)

        output = F.relu(output)
        output = self.dropout(output)
        
        return output

class CRNN1(nn.Module):
    def __init__(self, max_chars: int):
        super().__init__()
        self.name = CRNN1
        
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.c0 = nn.conv2d(3, 16, 5)
        self.r1 = resblk(16, 16)
        self.r2 = resblk(16, 16, skip = False)
        self.r3 = resblk(16, 32, 2)
        self.r4 = resblk(32, 32, skip = False)
        self.r5 = resblk(32, 32)
        self.r6 = resblk(32, 64, 2)
        self.r7 = resblk(64, 64)
        self.r8 = resblk(64, 64, skip = False)
        self.r9 = resblk(64, 64, skip = False)

        self.lstm0 = nn.LSTM(64, 128, bidirectional = True, num_layers = 1, batch_first = True)
        #upper/lowercase letters, numbers, special characters(. - * : # % / ( ) = ), max length of output
        self.fc0 = nn.Linear(128*2, max_chars + 1)
    
    def forward(self, x):
        
        x = F.relu(self.c0(x))
        x = self.pool(x)
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        x = self.r5(x)
        x = self.r6(x)
        x = self.r7(x)
        x = self.r8(x)
        x = self.r9(x)
        x = x.reshape(x.size(0), -1, x.size(1))

        x, _ = self.lstm0(x)
        x = self.dropout(x)

        x = self.fc0(x)
        
        x = F.log_softmax(x, 2)
        return x
         