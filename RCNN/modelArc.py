import torch.nn as nn
import torch
import torch.nn.functional as F

class resblk (nn.Module):
    def __init__(self, ichannel, ochannel, stride = 1, dropout = 0.2, skip = True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.c1 = nn.Sequential(nn.Conv2d(ichannel, ochannel, 3, stride, 1),
                                nn.BatchNorm2d(ochannel), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(ochannel, ochannel, 3, 1, 1),
                                nn.BatchNorm2d(ochannel), nn.ReLU())
        self.skipConnection = None

        if skip:
            if stride != 1 or ichannel != ochannel:
                self.skipConnection = nn.Conv2d(ichannel, ochannel, 1, stride)

    def forward(self, x):
        input = x

        output = self.c1(x)
        output = self.c2(output)
        
        if self.skipConnection is not None:
            input = self.skipConnection(input)

        output = output + input
        output = self.dropout(output)
        
        return output

class CRNNGBIGGER(nn.Module):
    def __init__(self, max_chars: int):
        super(CRNNGBIGGER, self).__init__()
        self.name = CRNNGBIGGER
        
        self.dropout = nn.Dropout(0.2)
        
        self.r1 = resblk(1, 32)
        self.r2 = resblk(32, 32, stride = 2)
        self.r3 = resblk(32, 32, skip = False)
        self.r4 = resblk(32, 64, stride = 2)
        self.r5 = resblk(64, 64, skip = False)
        self.r6 = resblk(64, 128, stride = 2)
        self.r7 = resblk(128, 128)
        self.r8 = resblk(128, 128, stride = 2)
        self.r9 = resblk(128, 128, skip = False)

        self.lstm0 = nn.LSTM(128, 256, bidirectional = True, batch_first = True)
        self.lstm1 = nn.LSTM(512, 64, bidirectional = True, batch_first = True)
        self.fc0 = nn.Linear(64*2, max_chars + 1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        processed = input / 255.0

        if processed.dim() == 3:  # Check if input is missing channel dimension
            processed = processed.unsqueeze(1)

        x = self.r1(processed)
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
        x, _ = self.lstm1(x)
        x = self.dropout(x)

        x = self.fc0(x)
        x = F.log_softmax(x, 2)

        return x