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
    
class CRNNG999(nn.Module):
    def __init__(self, max_chars: int):
        super(CRNNG999, self).__init__()
        self.name = CRNNG999
        
        self.dropout = nn.Dropout(0.2)
        
        self.r1 = resblk(1, 16)
        self.r2 = resblk(16, 16, stride = 2)
        self.r3 = resblk(16, 16, skip = False)
        self.r4 = resblk(16, 32, stride = 2)
        self.r5 = resblk(32, 32, skip = False)
        self.r6 = resblk(32, 64, stride = 2)
        self.r7 = resblk(64, 64)
        self.r8 = resblk(64, 64, skip = False)
        self.r9 = resblk(64, 64, skip = False)
        self.r10 = resblk(64, 64, skip = False)

        self.lstm0 = nn.LSTM(64, 128, bidirectional = True, num_layers = 1, batch_first = True)
        self.fc0 = nn.Linear(128*2, max_chars + 1)
    
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
        x = self.r10(x)

        x = x.reshape(x.size(0), -1, x.size(1))

        x, _ = self.lstm0(x)
        x = self.dropout(x)

        x = self.fc0(x)
        x = F.log_softmax(x, 2)

        return x

class CRNNgreyNEW(nn.Module):
    def __init__(self, max_chars: int):
        super(CRNNgreyNEW, self).__init__()
        self.name = CRNNgreyNEW
        
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1)
        # self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.c0 = nn.Conv2d(1, 8, 7)
        self.c1 = nn.Conv2d(8, 16, 5)
        # resblk(8, 16, skip = False)
        self.r2 = resblk(16, 16, skip = False)
        self.r3 = resblk(16, 32, stride = 2)
        self.r4 = resblk(32, 32, skip = False, dropout = 0.2)
        self.r5 = resblk(32, 64, stride = 2)
        self.r6 = resblk(64, 64)
        self.r7 = resblk(64, 64, skip = False, dropout = 0.25)
        self.r8 = resblk(64, 128, stride = 2)
        self.r9 = resblk(128, 128)
        self.r10 = resblk(128, 128, skip = False, dropout = 0.3)
        self.r11 = resblk(128, 128, skip = False)

        self.lstm0 = nn.LSTM(128, 256, bidirectional = True, num_layers = 1, batch_first = True)
        self.fc0 = nn.Linear(256*2, max_chars + 1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        processed = input/255.0

        if processed.dim() == 3:  # Check if input is missing channel dimension
            processed = processed.unsqueeze(1)

        x = F.relu(self.c0(processed))
        x = F.relu(self.c1(x))
        x = self.pool(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        x = self.r5(x)
        x = self.r6(x)
        x = self.r7(x)
        x = self.r8(x)
        x = self.r9(x)
        x = self.r10(x)
        x = self.r11(x)
        x = x.reshape(x.size(0), -1, x.size(1))

        x, _ = self.lstm0(x)

        x = self.dropout(x)
        x = self.fc0(x)
        x = F.log_softmax(x, 2)

        return x

class CRNNgreyBIG(nn.Module):
    def __init__(self, max_chars: int):
        super(CRNNgreyBIG, self).__init__()
        self.name = CRNNgreyBIG
        
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1)
        # self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.c0 = nn.Conv2d(1, 8, 7)
        self.c1 = nn.Conv2d(8, 16, 5)
        # resblk(8, 16, skip = False)
        self.r2 = resblk(16, 16, skip = False)
        self.r3 = resblk(16, 32, stride = 2)
        self.r4 = resblk(32, 32, skip = False, dropout = 0.2)
        self.r5 = resblk(32, 64, stride = 2)
        self.r6 = resblk(64, 64)
        self.r7 = resblk(64, 64, dropout = 0.25)
        self.r8 = resblk(64, 64, skip = False, dropout = 0.25)
        self.r9 = resblk(64, 128, stride = 2)
        self.r10 = resblk(128, 128)
        self.r11 = resblk(128, 128, skip = False, dropout = 0.3)
        self.r12 = resblk(128, 128, skip = False)

        self.lstm0 = nn.LSTM(128, 256, bidirectional = True, num_layers = 1, batch_first = True)
        self.fc0 = nn.Linear(256*2, max_chars + 1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:  # Check if input is missing channel dimension
            input = input.unsqueeze(1)

        x = F.relu(self.c0(input))
        x = F.relu(self.c1(x))
        x = self.pool(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        x = self.r5(x)
        x = self.r6(x)
        x = self.r7(x)
        x = self.r8(x)
        x = self.r9(x)
        x = self.r10(x)
        x = self.r11(x)
        x = self.r12(x)
        x = x.reshape(x.size(0), -1, x.size(1))

        x, _ = self.lstm0(x)

        x = self.dropout(x)
        x = self.fc0(x)
        x = F.log_softmax(x, 2)

        return x

class CRNN(nn.Module):
    def __init__(self, max_chars: int):
        super(CRNN, self).__init__()
        self.name = CRNN
        
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1)
        
        self.c0 = nn.Conv2d(3, 16, 3)
        self.r1 = resblk(16, 16)
        self.r2 = resblk(16, 16, skip = False)
        self.r3 = resblk(16, 32, stride = 2)
        self.r4 = resblk(32, 32, skip = False)
        self.r5 = resblk(32, 64, stride = 2)
        self.r6 = resblk(64, 64)
        self.r7 = resblk(64, 64)
        self.r8 = resblk(64, 64, skip = False)
        self.r9 = resblk(64, 64, skip = False)

        self.lstm0 = nn.LSTM(64, 128, bidirectional = True, num_layers = 1, batch_first = True)
        self.fc0 = nn.Linear(128*2, max_chars + 1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #input_norm = input/255.0
        input_norm = input.permute(0, 3, 1, 2)

        x = F.relu(self.c0(input_norm))
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


class CRNN2(nn.Module):
    def __init__(self, max_chars: int):
        super(CRNN2, self).__init__()
        self.name = CRNN2
        
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1)
        
        self.c0 = nn.Conv2d(3, 16, 3)
        self.r1 = resblk(16, 16)
        self.r2 = resblk(16, 16, skip = False)
        self.r3 = resblk(16, 32, stride = 2)
        self.r4 = resblk(32, 32, skip = False)
        self.r5 = resblk(32, 64, stride = 2)
        self.r6 = resblk(64, 64)
        self.r7 = resblk(64, 64, stride = 2)
        self.r8 = resblk(64, 64, skip = False)
        self.r9 = resblk(64, 64, skip = False)

        self.lstm0 = nn.LSTM(64, 128, bidirectional = True, num_layers = 1, batch_first = True)
        self.fc0 = nn.Linear(128*2, max_chars + 1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #input_norm = input/255.0
        input_norm = input.permute(0, 3, 1, 2)

        x = F.relu(self.c0(input_norm))
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

class CRNNGREY(nn.Module):
    def __init__(self, max_chars: int):
        super(CRNNGREY, self).__init__()
        self.name = CRNNGREY
        
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1)
        
        self.c0 = nn.Conv2d(1, 8, 3)
        self.r1 = resblk(8, 16, dropout = 0.3)
        self.r2 = resblk(16, 16, skip = False)
        self.r3 = resblk(16, 32, dropout=0.3,stride = 2)
        self.r4 = resblk(32, 32, skip = False)
        self.r5 = resblk(32, 64, dropout = 0.3, stride = 2)
        self.r6 = resblk(64, 64)
        self.r7 = resblk(64, 64)
        self.r8 = resblk(64, 64, skip = False)
        self.r9 = resblk(64, 64, skip = False, dropout = 0.3)

        self.lstm0 = nn.GRU(64, 128, bidirectional = True, num_layers = 1, batch_first = True)
        self.fc0 = nn.Linear(128*2, max_chars + 1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:  # Check if input is missing channel dimension
            input = input.unsqueeze(1)

        x = F.relu(self.c0(input))
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

