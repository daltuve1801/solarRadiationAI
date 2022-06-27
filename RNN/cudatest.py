import torch
import torch.nn as nn
import torchvision.models as models
import time
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self ,NumBins=32):
        self.InNodes=int(NumBins)*2
        self.MediumNode=self.InNodes*2
        super(CNN, self).__init__()
        self.Lin1 = nn.Linear(self.InNodes , self.MediumNode)
        self.Lin2 = nn.Linear(self.MediumNode,  self.MediumNode)
        self.Lin5 = nn.Linear(self.MediumNode, 2)
    def forward(self, input):
        Zoutput = self.Lin1(input)
        Zoutput= F.relu(Zoutput)
        Zoutput = self.Lin2(Zoutput)
        Zoutput= F.relu(Zoutput)
        Zoutput = self.Lin5(Zoutput)
        return Zoutput


x = torch.randn(10000, 64)
model = CNN()

cpu_times = []

for epoch in range(10000):
    t0 = time.perf_counter()
    output = model(x)
    t1 = time.perf_counter()
    cpu_times.append(t1 - t0)

device = 'cuda'
model = model.to(device)
x = x.to(device)
torch.cuda.synchronize()

gpu_times = []
for epoch in range(10000):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    output = model(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    gpu_times.append(t1 - t0)

print('CPU {}, GPU {}'.format(
    torch.tensor(cpu_times).mean(),
    torch.tensor(gpu_times).mean()))
