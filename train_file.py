import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import logging

def allocate_gpu_devices(num: int = 1):
    """ Allocate GPU devices
    Args:
        num: the number of devices we want to allocate
    Returns:
        device_list: list of pair (allocated_device_id, tensor)
    """
    device_list = []
    if not torch.cuda.is_available():
        return dev_list
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        if len(device_list) >= num:
            break
        try:
            # If cuda cards are in 'Exclusive_Process' mode, we can
            # try to get one device by putting a tensor on it.
            device = torch.device('cuda', i)
            tensor = torch.zeros(1)
            tensor = tensor.to(device)
            device_list.append((i, tensor))
        except RuntimeError:  # CUDA error: all CUDA-capable devices are busy or unavailable
            logging.info(
                'Failed to select GPU {} out of {} (starting from 0), skip it'
                .format(i, device_count - 1))
            pass
    return device_list

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--world_size", type=int)
parser.add_argument("--file_method", type=str)
args = parser.parse_args()


print("local_rank=%s,world_size=%s,file_method=%s"%(args.local_rank,args.world_size,args.file_method))
class AudioDataset(Dataset):
    def __init__(self):
      None
    def __getitem__(self, index):
      input = torch.randn(10,)
      label = 1
      return input,label
    def __len__(self):
      return 10000


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x,target):
        output = self.net2(self.relu(self.net1(x)))
        loss = self.criterion(output, target)
        return loss

dist.init_process_group(backend='nccl', init_method=args.file_method,world_size=args.world_size,rank=args.local_rank)

trainset = AudioDataset()
# ddp时才需要torch.utils.data.distributed.DistributedSampler

train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, sampler=train_sampler)


model = ToyModel()
print(model)

devices = allocate_gpu_devices(1)
device = torch.device('cuda',devices[0][0])
model = model.to(device)
print("real_gpuid=%s,local_rank=%s,world_size=%s,file_method=%s"%(devices[0][0],args.local_rank,args.world_size,args.file_method))

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[devices[0][0]],output_device=devices[0][0])

optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.9)
for epoch in range(50):
   print("epoch = %d"%epoch)
   for batch_idx, (data, target) in enumerate(trainloader):
      data = data.to(device)
      target = target.to(device)
      loss = model(data,target)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

print("finish %s"%(args.local_rank))
