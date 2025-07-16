import torch
from torch.utils.tensorboard import SummaryWriter
from Two_Network import *

model = tlw_Net()
writer = SummaryWriter(log_dir='./logs_net')

example_input = torch.rand(1, 1, 28, 28)

writer.add_graph(model, example_input)

writer.close()

print("Finished writing")