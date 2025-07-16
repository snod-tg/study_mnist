import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.MNIST(root='./dataset', train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, drop_last=False)

# 测试数据集中的第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter('logs_dataloader')
step = 0
for data in test_loader:
    imgs, labels = data
    print(imgs.shape)
    writer.add_images('test_data_shuffle', imgs, step)
    step += 1

writer.close()

