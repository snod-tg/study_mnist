import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=dataset_transforms)
test_set = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=dataset_transforms)

# train_loader = torch.utils.data.DataLoader('./dataset/train', batch_size=64, shuffle=True, num_workers=2)

print(train_set[0])
img, label = train_set[0]
print(img.shape)
print(label)

writer = SummaryWriter('logs_for_data')

for i in range(10):
    img, target = train_set[i]
    writer.add_image('test_set', img, i)

writer.close()
