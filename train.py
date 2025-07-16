import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Two_Network import *

# 准备数据集
train_data = torchvision.datasets.MNIST(root='./dataset', train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='./dataset', train=False, transform=torchvision.transforms.ToTensor())

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为{}".format(train_data_size))
print("测试数据集的长度为{}".format(test_data_size))

# 加载数据
train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 创建网络模型
model = tlw_Net()
model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.01
# momentum = 0.5
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4, betas=(0.9, 0.999))
# optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=5e-4)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epochs = 50

writer = SummaryWriter(log_dir='./logs_Adam')

for epoch in range(epochs):
    print("------第 {} 轮训练开始------".format(epoch + 1))

    # 训练
    train_loss = 0.0
    train_acc = 0
    train_total = 0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 把运行中的loss累加起来
        train_loss += loss.item()
        # 把运行中的准确率acc算出来
        _, predicted = torch.max(outputs.data, 1)
        train_total += images.shape[0]
        train_acc += (predicted == labels).sum().item()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数： {}， Loss： {}, acc: {:.4f} %".format(total_train_step, train_loss / 100,
                                                             100 * train_acc / train_total))

            writer.add_scalar('train_loss', train_loss, total_train_step)
            writer.add_scalar('train_acc', 100 * train_acc / train_total, total_train_step)
            train_loss = 0.0
            train_total = 0
            train_acc = 0

    # 测试
    test_total = 0
    test_acc = 0
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_acc += (predicted == labels).sum().item()

            test_loss = loss_fn(outputs, labels)

            total_test_loss += test_loss.item()

    print("整体测试集上的Loss： {}, acc: {:.4f} %".format(total_test_loss, 100 * test_acc / test_total))
    writer.add_scalar('test_loss_001', total_test_loss, total_test_step)
    writer.add_scalar('test_acc_001', 100 * test_acc / test_total, total_test_step)
    total_test_step += 1

    # torch.save(model, './model_Adam/mnist_{}.pth'.format(epoch))
    torch.save(model, './model_Adam/mnist_{}.pth'.format(epoch))
    # torch.save(model, './model_Adagrad/mnist_{}.pth'.format(epoch))
    print("模型已保存")

writer.close()
