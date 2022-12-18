import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
from models import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not os.path.exists('model'):
    os.makedirs('model')

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

EVAL_TEST = True
EVAL_SHOW = True

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=True)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=True)

'''
Training
'''

net = MLP(28*28, 256, 256, 10)                                        # 定义深度神经网络模型
# Move to GPU
device_ids = [0]
model = nn.DataParallel(net, device_ids=device_ids).cuda()
criterion = nn.CrossEntropyLoss()                                     # 定义损失函数
criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)    # 定义优化器，这里采用 torch.optim 库

for epoch in range(NUM_EPOCHS):
    with tqdm(train_loader, unit='batch') as train_epoch:
        train_epoch.set_description(f"Training Epoch {epoch}")
        train_loss = 0
        i = 0
        for images, labels in train_epoch:      # 调取训练数据
            images = images.cuda()
            labels = labels.cuda()
            output = model(images)                # 将数据输入到神经网络
            loss = criterion(output, labels)    # 测量损失
            optimizer.zero_grad()               # 清空所有参数的梯度缓存
            loss.backward()                     # 计算随机梯度进行反向传播
            # 如果调用 loss.backward() ，那么整个图都是可微分的，也就是说包括 loss ，图中的所有张量变量，只要其属性 requires_grad=True
            # 那么其梯度 .grad张量都会随着梯度一直累计。反向传播的实现只需要调用 loss.backward() 即可，当然首先需要清空当前梯度缓存，
            # 即.zero_grad() 方法，否则之前的梯度会累加到当前的梯度，这样会影响权值参数的更新。
            optimizer.step()                    # 应用梯度

            train_loss = train_loss + loss      # 训练误差
            i = i + 1
            train_epoch.set_postfix(loss=(train_loss/i).data.cpu().numpy())
            time.sleep(0.0001)

        torch.save(model.state_dict(), os.path.join('model/MLP_epoch_{}.pth'.format(epoch))) #存储训练得到的参数

    '''
    Evaluation
    '''
    # calculate the accuracy using testing dataset
    if EVAL_TEST:
        model.eval()
        with tqdm(test_loader, unit='batch') as test_epoch:
            test_epoch.set_description(f"Testing Epoch {epoch}")
            correct = 0
            num = 0
            for images, labels in test_epoch:
                output = model(images)
                pred_y = torch.max(output, 1)[1].data.cpu().numpy()
                correct = correct + float((pred_y == labels.data.cpu().numpy()).astype(int).sum())
                num = num + float(labels.size(0))

            accuracy = correct / num
            print('Epoch: ', epoch, '| test accuracy: %.4f' % accuracy)

            if EVAL_SHOW:
                print(pred_y)
                img = torchvision.utils.make_grid(images)
                img = img.numpy().transpose(1, 2, 0)
                plt.imshow(img)
                plt.show()

