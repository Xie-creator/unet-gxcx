from model import Unet
import torch
from dataset import train_loader
from dataset import test_loader
from dataset import trans
from matplotlib import pyplot as plt
torch.backends.cudnn.enabled = False
from torchvision import transforms
import numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet()
model.to(torch.device(device))

EPOCH = 200

# Construct loss and optimizer ------------------------------------------------------------------------------
criterion = torch.nn.MSELoss()
criterion = torch.nn.MSELoss().to(torch.device(device))
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.5)  # lr学习率，momentum冲量

trans1 = transforms.Compose([transforms.RandomCrop(256)])


# Train and Test CLASS --------------------------------------------------------------------------------------
# 把单独的一轮一环封装在函数类里
def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data

        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        inputs = trans1(inputs)
        torch.random.manual_seed(seed)
        target = trans1(target)

        inputs = inputs.to(device)#to GPU
        target = target.to(device)
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        # 把运行中的loss累加起来，为了下面300次一除
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 100 == 99:  # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0  # 这小批300的loss清零
            running_total = 0
            running_correct = 0  # 这小批300的acc清零

def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data

            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            images = trans1(images)
            torch.random.manual_seed(seed)
            labels = trans1(labels)

            images=images.to(device)# to GPU
            labels=labels.to(device)
            outputs = model(images)
            diff = labels-outputs
            diff = torch.square(diff)
            acc = torch.sum(diff).cpu().numpy()

    print('[%d / %d]: MSE on test set: %.1f  ' % (epoch+1, EPOCH, acc))  # 求测试的准确率，正确数/总数
    return acc

if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        # if epoch % 10 == 9:  #每训练10轮 测试1次
        acc_test = test()
        acc_list_test.append(acc_test)

    torch.save(model, r"unet1.path")
    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()
