import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    """神经网络主体"""

    def __init__(self):
        super().__init__()
        # 四个全联接层
        self.fc1 = torch.nn.Linear(28 * 28, 64)  # 第一层 输入为28*28像素尺寸的图像
        self.fc2 = torch.nn.Linear(64, 64)  # 第二层
        self.fc3 = torch.nn.Linear(64, 64)  # 第三层
        self.fc4 = torch.nn.Linear(64, 10)  # 第四层 输出层 输出为10个数字类别

    def forward(self, x):
        """
        前向传播过程
        :x 图像输入
        """
        x = torch.nn.functional.relu(self.fc1(x))  # 先全连接线性计算fc1 再套上激活函数relu
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)  # 输出节点的归一化
        return x


def get_data_loader(is_train):
    """获取数据集"""
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)  # 一个批次batch_size=15张图片


def evaluate(test_data, net):
    """准确率评估"""
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28 * 28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)

    net = Net()  # 实例化模型 还没训练

    print("initial accuracy:", evaluate(test_data, net))  # 没训练的时候

    # 训练模型
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(2):  # 在一个训练集 反复训练 2次 提高训练集利用率
        for (x, y) in train_data:
            net.zero_grad()  # 初始化
            output = net.forward(x.view(-1, 28 * 28))  # 正向传播
            loss = torch.nn.functional.nll_loss(output, y)  # 计算差值
            loss.backward()  # 反向误差传播
            optimizer.step()  # 优化网络参数
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))  # 打印当前网络的正确率

    # 随机抽取3张图片 显示预测结果
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()
