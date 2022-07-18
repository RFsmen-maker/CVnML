# -*- coding: UTF-8 -*-
# D:/Code/CVnML/MLPractice

import numpy as np
import torch
import torch.utils.data as data
import torch.optim as optim
from torch import nn
from torch.nn import init

'''  内部构造
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.liner = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y
'''

if __name__ == "__main__":
    '''  显示所有生成特征数据和结果
    for X, y in data_iter:
        print(X, y, sep='\n')
    '''
    torch.manual_seed(1)
    torch.set_default_tensor_type('torch.FloatTensor')

    # 真正函数
    true_w = [2, -3.4]
    true_b = 4.2

    # 生成训练数据
    num_inputs = 2  # 特征数
    num_examples = 1000
    features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    # 添加噪声
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                           dtype=torch.float)

    dataset = data.TensorDataset(features, labels)
    batch_size = 10

    # 把 dataset 放入 DataLoader
    data_iter = data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        # num_workers=2,              # 多线程来读数据，导致错误
    )

    # net = LinearNet(num_inputs)
    net = nn.Sequential(
        nn.Linear(num_inputs, 1)
        # 可以加入其他层
        )
    layer_num = len(net)
    for i in range(layer_num):
        init.normal_(net[i].weight, mean=0.0, std=0.01)  # 权重按照正态分布
        init.constant_(net[i].bias, val=0.0)  # 偏差默认为 0

    loss = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.03)

    # 训练模型
    num_epochs = 3
    for epoch in range(1, num_epochs+1):
        ls = None
        for X, y in data_iter:
            output = net(X)
            ls = loss(output, y.view(-1, 1))
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        print("epoch %d, loss %f" % (epoch, ls.item()))

    dense = net[0]
    print("weight :", true_w, ':', dense.weight.data,
          '\n',
          "bias :", true_b, ':', dense.bias.data.item(),
          )

    torch.manual_seed(1)
    torch.set_default_tensor_type('torch.FloatTensor')

    # 原函数
    true_w = [2, -3.4]
    true_b = 4.2

    # 生成训练数据、标记数据并添加噪声
    num_inputs = 2  # 特征数
    num_examples = 1000
    features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

    dataset = data.TensorDataset(features, labels)
    batch_size = 10

    # 把 dataset 放入 DataLoader
    data_iter = data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 打乱数据 (打乱比较好)
        # num_workers=2,              # 多线程来读数据，存在运行错误，关闭
    )

    # net = LinearNet(num_inputs)
    net = nn.Sequential(
        nn.Linear(num_inputs, 1)
        # 可以加入其他层
        )
    layer_num = len(net)
    for i in range(layer_num):
        init.normal_(net[i].weight, mean=0.0, std=0.01)  # 权重按照正态分布
        init.constant_(net[i].bias, val=0.0)  # 偏差默认为 0

    loss = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.03)

    # 训练模型
    num_epochs = 3
    for epoch in range(1, num_epochs+1):
        ls = None
        for X, y in data_iter:
            output = net(X)
            ls = loss(output, y.view(-1, 1))
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        print("epoch %d, loss %f" % (epoch, ls.item()))

    dense = net[0]
    print("weight :", true_w, ':', dense.weight.data,
          '\n',
          "bias :", true_b, ':', dense.bias.data.item(),
          )
