import torch
import torch.nn as nn
import numpy as np

class RBFLayer(nn.Module):
    def __init__(self, input_dim, num_centers, sigma=1.0):
        super(RBFLayer, self).__init__()

        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        self.linear = nn.Linear(num_centers, 3)

    def forward(self, x):
        # 计算径向基函数的输出
        x = x.unsqueeze(1).expand(x.size(0), self.num_centers, x.size(-1))
        c = self.centers.unsqueeze(0).expand(x.size(0), self.num_centers, x.size(-1))
        distances = (x - c).pow(2).sum(-1)
        output = (-distances / (2 * self.sigma ** 2)).exp()

        # 使用线性层计算最终输出
        output = self.linear(output)

        return output

# 创建数据集
x_train = np.random.rand(100, 1)
y_train = np.sin(x_train * np.pi) + np.random.normal(0, 0.1, (100, 1))

# 将数据转换为 PyTorch 张量
x_train = torch.tensor(x_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)

# 创建模型
model = RBFLayer(1, 72)

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器
class CustomOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(CustomOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                lr = group['lr']
                if len(state) == 0:
                    state['step'] = 0
                    state['sum'] = torch.zeros_like(p.data)
                state['step'] += 1
                state['sum'] += grad ** 2
                rms = state['sum'] / state['step']
                p.data -= lr * grad / torch.sqrt(rms + 1e-8)

        return loss

optimizer = CustomOptimizer(model.parameters(), lr=0.1)

# 训练模型
for epoch in range(1000):
    # 前向传播
    y_pred = model(x_train)

    # 计算损失函数
    loss = criterion(y_pred, y_train)

    # 反向传播和权重更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 输出每 100 次迭代后的损失值
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        #print(model.linear.weight)

# 测试模型
x_test = torch.linspace(0, 1, 100).unsqueeze(1)
y_test = model(x_test)

# 绘制预测结果和原始数据的比较图
import matplotlib.pyplot as plt
plt.plot(x_train.numpy(), y_train.numpy(), 'o', label='Original data')
plt.plot(x_test.numpy(), y_test.detach().numpy(), label='Fitted line')
plt.legend()
plt.show()


