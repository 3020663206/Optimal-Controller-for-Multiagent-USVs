import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
"""
参考：https://goodgoodstudy.blog.csdn.net/article/details/105756137
"""

def SaveImage(label,pre,path):
    label = label.view(-1).cpu().detach().numpy()
    pre = pre.view(-1).cpu().detach().numpy()
    plt.rcParams['font.sans-serif'] = 'KaiTi'
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)
    ax.plot(label, color='blue', label="实际值")
    ax.plot(pre, color='red', linestyle='--', label='拟合值')
    ax.legend()
    fig.savefig(path, dpi=400)

class RBF(nn.Module):
    def __init__(self,centers,n_out=1):
        super(RBF, self).__init__()
        self.n_out = n_out
        self.num_centers = centers.size(0)

        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(torch.ones(1,self.num_centers))
        self.linear = nn.Linear(self.num_centers+n_out,self.n_out)
        self.initialize_weights()

    def kernel_fun(self,batches):
        n_input = batches.size(0)
        c = self.centers.view(self.num_centers,-1).repeat(n_input,1,1)# torch.Size([500, 500, 1])
        x = batches.view(n_input,-1).unsqueeze(1).repeat(1,self.num_centers,1)# torch.Size([500, 500, 1])
        radial_val = torch.exp(-self.beta.mul((c-x).pow(2).sum(2)))
        return radial_val

    def forward(self,x):
        # 计算径向基距离函数
        radial_val = self.kernel_fun(x)
        out = self.linear(torch.cat([x,radial_val],dim=1))
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.2)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.2)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.2)
                m.bias.data.zero_()


num_centers = 72
n_out = 3
centers = torch.randn(num_centers,n_out)

model = RBF(centers,n_out=3)
optimizer = optim.Adam(model.parameters(),lr=0.1)
loss_fun = nn.MSELoss()

X_ = torch.linspace(-5,5,500).view(500,1)
Y_ = torch.mul(1.1*(1-X_+X_.pow(2).mul(2)),torch.exp(X_.pow(2).mul(-0.5)))

start = time.time()
epochs = 1000
for epoch in range(epochs):
    avg_loss = 0
    Y_pre = model(X_)
    loss = loss_fun(Y_pre,Y_)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("epoch:{}\t  loss:{:>.9}".format(epoch+1,loss.item()))
end = time.time()
print("time:",end-start)

Y_pre = model(X_)
SaveImage(Y_,Y_pre,"RBF.png")

