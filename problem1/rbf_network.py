import torch, random
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)


class RBFN_type1(nn.Module):
    """
    以高斯核作为径向基函数
    """

    def __init__(self, centers, n_out=3):
        """
        :param centers: shape=[center_num,data_dim]
        :param n_out:
        """
        super(RBFN_type1, self).__init__()
        self.n_out = n_out
        self.num_centers = centers.size(0)  # 隐层节点的个数
        self.dim_centure = centers.size(1)  #
        self.centers = nn.Parameter(centers)
        # self.beta = nn.Parameter(torch.ones(1, self.num_centers), requires_grad=True)
        self.beta = torch.ones(1, self.num_centers) * 10
        # 对线性层的输入节点数目进行了修改
        self.linear = nn.Linear(self.num_centers + self.dim_centure, self.n_out, bias=True)
        self.initialize_weights()  # 创建对象时自动执行

    def kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False)))
        return C

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(torch.cat([batches, radial_val], dim=1))
        return class_score

    def initialize_weights(self, ):
        """
        网络权重初始化
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)

class RBFN_type2(nn.Module):
    """
    以高斯核作为径向基函数
    """

    def __init__(self, centers, n_out=3):
        """
        :param centers: shape=[center_num,data_dim]
        :param n_out:
        """
        super(RBFN_type2, self).__init__()
        self.n_out = n_out
        self.num_centers = centers.size(0)  # 隐层节点的个数
        self.dim_centure = centers.size(1)  #
        self.centers = nn.Parameter(centers)
        # self.beta = nn.Parameter(torch.ones(1, self.num_centers), requires_grad=True)
        self.beta = torch.ones(1, self.num_centers) * 10
        # 对线性层的输入节点数目进行了修改
        self.linear = nn.Linear(self.num_centers + self.dim_centure, self.n_out, bias=True)
        self.initialize_weights()  # 创建对象时自动执行

    def kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False)))
        return C
    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(torch.cat([batches, radial_val], dim=1))
        return class_score

    def initialize_weights(self, ):
        """
        网络权重初始化
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)

class RBFN_type3(torch.nn.Module):
    def __init__(self, centers, n_out=3):
        super(RBFN_type3, self).__init__()
        self.n_out = n_out
        self.num_centers = centers.size(0)
        self.centers = torch.nn.Parameter(centers)
        self.beta = torch.nn.Parameter(torch.ones(1, self.num_centers))
        self.linear = torch.nn.Linear(self.num_centers + n_out, self.n_out)
        self.initialize_weights()

    def kernel_fun(self, batches):
        n_input = batches.size(0)
        c = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)  # torch.Size([500, 500, 1])
        x = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)  # torch.Size([500, 500, 1])
        radial_val = torch.exp(-self.beta.mul((c - x).pow(2).sum(2)))
        return radial_val

    def forward(self, x):
        radial_val = self.kernel_fun(x)
        out = self.linear(torch.cat([x, radial_val], dim=1))
        return out, torch.cat([x, radial_val])

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.data.normal_(0, 0.2)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.2)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.2)
                m.bias.data.zero_()


# centers = torch.rand((5,8))
# rbf_net = RBFN(centers)
# rbf_net.print_network()
# rbf_net.initialize_weights()


if __name__ == "__main__":
    data = torch.tensor([[0.25, 0.75], [0.75, 0.75], [0.25, 0.5], [0.5, 0.5], [0.75, 0.5],
                         [0.25, 0.25], [0.75, 0.25], [0.5, 0.125], [0.75, 0.125]], dtype=torch.float32)
    label = torch.tensor([[-1, 1, -1], [1, -1, -1], [-1, -1, 1], [-1, -1, 1], [-1, -1, 1],
                          [1, -1, -1], [-1, 1, -1], [-1, 1, -1], [1, -1, -1]], dtype=torch.float32)
    print(data.size())

    centers = data[0:8, :]
    rbf = RBFN_type1(centers, 3)
    params = rbf.parameters()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9)

    for i in range(10000):
        optimizer.zero_grad()

        y = rbf.forward(data)
        loss = loss_fn(y, label)
        loss.backward()
        optimizer.step()
        print(i, "\t", loss.data)

    # 加载使用
    y = rbf.forward(data)
    print(y.data)
    print(label.data)