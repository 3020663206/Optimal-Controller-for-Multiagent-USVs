import numpy as np

import model_def

eta_c_1 = 0.1
eta_a_1 = 0.3
eta_c_2 = 0.01
eta_a_2 = 0.4

zeta_1 = 10
zeta_2 = 14

class RBFN(object):

    def __init__(self, hidden_nums, output_nums): #还有一些超参数可能需要初始化
        self.hidden_nums = hidden_nums
        self.output_nums = output_nums
        self.feature_nums = 0
        self.sample_nums = 0
        self.gaussian_kernel_width = 0  # 高斯核宽度
        self.hiddencenters = 0
        self.hiddenoutputs = 0
        self.hiddenoutputs_expand = 0
        self.linearweights = 0
        self.finaloutputs = 0

    def init(self):
        gaussian_kernel_width = np.random.random((self.hiddencenters, 1))               #待修改
        hiddencenters = np.random.random((self.hidden_nums, self.feature_nums))     #待修改
        linearweights = np.random.random((self.hidden_nums + 1, self.output_nums))                 #待修改
        return gaussian_kernel_width, hiddencenters, linearweights

    def forward(self, inputs):
        self.sample_nums, self.feature_nums = inputs.shape
        self.gaussian_kernel_width, self.hiddencenters, self.linearweights = self.init()
        self.hiddenoutputs = self.guass_change(self.gaussian_kernel_width, inputs, self.hiddencenters)
        self.hiddenoutputs_expand = self.add_intercept(self.hiddenoutputs)
        self.finaloutputs = np.dot(self.hiddenoutputs_expand, self.linearweights)

    def guass_function(self, gaussian_kernel_width, inputs, hiddencenters_i):
        return np.exp(-np.linalg.norm((inputs-hiddencenters_i), axis=1)**2/(2*gaussian_kernel_width**2))

    def guass_change(self, gaussian_kernel_width, inputs, hiddencenters):
        hiddenresults = np.zeros((self.sample_nums, len(hiddencenters)))
        for i in range(len(hiddencenters)):
            hiddenresults[:,i] = self.guass_function(gaussian_kernel_width[i], inputs, hiddencenters[i])
        return hiddenresults

    def add_intercept(self, hiddenoutputs):
        return np.hstack((hiddenoutputs, np.ones((self.sample_nums,1))))

class Critic1_NN(RBFN):

    def __init__(self, hidden_nums, output_nums):
        super().__init__(hidden_nums, output_nums)
        self.varpi_super = 0
        self.linearweights_last = 0

    def backward(self, certain_model, actor_1_nn):
        self.varpi_super = -self.hiddenoutputs_expand * [
            certain_model.a * zeta_1 * certain_model.z_1 +
            1 / 2 * certain_model.a * actor_1_nn.finaloutputs + certain_model.lambda_2]
        self.linearweights_last = self.linearweights
        reg_1 = (-2 * zeta_1 * certain_model.z_1.T * certain_model.lambda_2)
        reg_2 = -((certain_model.a * zeta_1 * zeta_1 - 1) * certain_model.z_1.T * certain_model.z_1)
        reg_3 = (1 / 4 * certain_model.a * actor_1_nn.finaloutputs * actor_1_nn.finaloutputs.T)
        reg_4 = (self.varpi_super.T * self.linearweights_last)
        self.linearweights += (-eta_c_1 / (1 + self.varpi_super * self.varpi_super.T) * self.varpi_super) * (reg_1 + reg_2 + reg_3 +reg_4)

class Actor1_NN(RBFN):

    def __init__(self, hidden_nums, output_nums):
        super().__init__(hidden_nums, output_nums)
        self.varpi_super = 0
        self.linearweights_last = 0

    def backward(self, certain_model, critic_1_nn):
        self.varpi_super = -self.hiddenoutputs_expand * [
            certain_model.a * zeta_1 * certain_model.z_1 +
            1 / 2 * certain_model.a * self.finaloutputs + certain_model.lambda_2]
        self.linearweights_last = self.linearweights
        reg_1 = (1 / 2 * self.hiddenoutputs_expand.T * certain_model.z_1)
        reg_2 = ((eta_c_1 / (4 * (1 + self.varpi_super * self.varpi_super.T))) * self.hiddenoutputs_expand * self.hiddenoutputs_expand.T * self.linearweights_last * self.varpi_super.T * critic_1_nn.linearweights_last)
        reg_3 = (-eta_a_1 * self.hiddenoutputs_expand * self.hiddenoutputs_expand.T * self.linearweights_last)
        self.linearweights += (reg_1 + reg_2 + reg_3)

class Critic2_NN(RBFN):

    def __init__(self, hidden_nums, output_nums):
        super().__init__(hidden_nums, output_nums)
        self.varpi_sub = 0
        self.linearweights_last = 0

    def backward(self, certain_model, actor_2_nn, a_hat_dot):
        self.varpi_sub = -self.hiddenoutputs_expand * [certain_model.function_V - zeta_2 * certain_model.z_2
                            - 1 / 2 * actor_2_nn.finaloutputs - a_hat_dot]
        self.linearweights_last = self.linearweights
        reg_1 = 2 * zeta_2 * certain_model.z_2.T * (certain_model.function_V - a_hat_dot)
        reg_2 = - (zeta_2 * zeta_2 - 1) * certain_model.z_2.T * certain_model.z_2
        reg_3 = 1 / 4 * actor_2_nn.finaloutputs * actor_2_nn.finaloutputs.T
        reg_4 = self.varpi_sub.T * self.linearweights_last
        self.linearweights += (-eta_c_2 / (1 + self.varpi_sub * self.varpi_sub.T) * self.varpi_sub) * (reg_1 + reg_2 + reg_3 +reg_4)

class Actor2_NN(RBFN):

    def __init__(self, hidden_nums, output_nums):
        super().__init__(hidden_nums, output_nums)
        self.varpi_sub = 0
        self.linearweights_last = 0

    def backward(self, certain_model, critic_2_nn, a_hat_dot):
        self.varpi_sub = -self.hiddenoutputs_expand * [certain_model.function_V - zeta_2 * certain_model.z_2
                            - 1 / 2 * self.finaloutputs - a_hat_dot]
        self.linearweights_last = self.linearweights
        reg_1 = (1 / 2 * self.hiddenoutputs_expand.T * certain_model.z_2)
        reg_2 = ((eta_c_2 / (4 * (1 + self.varpi_sub * self.varpi_sub.T))) * self.hiddenoutputs_expand * self.hiddenoutputs_expand.T * self.linearweights_last * self.varpi_sub.T * critic_2_nn.linearweights_last)
        reg_3 = (-eta_a_2 * self.hiddenoutputs_expand * self.hiddenoutputs_expand.T * self.linearweights_last)
        self.linearweights += (reg_1 + reg_2 + reg_3)