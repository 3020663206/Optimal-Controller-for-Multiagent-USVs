#  使用深度强化学习实现无人艇集群合围算法

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from math import pi, atan2, sqrt, sin, cos, atan
import torch

M_11 = 1.956
M_22 = 2.405
M_33 = 0.043
D_11 = 2.436
D_22 = 12.992
D_33 = 0.0564
rho_0 = 0.1
R = np.eye(3)
zeta_1 = 10
zeta_2 = 14
pian_dot_v1_0 = 1
gamma_c_1 = 0.1
gamma_c_2 = 0.01
gamma_a_1 = 0.3
gamma_a_2 = 0.4
num_centers = 72
n_out_c_1 = 1
n_out_c_2 = 3
n_out_a_1 = 3
n_out_a_2 = 2
centers_c_1 = torch.randn(num_centers, n_out_c_1)
centers_c_2 = torch.randn(num_centers, n_out_c_2)
centers_a_1 = torch.randn(num_centers, n_out_a_1)
centers_a_2 = torch.randn(num_centers, n_out_a_2)

T = 0.1

NumberofHunters = 3

class model():

    def __init__(self, x, y, ang, u, v, r):
        self.x = x
        self.y = y
        self.ang = ang
        self.u = u
        self.v = v
        self.r = r
        self.V = np.array([u, v, r]).T

    def get_rho_and_ang(self, x_0, y_0):
        self.rho = np.sqrt((x_0 - self.x) * (x_0 - self.x) + (y_0 - self.y) * (y_0 - self.y))
        self.angle = atan2(y_0 - self.y, x_0 - self.x)
        self.q = np.array([cos(self.angle), -sin(self.angle)], [sin(self.angle), cos(self.angle)])

    def change_position(self, tau_1, tau_2):
        u_next = (tau_1 - D_11 * self.u + M_22 * self.v * self.r) / M_11 * T
        v_next = (-D_22 * self.v - M_11 * self.u * self.r) / M_22 * T
        r_next = (tau_2 - D_33 * self.r - (M_22 - M_11) * self.u * self.v) / M_33 * T
        # 运动学模型 （1）
        self.u += u_next
        self.v += v_next
        self.r += r_next
        self.x += (self.u * cos(self.ang) - self.v * sin(self.ang)) * T
        self.y += (self.u * sin(self.ang) + self.v * cos(self.ang)) * T
        self.ang += self.r * T
        self.f_v = np.array([(-D_11 * self.u + M_22 * self.v * self.r) / M_11] , [(-D_22 * self.v - M_11 * self.u * self.r) / M_22] , [(-D_33 * self.r - (M_22 - M_11) * self.u * self.v) / M_33])

    def z_1_and_z_1_dot(self, angle_up, angle_down, rho_up, rho_down, u_up, u_down, v_up, v_down, v_x0, v_y0):
        self.z_1 = (self.rho - rho_0) ** 2 + (2 * self.angle - angle_up - angle_down) ** 2 + (
                self.ang + pi / 2 - self.angle)
        self.alpha_1 = 2 * (self.rho - rho_0) * (np.array([[cos(self.angle), sin(self.angle)]]))
        self.beta_1 = (4 * (2 * self.angle - angle_up - angle_down) * (
            np.array([[-sin(self.angle), cos(self.angle)]]))) / self.rho
        self.beta_2 = (2 * (2 * self.angle - angle_up - angle_down) * (
            np.array([[-sin(angle_up), cos(angle_up)]]))) / rho_up
        self.beta_3 = (2 * (2 * self.angle - angle_up - angle_down) * (
            np.array([[-sin(angle_down), cos(angle_down)]]))) / rho_down
        self.gamma_1 = 2 * (self.ang + pi / 2 - self.angle)
        self.gamma_2 = (2 * (self.ang + pi / 2 - self.angle) * (
            np.array([[-sin(self.angle), cos(self.angle)]]))) / self.rho
        q_up = np.array([[cos(angle_up), -sin(angle_up)], [sin(angle_up), cos(angle_up)]])
        q_down = np.array([[cos(angle_down), -sin(angle_down)], [sin(angle_down), cos(angle_down)]])
        self.delta = self.beta_2 * q_up * np.array([u_up, v_up]) + self.beta_3 * q_down * np.array([u_down, v_down])
        self.lambda_1 = np.array([(self.alpha_1 + self.beta_1 - self.gamma_1) * self.q, self.gamma_1])
        self.lambda_2 = (self.alpha_1 + self.beta_1 - self.beta_2 - self.beta_3) * np.array([v_x0, v_y0])
        self.dot_z_1 = self.lambda_1 * np.array([self.u, self.v, self.r]) * self.lambda_2
        self.a = self.lambda_1 * np.linalg.inv(R) * self.lambda_1.T
        return self.z_1

    def get_optimal_a(self):
        r1 = self.z_1 * self.z_1 + self.V.T * R * self.V
        a_star = np.linalg.inv(R) * self.lambda_1.T * (-zeta_1 * self.z_1 - 1 / 2 * pian_dot_v1_0)

    def z_2_and_z_2_dot(self, a_hat):
        self.z_2 = self.V - a_hat
        return self.z_2


class optimal_Model(torch.nn.Module):
    def __init__(self):
        super(optimal_Model, self).__init__()
        self.linear = torch.nn.Linear(72, 1)

    pass


class target():
    x, y, ang = 0, 0, 0
    v_x, v_y = 0, 0
    u, v = 0, 0

    def __init__(self, x, y, ang):
        self.x = x
        self.y = y
        self.ang = ang

    def change_v(self, v_x, v_y):
        self.v_x = v_x
        self.v_y = v_y
        self.ang = atan2(v_y, v_x)
        self.u = v_x * cos(self.ang) + v_y * sin(self.ang)
        self.v = -v_x * sin(self.ang) + v_y * cos(self.ang)

    def change_position(self):
        self.x += self.v_x * T
        self.y += self.v_y * T


class RBF(torch.nn.Module):
    def __init__(self, centers, n_out=3):
        super(RBF, self).__init__()
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


#model_1 = model(0, 0, 0, 0, 0, 0)
#model_2 = model(0, 0, 0, 0, 0, 0)
#model_3 = model(0, 0, 0, 0, 0, 0)

allhunters = [model(0, 0, 0, 0, 0, 0) for i in range(NumberofHunters)]

model_c_1 = RBF(centers_c_1, n_out_c_1)
model_a_1 = RBF(centers_a_1, n_out_a_1)
model_c_2 = RBF(centers_c_2, n_out_c_2)
model_a_2 = RBF(centers_a_2, n_out_a_2)

epochs = 500
a_hat_dot = 0

for epoch in range(epochs):

    for i in range(NumberofHunters):
        # critic 网络更新
        out_put_c_1 = model_c_1(allhunters[i].z_1)
        dot_v_1 = 2 * zeta_1 * allhunters[i].z_1 + out_put_c_1
        out_put_a_1 = model_a_1(allhunters[i].z_1)
        a_hat = np.linalg.inv(R) * allhunters[i].lambda_1 * (-zeta_1 * allhunters[i].z_1 - 1 / 2 * out_put_a_1)
        omaga_c_1 = -model_c_1.forward(allhunters[i].z_1)[1] * [
            allhunters[i].a * zeta_1 * allhunters[i].z_1 + 1 / 2 * allhunters[i].a * out_put_a_1 + allhunters[
                i].lambda_2]
        model_c_1.linear.weight += (-gamma_c_1 / (1 + omaga_c_1 * omaga_c_1.T) * omaga_c_1 * [
            -2 * zeta_1 * allhunters[i].z_1.T * allhunters[i].lambda_2
            - (allhunters[i].a * zeta_1 * zeta_1 - 1) * allhunters[i].z_1.T * allhunters[i].z_1 + 1 / 4 * allhunters[
                i].a * out_put_a_1 * out_put_a_1.T + omaga_c_1.T * model_c_1.linear.weight])
        model_a_1.linear.weight += (
                    1 / 2 * model_c_1.forward(allhunters[i].z_1)[1].T * allhunters[i].z_1 + gamma_c_1 / (
                    4 * (1 + omaga_c_1 * omaga_c_1.T))
                    * model_c_1.forward(allhunters[i].z_1)[1].T * model_c_1.forward(allhunters[i].z_1)[
                        1] * model_a_1.linear.weight * omaga_c_1.T * model_c_1.linear.weight
                    - gamma_a_1 * model_c_1.forward(allhunters[i].z_1)[1].T * model_c_1.forward(allhunters[i].z_1)[
                        1] * model_a_1.linear.weight)

        # actor 网络更新
        output_c_2 = model_c_1(allhunters[i].z_2)
        dot_v_2 = 2 * zeta_2 * allhunters[i].z_2 + output_c_2
        out_put_a_2 = model_a_2(allhunters[i].z_2)
        u_hat = -zeta_2 * allhunters[i].z_2 - 1 / 2 * out_put_a_2
        omaga_c_2 = -model_c_2.forward(allhunters[i].z_2)[1] * [
            allhunters[i].f_v - zeta_2 * allhunters[i].z_2 - 1 / 2 * out_put_a_2 - a_hat_dot]
        model_c_2.linear.weight += (-gamma_c_2 / (1 + omaga_c_2 * omaga_c_2.T) * omaga_c_2 * [
            2 * zeta_2 * allhunters[i].z_2.T * (allhunters[i].f_v - a_hat_dot)
            - (zeta_2 * zeta_2 - 1) * allhunters[i].z_2.T * allhunters[
                i].z_2 + 1 / 4 * out_put_a_2 * out_put_a_2.T + omaga_c_2.T * model_c_2.linear.weight])
        model_a_2.linear.weight += (
                    1 / 2 * model_c_2.forward(allhunters[i].z_2)[1].T * allhunters[i].z_2 + gamma_c_2 / (
                    4 * (1 + omaga_c_2 * omaga_c_2.T))
                    * model_c_2.forward(allhunters[i].z_2)[1].T * model_c_2.forward(allhunters[i].z_2)[
                        1] * model_a_2.linear.weight * omaga_c_2.T * model_c_2.linear.weight
                    - gamma_a_2 * model_c_2.forward(allhunters[i].z_2)[1].T * model_c_2.forward(allhunters[i].z_2)[
                        1] * model_a_2.linear.weight)





