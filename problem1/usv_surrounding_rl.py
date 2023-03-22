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
zeta_1 =1
pian_dot_v1_0 =1

T = 0.1

class model():

    def __init__(self,x,y,ang,u,v,r):
        self.x = x
        self.y = y
        self.ang = ang
        self.u = u
        self.v = v
        self.r = r
        self.V = np.array([u,v,r]).T

    def get_rho_and_ang(self,x_0,y_0):
        self.rho = np.sqrt((x_0 - self.x) * (x_0 - self.x) + (y_0 - self.y) * (y_0 - self.y))
        self.angle = atan2(y_0 - self.y, x_0 - self.x)
        self.q = np.array([cos(self.angle), -sin(self.angle)], [sin(self.angle), cos(self.angle)])

    def change_position(self,tau_1,tau_2):
        u_next = (tau_1-D_11*self.u+M_22*self.v*self.r)/M_11*T
        v_next = (-D_22*self.v-M_11*self.u*self.r)/M_22*T
        r_next = (tau_2-D_33*self.r-(M_22-M_11)*self.u*self.v)/M_33*T
        # 运动学模型 （1）
        self.u += u_next
        self.v += v_next
        self.r += r_next
        self.x += (self.u*cos(self.ang) - self.v*sin(self.ang))*T
        self.y += (self.u*sin(self.ang) + self.v*cos(self.ang))*T
        self.ang += self.r*T

    def z_and_z_dot(self,angle_up,angle_down,rho_up,rho_down,u_up,u_down,v_up,v_down,v_x0,v_y0):
        self.z_1 = (self.rho - rho_0) ** 2 + (2 * self.angle - angle_up - angle_down) ** 2 + (self.ang + pi/2 - self.angle)
        self.alpha_1 = 2 * (self.rho - rho_0) * (np.array([[cos(self.angle),sin(self.angle)]]))
        self.beta_1 = (4 * (2 * self.angle - angle_up - angle_down) * (np.array([[-sin(self.angle),cos(self.angle)]]))) / self.rho
        self.beta_2 = (2 * (2 * self.angle - angle_up - angle_down) * (np.array([[-sin(angle_up),cos(angle_up)]]))) / rho_up
        self.beta_3 = (2 * (2 * self.angle - angle_up - angle_down) * (np.array([[-sin(angle_down),cos(angle_down)]]))) / rho_down
        self.gamma_1 = 2 * (self.ang + pi/2 - self.angle)
        self.gamma_2 = (2 * (self.ang + pi/2 - self.angle) * (np.array([[-sin(self.angle),cos(self.angle)]]))) / self.rho
        q_up = np.array([[cos(angle_up), -sin(angle_up)], [sin(angle_up), cos(angle_up)]])
        q_down = np.array([[cos(angle_down), -sin(angle_down)],[sin(angle_down), cos(angle_down)]])
        self.delta = self.beta_2 * q_up * np.array([u_up,v_up]) + self.beta_3 * q_down * np.array([u_down,v_down])
        self.lambda_1 = np.array([(self.alpha_1 + self.beta_1 - self.gamma_1) * self.q,self.gamma_1])
        self.lambda_2 = (self.alpha_1 + self.beta_1 - self.beta_2 - self.beta_3) * np.array([v_x0,v_y0])
        self.dot_z_1 = self.lambda_1 * np.array([self.u,self.v,self.r]) * self.lambda_2

    def get_optimal_a(self):
        r1 = self.z_1 * self.z_1 + self.V.T * R * self.V
        a_star = np.linalg.inv(R) * self.lambda_1.T *(-zeta_1 * self.z_1 - 1/2 * pian_dot_v1_0 )

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
        self.u = v_x*cos(self.ang)+v_y*sin(self.ang)
        self.v = -v_x*sin(self.ang)+v_y*cos(self.ang)

    def change_position(self):
        self.x += self.v_x * T
        self.y += self.v_y * T
