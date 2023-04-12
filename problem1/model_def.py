import numpy as np
from math import pi, atan2, sin, cos

import agent_def

M_11 = 1.956
M_22 = 2.405
M_33 = 0.043
D_11 = 2.436
D_22 = 12.992
D_33 = 0.0564

rho_0 = 3
R_rewardweights = np.eye(3)

T_changeNN = 0.1
T_changeState = 0.01

class Hunter(agent_def.Agent):

    def __init__(self, id, pos_x, pos_y, orientation, speed_u, speed_v, speed_r):
        super().__init__(id, pos_x, pos_y)
        self.orientation = orientation
        self.speed_u = speed_u
        self.speed_v = speed_v
        self.speed_r = speed_r
        self.vector_V = np.array([speed_u, speed_v, speed_r]).T
        self.a = 0

    def change_position(self, tau_1, tau_2):
        u_next = (tau_1 - D_11 * self.speed_u + M_22 * self.speed_v * self.speed_r) / M_11 * T_changeState
        v_next = (-D_22 * self.speed_v - M_11 * self.speed_u * self.speed_r) / M_22 * T_changeState
        r_next = (tau_2 - D_33 * self.speed_r - (M_22 - M_11) * self.speed_u * self.speed_v) / M_33 * T_changeState
        # 运动学模型 （1）
        self.speed_u += u_next
        self.speed_v += v_next
        self.speed_r += r_next
        self.pos_x += (self.speed_u * cos(self.orientation) - self.speed_v * sin(self.orientation)) * T_changeState
        self.pos_y += (self.speed_u * sin(self.orientation) + self.speed_v * cos(self.orientation)) * T_changeState
        self.orientation += self.speed_r * T_changeState
        self.function_V = np.array([(-D_11 * self.speed_u + M_22 * self.speed_v * self.speed_r) / M_11],
                                   [(-D_22 * self.speed_v - M_11 * self.speed_u * self.speed_r) / M_22],
                                   [(-D_33 * self.speed_r - (M_22 - M_11) * self.speed_u * self.speed_v) / M_33])

    def calculate_super_error(self, angle_front, angle_behind, rho_front, rho_behind, u_front, u_behind, v_front, v_behind, v_target_x, v_target_y):

        self.z_1 = (self.distance - rho_0) ** 2 + (2 * self.angle - angle_front - angle_behind) ** 2 + (self.orientation + pi / 2 - self.angle)

        self.alpha_1 = 2 * (self.distance - rho_0) * (np.array([[cos(self.angle), sin(self.angle)]]))
        #print(self.alpha_1)

        self.beta_1 = (4 * (2 * self.angle - angle_front - angle_behind) * (np.array([[-sin(self.angle), cos(self.angle)]]))) / self.distance
        self.beta_2 = (2 * (2 * self.angle - angle_front - angle_behind) * (np.array([[-sin(angle_front), cos(angle_behind)]]))) / rho_front
        self.beta_3 = (2 * (2 * self.angle - angle_front - angle_behind) * (np.array([[-sin(angle_behind), cos(angle_behind)]]))) / rho_behind
        #print(self.beta_1)
        #print(self.beta_2)
        #print(self.beta_3)
        self.gamma_1 = 2 * (self.orientation + pi / 2 - self.angle)
        self.gamma_2 = (2 * (self.orientation + pi / 2 - self.angle) * (np.array([[-sin(self.angle), cos(self.angle)]]))) / self.distance
        #print(self.gamma_1)
        #print(self.gamma_2)
        Q_transform_front = np.array([[cos(angle_front), -sin(angle_front)], [sin(angle_front), cos(angle_front)]])
        Q_transform_behind = np.array([[cos(angle_behind), -sin(angle_behind)], [sin(angle_behind), cos(angle_behind)]])

        self.delta = self.beta_2 * Q_transform_front * np.array([u_front, v_front]) + self.beta_3 * Q_transform_behind * np.array([u_behind, v_behind])

        self.lambda_1 = np.array([(self.alpha_1 + self.beta_1 - self.gamma_1) * self.Q_transform, self.gamma_1])
        print(self.Q_transform)
        print(self.Q_transform.shape)
        print((self.alpha_1 + self.beta_1 - self.gamma_1))
        print((self.alpha_1 + self.beta_1 - self.gamma_1).shape)
        print(((self.alpha_1 + self.beta_1 - self.gamma_1) * self.Q_transform))
        print(((self.alpha_1 + self.beta_1 - self.gamma_1) * self.Q_transform).shape)
        print(self.gamma_1)
        print(self.lambda_1)
        print(self.lambda_1.shape)
        self.lambda_2 = (self.alpha_1 + self.beta_1 - self.beta_2 - self.beta_3) * np.array([v_target_x, v_target_y])

        self.dot_z_1 = self.lambda_1 * np.array([self.speed_u, self.speed_v, self.speed_r]) * self.lambda_2

        self.a = self.lambda_1 * np.linalg.inv(R_rewardweights) * self.lambda_1.T

        return self.z_1, self.dot_z_1

    def calculate_sub_error(self, optimal_V_hat):

        self.z_2 = self.vector_V - optimal_V_hat
        return self.z_2

class Invader(agent_def.Agent):

    pos_x, pos_y, orientation = 0, 0, 0
    speed_x_axis, speed_y_axis = 0, 0
    speed_u, speed_v = 0, 0

    def __init__(self, id, pos_x, pos_y, orientation):
        super().__init__(id, pos_x, pos_y)
        self.orientation = orientation

    def change_speed(self, speed_x_axis, speed_y_axis):
        self.speed_x_axis = speed_x_axis
        self.speed_y_axis = speed_y_axis
        self.ang = atan2(speed_y_axis, speed_x_axis)
        self.speed_u = speed_x_axis * cos(self.orientation) + speed_y_axis * sin(self.orientation)
        self.speed_v = -speed_x_axis * sin(self.orientation) + speed_y_axis * cos(self.orientation)

    def change_position(self):
        self.pos_x += self.speed_x_axis * T_changeState
        self.pos_y += self.speed_y_axis * T_changeState