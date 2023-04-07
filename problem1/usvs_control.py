#  使用深度强化学习实现无人艇集群合围算法

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from math import pi, atan2, sqrt, sin, cos, atan
import torch
import model_def
import rbf_network


#model_1 = model(0, 0, 0, 0, 0, 0)
#model_2 = model(0, 0, 0, 0, 0, 0)
#model_3 = model(0, 0, 0, 0, 0, 0)

allhunters = [model_def.Hunter(0, 0, 0, 0, 0, 0) for i in range(model_def.NumberofHunters)]

model_c_1 = rbf_network.RBFN_type1(model_def.centers_c_1, model_def.n_out_c_1)
model_a_1 = rbf_network.RBFN_type1(model_def.centers_a_1, model_def.n_out_a_1)
model_c_2 = rbf_network.RBFN_type1(model_def.centers_c_2, model_def.n_out_c_2)
model_a_2 = rbf_network.RBFN_type1(model_def.centers_a_2, model_def.n_out_a_2)

epochs = 500
a_hat_dot = 0
a_hat = 0

for epoch in range(epochs):

    for i in range(model_def.NumberofHunters):
        # Step 1
        out_put_c_1 = model_c_1.forward(allhunters[i].z_1)[0]
        dot_v_1 = 2 * model_def.zeta_1 * allhunters[i].z_1 + out_put_c_1
        out_put_a_1 = model_a_1.forward(allhunters[i].z_1)[0]
        a_hat_last = a_hat
        a_hat = np.linalg.inv(model_def.R) * allhunters[i].lambda_1 * (-model_def.zeta_1 * allhunters[i].z_1 - 1 / 2 * out_put_a_1)
        omaga_c_1 = -model_c_1.forward(allhunters[i].z_1)[1] * [allhunters[i].a * model_def.zeta_1 * allhunters[i].z_1 + 1 / 2 * allhunters[i].a * out_put_a_1 + allhunters[i].lambda_2]
        # critic 网络更新
        model_c_1.linear.weight += (-model_def.gamma_c_1 / (1 + omaga_c_1 * omaga_c_1.T) * omaga_c_1 * [-2 * model_def.zeta_1 * allhunters[i].z_1.T * allhunters[i].lambda_2
            - (allhunters[i].a * model_def.zeta_1 * model_def.zeta_1 - 1) * allhunters[i].z_1.T * allhunters[i].z_1 + 1 / 4 * allhunters[i].a * out_put_a_1 * out_put_a_1.T + omaga_c_1.T * model_c_1.linear.weight])
        # actor 网络更新
        model_a_1.linear.weight += (1 / 2 * model_c_1.forward(allhunters[i].z_1)[1].T * allhunters[i].z_1 + model_def.gamma_c_1 / (4 * (1 + omaga_c_1 * omaga_c_1.T))
                    * model_c_1.forward(allhunters[i].z_1)[1].T * model_c_1.forward(allhunters[i].z_1)[1] * model_a_1.linear.weight * omaga_c_1.T * model_c_1.linear.weight
                    - model_def.gamma_a_1 * model_c_1.forward(allhunters[i].z_1)[1].T * model_c_1.forward(allhunters[i].z_1)[1] * model_a_1.linear.weight)

        # Step 2
        a_hat_dot = (a_hat - a_hat_last)/model_def.T_changeNN
        output_c_2 = model_c_1.forward(allhunters[i].z_2)[0]
        dot_v_2 = 2 * model_def.zeta_2 * allhunters[i].z_2 + output_c_2
        out_put_a_2 = model_a_2.forward(allhunters[i].z_2)[0]
        u_hat = -model_def.zeta_2 * allhunters[i].z_2 - 1 / 2 * out_put_a_2
        omaga_c_2 = -model_c_2.forward(allhunters[i].z_2)[1] * [allhunters[i].f_v - model_def.zeta_2 * allhunters[i].z_2 - 1 / 2 * out_put_a_2 - a_hat_dot]
        # critic 网络更新
        model_c_2.linear.weight += (-model_def.gamma_c_2 / (1 + omaga_c_2 * omaga_c_2.T) * omaga_c_2 * [2 * model_def.zeta_2 * allhunters[i].z_2.T * (allhunters[i].f_v - a_hat_dot)
            - (model_def.zeta_2 * model_def.zeta_2 - 1) * allhunters[i].z_2.T * allhunters[i].z_2 + 1 / 4 * out_put_a_2 * out_put_a_2.T + omaga_c_2.T * model_c_2.linear.weight])
        # actor 网络更新
        model_a_2.linear.weight += (1 / 2 * model_c_2.forward(allhunters[i].z_2)[1].T * allhunters[i].z_2 + model_def.gamma_c_2 / (4 * (1 + omaga_c_2 * omaga_c_2.T))
                    * model_c_2.forward(allhunters[i].z_2)[1].T * model_c_2.forward(allhunters[i].z_2)[1] * model_a_2.linear.weight * omaga_c_2.T * model_c_2.linear.weight
                    - model_def.gamma_a_2 * model_c_2.forward(allhunters[i].z_2)[1].T * model_c_2.forward(allhunters[i].z_2)[1] * model_a_2.linear.weight)





