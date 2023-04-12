import numpy as np
import model_def
import agent_def
import adp_drl_nn
from math import atan,sin,cos,pi

NumberofHunters = 3
center_numbers = 72

agent_hunter_0 = model_def.Hunter(0, 1, 7, atan(4), 0, 0, 0)
agent_hunter_1 = model_def.Hunter(1, 5, 1, atan(1/4), 0, 0, 0)
agent_hunter_2 = model_def.Hunter(2, 2, 8, atan(1/3), 0, 0, 0)
allhunters = []
allhunters.append(agent_hunter_0)
allhunters.append(agent_hunter_1)
allhunters.append(agent_hunter_2)

agent_invader = model_def.Invader(0,8, 6, atan(1.0/5))

all_critic_1 = []
all_critic_2 = []
all_actor_1 = []
all_actor_2 = []

hunter_0_critic_1 = adp_drl_nn.Critic1_NN(center_numbers)
hunter_0_actor_1 = adp_drl_nn.Actor1_NN(center_numbers)
hunter_0_critic_2 = adp_drl_nn.Critic2_NN(center_numbers)
hunter_0_actor_2 = adp_drl_nn.Actor2_NN(center_numbers)


hunter_1_critic_1 = adp_drl_nn.Critic1_NN(center_numbers)
hunter_1_actor_1 = adp_drl_nn.Actor1_NN(center_numbers)
hunter_1_critic_2 = adp_drl_nn.Critic2_NN(center_numbers)
hunter_1_actor_2 = adp_drl_nn.Actor2_NN(center_numbers)

hunter_2_critic_1 = adp_drl_nn.Critic1_NN(center_numbers)
hunter_2_actor_1 = adp_drl_nn.Actor1_NN(center_numbers)
hunter_2_critic_2 = adp_drl_nn.Critic2_NN(center_numbers)
hunter_2_actor_2 = adp_drl_nn.Actor2_NN(center_numbers)

all_critic_1.append(hunter_0_critic_1)
all_critic_1.append(hunter_1_critic_1)
all_critic_1.append(hunter_2_critic_1)

all_actor_1.append(hunter_0_actor_1)
all_actor_1.append(hunter_1_actor_1)
all_actor_1.append(hunter_2_actor_1)

all_critic_2.append(hunter_0_critic_2)
all_critic_2.append(hunter_1_critic_2)
all_critic_2.append(hunter_2_critic_2)

all_actor_2.append(hunter_0_actor_2)
all_actor_2.append(hunter_1_actor_2)
all_actor_2.append(hunter_2_actor_2)



z_1_set = []
z_2_set = []
optimal_V_hat = 0
for
    for i in range(NumberofHunters):
        # 计算智能体与目标的距离和角度
        allhunters[i].get_distance_and_angle(agent_invader.pos_x,agent_invader.pos_y)
    for i in range(NumberofHunters):
        # 得到智能体的邻居
        neighbor_id = allhunters[i].get_sametype_neighbors(allhunters)
        # 计算各种系数,并得到z_1
        z_i_1 = allhunters[i].calculate_super_error(allhunters[neighbor_id[0]].angle,allhunters[neighbor_id[1]].angle,
                                                     allhunters[neighbor_id[0]].angle,allhunters[neighbor_id[1]].angle,
                                                     allhunters[neighbor_id[0]].speed_u,allhunters[neighbor_id[1]].speed_u,
                                                     allhunters[neighbor_id[0]].speed_v,allhunters[neighbor_id[1]].speed_v,
                                                     agent_invader.speed_u,agent_invader.speed_v)
        z_1_set.append(z_i_1)
        # RBF网络得到近似值\
        # step1
        all_critic_1[i].forward(allhunters[i].z_1)
        all_actor_1[i].forward(allhunters[i].z_1)
        a_hat_last = optimal_V_hat
        optimal_V_hat = np.linalg.inv(model_def.R_rewardweights) * allhunters[i].gamma_1.T * (-adp_drl_nn.zeta_1 * allhunters[i].z_1 - 1/2 * all_actor_1[i].forward(allhunters[i].z_1))
        all_critic_1[i].backward(allhunters[i],all_actor_1[i])
        all_actor_1[i].backward(allhunters[i],all_critic_1[i])
        # step2

        allhunters[i].calculate_sub_error(optimal_V_hat)
        z_2_set.append(allhunters[i].z_2)
        all_critic_2[i].forward(allhunters[i].z_2)
        all_actor_2[i].forward(allhunters[i].z_2)
        u_hat = -adp_drl_nn.zeta_2 * allhunters[i].z_2 - 1/2 * all_actor_2[i].finaloutputs
        a_hat_dot = (optimal_V_hat - a_hat_last) / model_def.T_changeNN
        all_critic_2[i].backward(allhunters[i],all_actor_2[i],a_hat_dot)
        all_actor_2[i].backward(allhunters[i],all_critic_2[i],a_hat_dot)
        # 重新得到状态信息
        allhunters[i].change_position(u_hat[0],u_hat[1])
        agent_invader.change_position()

