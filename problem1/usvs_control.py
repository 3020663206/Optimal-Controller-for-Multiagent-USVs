import numpy as np
import model_def
import agent_def
import adp_drl_nn
from math import atan,sin,cos,pi

NumberofHunters = 3

Timeoftheworld = 0

SimulationLimits = 1000000

allhunters = [None, None, None]

agent_invader = [None]

all_critic_1 = [None, None, None]
all_critic_2 = [None, None, None]
all_actor_1 = [None, None, None]
all_actor_2 = [None, None, None]

z_1_set = [None, None, None]
z_2_set = [None, None, None]

def create_world():

    allhunters[0] = model_def.Hunter(0, 1, 7, atan(4), 0, 0, 0)
    allhunters[1] = model_def.Hunter(1, 5, 1, atan(1 / 4), 0, 0, 0)
    allhunters[2] = model_def.Hunter(2, 2, 8, atan(1 / 3), 0, 0, 0)

    agent_invader[0] = model_def.Invader(0, 8, 6, atan(1.0 / 5))

    for i in range(NumberofHunters):

        z_1_set[i] = []
        z_2_set[i] = []

        all_critic_1[i] = adp_drl_nn.Critic1_NN(72, 1)
        all_actor_1[i] = adp_drl_nn.Actor1_NN(72, 1)
        all_critic_2[i] = adp_drl_nn.Critic2_NN(72, 2)
        all_actor_2[i] = adp_drl_nn.Actor2_NN(72, 2)

def change_state(Timeoftheworld):

    for i in range(NumberofHunters):
        # 改变每个智能体的状态
        allhunters[i].change_position(allhunters[i].u_hat[0], allhunters[i].u_hat[1])
        # 计算智能体与目标的距离和角度
        allhunters[i].get_distance_and_angle(agent_invader[0].pos_x, agent_invader[0].pos_y)

    agent_invader[0].change_speed(0.2, 0.1)
    agent_invader[0].change_position()

def change_network(Timeoftheworld):

    for i in range(NumberofHunters):

        # 得到智能体的邻居
        neighbor_id = allhunters[i].get_sametype_neighbors(allhunters)

        # 计算各种系数,并得到z_1
        allhunters[i].calculate_super_error(allhunters[neighbor_id[0]].angle, allhunters[neighbor_id[1]].angle,
                                                allhunters[neighbor_id[0]].distance, allhunters[neighbor_id[1]].distance,
                                                allhunters[neighbor_id[0]].speed_u, allhunters[neighbor_id[1]].speed_u,
                                                allhunters[neighbor_id[0]].speed_v, allhunters[neighbor_id[1]].speed_v,
                                                agent_invader[0].speed_u, agent_invader[0].speed_v)

        # RBF网络得到近似值 Step1
        z_1_set[i].append(np.norm(allhunters[i].z_1))

        all_critic_1[i].forward(allhunters[i].z_1)
        all_actor_1[i].forward(allhunters[i].z_1)

        allhunters[i].calculate_virtual_opitmal(all_actor_1[i].finaloutputs)

        all_critic_1[i].backward(allhunters[i],all_actor_1[i])
        all_actor_1[i].backward(allhunters[i],all_critic_1[i])

        # 计算z_2
        allhunters[i].calculate_sub_error()

        # RBF网络得到近似值 Step2
        z_2_set[i].append(np.norm(allhunters[i].z_2))

        all_critic_2[i].forward(allhunters[i].z_2)
        all_actor_2[i].forward(allhunters[i].z_2)

        allhunters[i].calculate_actual_opitmal(all_actor_2[i].finaloutputs)

        all_critic_2[i].backward(allhunters[i], all_actor_2[i], allhunters[i].optimal_V_hat_dot)
        all_actor_2[i].backward(allhunters[i], all_critic_2[i], allhunters[i].optimal_V_hat_dot)

def train_world():

    numbercount = 0

    for i in range(SimulationLimits):

        global Timeoftheworld

        change_state(Timeoftheworld)
        Timeoftheworld += model_def.T_changeState

        numbercount += 1

        if(numbercount == (model_def.T_changeNN / model_def.T_changeState)):

            change_network(Timeoftheworld)
            numbercount = 0


if __name__ == "__main__":

    create_world()
    train_world()
