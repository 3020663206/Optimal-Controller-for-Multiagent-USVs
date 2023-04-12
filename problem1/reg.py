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