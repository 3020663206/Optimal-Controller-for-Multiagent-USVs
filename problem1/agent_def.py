import numpy as np
from math import pi, atan2, sqrt, sin, cos, atan


def get_abs_min_key(dct):
    return min(dct, key=lambda k: abs(dct[k]))

def get_abs_max_key(dct):
    return max(dct, key=lambda k: abs(dct[k]))


class Agent(object):
    def __init__(self, id, pos_x, pos_y):
        self.front_neighbor = None
        self.back_neighbor = None
        self.id = id
        self.pos_x = pos_x
        self.pos_y = pos_y
        #self.angle = angle
        self.front_neighbors_sets = {}
        self.back_neighbors_sets = {}
        self.front_neighbors_sets_1 = {}
        self.back_neighbors_sets_1 = {}
        self.front_neighbors_sets_2 = {}
        self.back_neighbors_sets_2 = {}

    def get_distance_and_angle(self, target_x, target_y):
        self.distance = np.sqrt((target_x - self.pos_x) * (target_x - self.pos_x) + (target_y - self.pos_y) * (target_y - self.pos_y))
        self.angle = (180/pi) * atan2(self.pos_y - target_y, self.pos_x - target_x)
        #self.Q_transform = np.array([cos(self.angle), -sin(self.angle)], [sin(self.angle), cos(self.angle)])

    def get_sametype_neighbors(self, agents):
        self.front_neighbors_sets = {}
        self.back_neighbors_sets = {}
        self.front_neighbors_sets_1 = {}
        self.back_neighbors_sets_1 = {}
        self.front_neighbors_sets_2 = {}
        self.back_neighbors_sets_2 = {}
        for agent in agents:
            if agent.id != self.id:
                angle_diff = self.angle - agent.angle
                if self.angle >= 0:
                    if self.angle - 180 <= angle_diff <= 0:
                        self.front_neighbors_sets_1[agent.id] = angle_diff
                    elif 180 < angle_diff <= 180 + self.angle:
                        self.front_neighbors_sets_2[agent.id] = angle_diff
                    elif 0 < angle_diff < 180:
                        self.back_neighbors_sets[agent.id] = angle_diff
                else:
                    if -180 <= angle_diff <= 0:
                        self.front_neighbors_sets[agent.id] = angle_diff
                    elif self.angle - 180 <= angle_diff < -180:
                        self.back_neighbors_sets_2[agent.id] = angle_diff
                    elif 0 < angle_diff <= 180 + self.angle:
                        self.back_neighbors_sets_1[agent.id] = angle_diff
        if self.angle >= 0:
            if self.front_neighbors_sets_1 == {}:
                self.front_neighbor = get_abs_max_key(self.front_neighbors_sets_2)
            else:
                self.front_neighbor = get_abs_min_key(self.front_neighbors_sets_1)
            self.back_neighbor = get_abs_min_key(self.back_neighbors_sets)
        elif  self.angle < 0:
            if self.back_neighbors_sets_1 == {}:
                self.back_neighbor = get_abs_max_key(self.back_neighbors_sets_2)
            else:
                self.back_neighbor = get_abs_min_key(self.back_neighbors_sets_1)
            self.front_neighbor = get_abs_min_key(self.front_neighbors_sets)


if __name__ == "__main__":
    agents = []
    agents.append(Agent(0, 3, 4))
    agents.append(Agent(1, 4, -6))
    agents.append(Agent(2, 2, 4))
    agents.append(Agent(3, -3, 3))
    agents.append(Agent(4, 6, -5))
    agents.append(Agent(5, 4, 2))
    agents.append(Agent(6, -6, 6))
    agents.append(Agent(7, 7, -3))
    for agent_i in agents:
        agent_i.get_distance_and_angle(0, 0)
    for agent_i in agents:
        agent_i.get_sametype_neighbors(agents)
        print(f"Agent {agent_i.id} front neighbors: {agent_i.front_neighbor}")
        print(f"Agent {agent_i.id} back neighbors: {agent_i.back_neighbor}")