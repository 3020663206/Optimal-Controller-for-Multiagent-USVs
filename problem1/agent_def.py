import math


def get_abs_min_key(dct):
    return min(dct, key=lambda k: abs(dct[k]))

def get_abs_max_key(dct):
    return max(dct, key=lambda k: abs(dct[k]))


class Agent:
    def __init__(self, id, x, y, angle):
        self.front_neighbor = None
        self.back_neighbor = None
        self.id = id
        self.x = x
        self.y = y
        self.angle = angle
        self.front_neighbors_sets = {}
        self.back_neighbors_sets = {}
        self.front_neighbors_sets_1 = {}
        self.back_neighbors_sets_1 = {}
        self.front_neighbors_sets_2 = {}
        self.back_neighbors_sets_2 = {}

    def get_neighbors(self, agents):
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
    agents.append(Agent(0, 0, 0, 30))
    agents.append(Agent(1, 1, 1, -60))
    agents.append(Agent(2, 2, 2, 90))
    agents.append(Agent(3, 3, 3, 135))
    agents.append(Agent(4, 4, 4, -95))
    agents.append(Agent(5, 5, 5, -34))
    agents.append(Agent(6, 6, 6, 60))
    agents.append(Agent(7, 7, 7, -61))
    for agent in agents:
        agent.get_neighbors(agents)
        print(f"Agent {agent.id} front neighbors: {agent.front_neighbor}")
        print(f"Agent {agent.id} back neighbors: {agent.back_neighbor}")
