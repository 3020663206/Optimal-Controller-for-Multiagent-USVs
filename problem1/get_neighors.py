class Agent:
    def __init__(self, id, x, y, angle):
        self.id = id
        self.x = x
        self.y = y
        self.angle = angle
        self.front_neighbors = []
        self.back_neighbors = []
    def get_neighbors(self, agents, front_angle_threshold, back_angle_threshold):
        self.front_neighbors = []
        self.back_neighbors = []
        for agent in agents:
            if agent.id != self.id:
                angle_diff = abs(self.angle - agent.angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                if angle_diff <= front_angle_threshold:
                    self.front_neighbors.append(agent)
                elif angle_diff <= back_angle_threshold:
                    self.back_neighbors.append(agent)
if __name__ == "__main__":
    agents = []
    agents.append(Agent(0, 0, 0, 0))
    agents.append(Agent(1, 1, 1, 45))
    agents.append(Agent(2, 2, 2, 90))
    agents.append(Agent(3, 3, 3, 135))
    agents.append(Agent(4, 4, 4, 180))
    agents.append(Agent(5, 5, 5, 225))
    agents.append(Agent(6, 6, 6, 270))
    agents.append(Agent(7, 7, 7, 315))
    for agent in agents:
        agent.get_neighbors(agents, 45, 135)
        print(f"Agent {agent.id} front neighbors: {[neighbor.id for neighbor in agent.front_neighbors]}")
        print(f"Agent {agent.id} back neighbors: {[neighbor.id for neighbor in agent.back_neighbors]}")