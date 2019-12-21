import numpy
from RRTTree import RRTTree
from RRTPlanner import RRTPlanner

class RRTStarPlanner(RRTPlanner):

    def __init__(self, planning_env, k=4, goal_sample_rate=0.2):
            super().__init__(planning_env, goal_sample_rate)
            self.k = k

    def Plan(self, start_config, goal_config, sample_dist = 5, max_iter = 1000):
        return super().Plan(start_config, goal_config, sample_dist = sample_dist, k = self.k, max_iter = max_iter)