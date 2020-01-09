import queue as q

import numpy as np
from matplotlib import pyplot as plt


class MultiHeuristicPlanner(object):
    def __init__(self, planning_env, guidance, w1=20, w2=1.5):
        """

        :param planning_env: The planning environment for the algorithm
        :param guidance: a list of tuples containing the user guidance points
        :param w1: inflation parameter of individual searches
        :param w2: The factor of comparison between anchor heuristic and inadmissible heuristics 
        """
        self.guidance = guidance
        self.planning_env = planning_env
        self.nodes = dict()

    '''
    This function assume the input start_config and goal_config matches the start and goal
    in the environment object.
    '''

    def Plan(self, start_config, goal_config):
        plan = []

        # TODO (student): Implement your planner here.

        plan.append(start_config)
        plan.append(goal_config)

        return np.array(plan)
