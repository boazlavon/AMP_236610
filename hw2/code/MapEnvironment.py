import numpy as np
from IPython import embed
from matplotlib import pyplot as plt
import time

from RRTPlanner import RRTPlanner

class MapEnvironment(object):
    
    def __init__(self, mapfile, start, goal, map_resolution = 0.5):

        # Obtain the boundary limits.
        # Check if file exists.
        self.map = np.loadtxt(mapfile)
        self.xlimit = [1, np.shape(self.map)[0]] # TODO (avk): Check if this needs to flip.
        self.ylimit = [1, np.shape(self.map)[1]]
        self.map = np.pad(self.map, 1) # for graph visualization

        # Check if start and goal are within limits and collision free
        if not self.state_validity_checker(start) or not self.state_validity_checker(goal):
            raise ValueError('Start and Goal state must be within the map limits - x:{} y:{}'.format(self.xlimit, self.ylimit));
            exit(0)
        self.start = start
        self.goal = goal
        self.map_resolution = map_resolution
        self.valid_edge_cache = {}
        self.obstacles = self._get_obstacles()

    def compute_distance(self, start_config, end_config):
        
        #
        # TODO: Implement a function which computes the distance between
        # two configurations.
        #
        return np.linalg.norm(np.array(start_config) - np.array(end_config))


    def state_validity_checker(self, config, check_obst=True):

        #
        # TODO: Implement a state validity checker
        # Return true if valid.
        #
        if not (self.xlimit[0] <= config[0] < self.xlimit[1] and self.ylimit[0] <= config[1] < self.ylimit[1]):
            return False

        if check_obst:
            if self.map[config[0],config[1]] == 1:
                return False
        return True

    def edge_validity_checker(self, config1, config2):

        #
        # TODO: Implement an edge validity checker
        #
        #
        from_node = config1
        to_node   = config2
        if (from_node, to_node) in self.valid_edge_cache:
            return self.valid_edge_cache[(from_node, to_node)]

        R, theta = RRTPlanner.calc_distance_and_angle(from_node, to_node)
        n_iter = int(np.floor(R / self.map_resolution)) + 1
        path = [(from_node.x + self.map_resolution * np.cos(theta) * i,  from_node.y + self.map_resolution * np.sin(theta) * i)\
                for i in range(n_iter)]
        diffs = [(abs(ox - iter_x) < 0.5 and abs(oy - iter_y) < 0.5)\
                 for (iter_x, iter_y) in path\
                 for (ox, oy) in self.obstacles]
        # if diffs has one entry of True - there is a collision.      
        self.valid_edge_cache[(from_node, to_node)] = not (any(diffs))
        return self.valid_edge_cache[(from_node, to_node)]


    def compute_heuristic(self, config):
        
        #
        # TODO: Implement a function to compute heuristic.
        #
        return self.compute_distance(config,self.goal)

    def _get_obstacles(self):
        result = np.where(self.map == 1)
        obstacles = list(zip(result[0], result[1]))
        return obstacles

    def visualize_plan(self, plan, filename=None, title=None):
        '''
        Visualize the final path
        @param plan Sequence of states defining the plan.
        '''
        plt.imshow(self.map, interpolation='nearest')
        plt.plot(self.start[1], self.start[0], "oc")
        plt.plot(self.goal[1], self.goal[0], "oc")
        for (c_cord, n_cord) in zip(plan, plan[1:]):
            dx = (c_cord[0], n_cord[0])
            dy = (c_cord[1], n_cord[1])
            plt.plot(dy, dx, 'g')

        if title is not None:
            plt.xlabel(title, fontsize=12)
        if filename is not None:
            plt.savefig('{}.png'.format(filename))
        plt.show()

    def calc_plan_cost(self, plan):
        cost = 0
        for (c_cord, n_cord) in zip(plan, plan[1:]):
            dx = (c_cord[0] - n_cord[0])
            dy = (c_cord[1] - n_cord[1])
            c = (dx**2 + dy**2)**0.5
            cost += c
        return cost