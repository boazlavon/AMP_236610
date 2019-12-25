import numpy as np
from IPython import embed
from matplotlib import pyplot as plt
import time

from RRTPlanner import RRTPlanner

class MapEnvironment(object):
    
    def __init__(self, mapfile, start, goal):

        # Obtain the boundary limits.
        # Check if file exists.
        self.map = np.loadtxt(mapfile)
        self.xlimit = [1, np.shape(self.map)[0] - 1]
        self.ylimit = [1, np.shape(self.map)[1] - 1]
        self.map = np.pad(self.map, 1) # for graph visualization

        # Check if start and goal are within limits and collision free
        if not self.state_validity_checker(start) or not self.state_validity_checker(goal):
            raise ValueError('Start and Goal state must be within the map limits - x:{} y:{}'.format(self.xlimit, self.ylimit));
            exit(0)
        self.start = start
        self.goal = goal
        self.valid_edge_cache = {}
        self.obstacles = self._get_obstacles()

    def compute_distance(self, start_config, end_config):
        
        #
        # TODO: Implement a function which computes the distance between
        # two configurations.
        #
        x0, y0 = start_config
        x1, y1 = end_config
        dx = x0 - x1
        dy = y0 - y1
        return ((dx**2 + dy**2)**0.5)

    def state_validity_checker(self, config, check_obst=True):

        #
        # TODO: Implement a state validity checker
        # Return true if valid.
        #
        x, y = config
        if not (self.xlimit[0] <= x <= self.xlimit[1] and\
                 self.ylimit[0] <= y <= self.ylimit[1]):
            return False

        if check_obst:
            if self.map[self._find_square(x), self._find_square(y)] == 1:
                return False

        return True


    def _find_square(self, x):
        if x - int(x) > 0.5:
            x = int(x) + 1
        else:
            x = int(x)
        return x

    def edge_validity_checker(self, config1, config2):
        configs = tuple((config1, config2))
        configs_flipped = tuple((config2, config1))

        if configs in self.valid_edge_cache:
            return self.valid_edge_cache[configs]

        if configs_flipped in self.valid_edge_cache:
            return self.valid_edge_cache[configs_flipped]
        
        x0, y0 = self._find_square(config1[0]), self._find_square(config1[1])
        x1, y1 = self._find_square(config2[0]), self._find_square(config2[1])

        squares = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x = x0
        y = y0
        n = 1 + dx + dy
        x_inc = 1 if (x1 > x0) else -1
        y_inc = 1 if (y1 > y0) else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        while n > 0:
            squares.append((x, y))

            if (error > 0):
                x += x_inc
                error -= dy

            else:
                y += y_inc
                error += dx
            n -= 1

        squares = set(squares)
        is_valid = not bool(self.obstacles.intersection(squares))
        self.valid_edge_cache[configs] = is_valid
        return self.valid_edge_cache[configs]


    def compute_heuristic(self, config):
        
        #
        # TODO: Implement a function to compute heuristic.
        #
        return self.compute_distance(config, self.goal)


    def _get_obstacles(self):
        result = np.where(self.map == 1)
        obstacles = set(list(zip(result[0], result[1])))
        return obstacles


    def draw_graph(self, rnd_node=None, tree=None, plan=None, filename=None, title=None, show=False, close=False):
        '''
        Visualize the final path
        @param plan Sequence of states defining the plan.
        '''
        plt.clf()
        plt.imshow(self.map, interpolation='nearest')
        if tree is not None:
            for node_idx, node in enumerate(tree.vertices):
                if node_idx in tree.edges:
                    dx = [node.x, tree.vertices[tree.edges[node_idx]].x ]
                    dy = [node.y, tree.vertices[tree.edges[node_idx]].y ]
                    plt.plot(dy, dx, 'b')

        if rnd_node is not None:
            plt.plot(rnd_node.y, rnd_node.x, "or")

        if plan is not None:
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
        
        if show:
            plt.show()
        else:
            plt.pause(0.01)
            plt.draw()

        if close:
            plt.close()

    def calc_cost(self, plan):
        return sum([self.compute_distance(cur_p, next_p) for (cur_p, next_p) in zip(plan, plan[1:])])