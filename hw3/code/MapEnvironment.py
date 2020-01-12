import numpy as np
from IPython import embed
from matplotlib import pyplot as plt

class MapEnvironment(object):
    
    def __init__(self, mapfile, start, goal):

        # Obtain the boundary limits.
        # Check if file exists.
        self.map = np.loadtxt(mapfile)
        self.xlimit = [0, np.shape(self.map)[0]-1] # TODO (avk): Check if this needs to flip.
        self.ylimit = [0, np.shape(self.map)[1]-1]

        # Check if start and goal are within limits and collision free
        if not self.state_validity_checker(start) or not self.state_validity_checker(goal):
            raise ValueError('Start and Goal state must be within the map limits');
            exit(0)
        # store start and goal
        self.start = start
        self.goal = goal
        # Display the map
        #plt.imshow(self.map, interpolation='nearest')

    def compute_distance(self, start_config, end_config):
        s, g = np.array(start_config), np.array(end_config)
        return np.linalg.norm(g-s)

    def state_validity_checker(self, config):
        # check within limit
        if config[0] < self.xlimit[0] or config[0] > self.xlimit[1] or\
         config[1] < self.ylimit[0] or config[1] > self.ylimit[1]:
            return False
        # check collision
        x, y = int(config[0]), int(config[1])
        return not self.map[x][y]

    def to_map(self, config):
        ''' helper function mapping continuous config to discredited map'''
        return config.astype(int)

    def edge_validity_checker(self, config1, config2, step_size=0.3):
        if not self.state_validity_checker(config1) or not self.state_validity_checker(config2):
            return False
        c1, c2 = np.array(config1).astype(float)+.5, np.array(config2).astype(float)+.5
        v_norm = np.linalg.norm(c2-c1)
        count = 1
        while (count * step_size) < v_norm:
            a = count * step_size / v_norm
            check_config = (1-a)*c1 + a*c2
            if not self.state_validity_checker(check_config):
                return False
            count += 1
        return True

    def compute_heuristic(self, config):
        return self.compute_distance(config, self.goal)

    def build_user_guided_heuristic(self, g_config):
        def user_guided_heuristic(config, is_ancestor):
            if g_config in is_ancestor and is_ancestor[g_config]:
                return self.compute_heuristic(config)
            else:
                return (self.compute_distance(config, g_config) +\
                        self.compute_heuristic(g_config))

        return user_guided_heuristic

    def successors(self, config):
        x, y = config
        successors = [(x + i, y + j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
        successors.remove(config)
        successors = [config for config in successors\
                         if self.state_validity_checker(config)]
        return successors

    def visualize_plan(self, plan):
        '''
        Visualize the final path
        @param plan Sequence of states defining the plan.
        '''
        plt.imshow(self.map, interpolation='nearest')
        plt.plot(self.start[1], self.start[0], 'o', color='r')
        plt.plot(self.goal[1], self.goal[0], 'o', color='g')
        for i in range(np.shape(plan)[0] - 1):
            x = [plan[i,0], plan[i+1, 0]]
            y = [plan[i,1], plan[i+1, 1]]
            plt.plot(y, x, 'k')
        plt.show()

    def draw_graph(self, bp=None, plan=None, filename=None, title=None, show=False, close=False):
        '''
        Visualize the final path
        @param plan Sequence of states defining the plan.
        '''
        plt.clf()
        plt.imshow(self.map, interpolation='nearest')
        if bp is not None:
            for config in bp:
                if bp[config] is not None:
                    dx = [config[0], bp[config][0] ]
                    dy = [config[1], bp[config][1] ]
                    plt.plot(dy, dx, 'b')

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


    def visualize_env(self):
        '''
        Visualize the environment
        '''
        plt.imshow(self.map, interpolation='nearest')
        plt.plot(self.start[1], self.start[0], 'o', color='r')
        plt.plot(self.goal[1], self.goal[0], 'o', color='g')
        plt.show()