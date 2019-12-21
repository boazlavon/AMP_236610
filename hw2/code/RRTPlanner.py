import numpy as np
import math
from RRTTree import RRTTree
from matplotlib import pyplot as plt


class Node(object):
        """
        RRT Node
        """
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path = []
            self.parent = None
            self.d = 0

        def __repr__(self):
            return str((self.x, self.y))


class RRTPlanner(object):

    def __init__(self, planning_env, path_resolution=1, goal_sample_rate=0.2, k=None):
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        

    def Plan(self, start_config, goal_config, step_size = 10, k = None):
        
        plan = []
        start_node = Node(*start_config)
        goal_node  = Node(*goal_config)

        # Start with adding the start configuration to the tree.
        start_idx = self.tree.AddVertex(start_node)
        i = 0
        while True:
            rnd_node = self.get_random_node()
            nearest_node_idx, nearest_node = self.tree.GetNearestVertex(rnd_node)
            new_node = self.extend(nearest_node, rnd_node, step_size)
            if self.check_collision(new_node):
                new_node_idx = self.tree.AddVertex(new_node)
                self.tree.AddEdge(nearest_node_idx, new_node_idx)

                if k is not None:
                    self.rewire(new_node, new_node_idx, k, step_size)

            i += 1
            print(i)
            if (i % 3) == 1:
                self.draw_graph(new_node)
            
            last_node = self.tree.vertices[-1]
            goal_dist, _ = self.calc_distance_and_angle(last_node, goal_node)
            if goal_dist <= step_size:
                final_node = self.extend(last_node, goal_node, step_size)
                if self.check_collision(final_node):
                    goal_idx = self.tree.AddVertex(goal_node)
                    self.tree.AddEdge(new_node_idx, goal_idx)
                    plan = self.extract_plan(start_idx, goal_idx)
                    break

        # TODO (student): Implement your planner here.
        return np.array(plan)


    def rewire(self, x_child, x_child_idx, k, step_size):
        
        k = min(k, len(self.tree.vertices) - 1)
        knnIDs, k_nodes = self.tree.GetKNN(x_child, k)
        for pparent_id, x_pparent in zip(knnIDs, k_nodes):
            if pparent_id == x_child_idx:
                continue

            new_child_node = self.extend(x_pparent, x_child, step_size)
            if self.check_collision(new_child_node):
                if new_child_node.path[-1] != (x_child.x, x_child.y):
                    continue
                if (new_child_node.d < x_child.d): # rewire
                    self.tree.AddEdge(pparent_id, x_child_idx)                    


    def extract_plan(self, start_idx, goal_idx):
        plan = []
        plan.append(self.tree.vertices[goal_idx])
        iter_idx = goal_idx
        while True:
            iter_idx = self.tree.edges[iter_idx]
            iter_node = self.tree.vertices[iter_idx]
            plan.append((iter_node.x, iter_node.y))
            if start_idx == iter_idx:
                break
        plan = plan[::-1]
        return plan

    def check_collision(self, node):
        for (ox, oy) in self.planning_env.obstacles:

            dxdy_list = [(abs(ox - x), abs(oy - y)) for x, y in node.path]
            for dx, dy in dxdy_list:
                if dx < 0.5 and dy < 0.5: # an obstacle is represented by a square
                    return False # collision
            
        return True  # safe

    def extend(self, from_node, to_node, step_size):
        assert (step_size > self.path_resolution)

        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if step_size > d:
            step_size = d

        n_expand = int(np.floor(step_size / self.path_resolution))

        for _ in range(n_expand):
            new_node.x += self.path_resolution * np.cos(theta)
            new_node.y += self.path_resolution * np.sin(theta)
            new_node.path.append((new_node.x, new_node.y))

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path.append((to_node.x, to_node.y))

        d, _ = self.calc_distance_and_angle(from_node, to_node)
        new_node.d = from_node.d + d
        return new_node

    def get_random_node(self):
        if np.random.rand() > self.goal_sample_rate:
            rnd = Node(np.random.random_integers(*self.planning_env.xlimit),
                       np.random.random_integers(*self.planning_env.ylimit))
        else:  # goal point sampling
            rnd = Node(*self.planning_env.goal)
        return rnd

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d  = ((dx**2 + dy**2)**0.5)
        theta = math.atan2(dy, dx)
        return d, theta

    def ShortenPath(self, path):
        # TODO (student): Postprocessing of the plan.
        return path

    def draw_graph(self, rnd=None):
        plt.clf()
        plt.imshow(self.planning_env.map, interpolation='nearest')
        if rnd is not None:
            plt.plot(rnd.y, rnd.x, "oc")
        
        for node_idx, node in enumerate(self.tree.vertices):
            if node_idx in self.tree.edges:
                dx = [node.x, self.tree.vertices[self.tree.edges[node_idx]].x ]
                dy = [node.y, self.tree.vertices[self.tree.edges[node_idx]].y ]
                plt.plot(dy, dx, 'b')

        plt.pause(0.01)
        plt.draw()