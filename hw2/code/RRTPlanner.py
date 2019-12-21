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
            self.cost = 0
            self.parent = None

        def __repr__(self):
            return str((self.x, self.y))


class RRTPlanner(object):

    def __init__(self, planning_env, goal_sample_rate=0.2):
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)
        self.goal_sample_rate = goal_sample_rate


    def Plan(self, start_config, goal_config, sample_dist = 10, k = None, max_iter = 1000):
        
        plan = []
        start_node = Node(*start_config)
        goal_node  = Node(*goal_config)

        # Start with adding the start configuration to the tree.
        start_idx = self.tree.AddVertex(start_node)
        i = 0
        while i < max_iter:
            rnd_node = self.get_random_node()
            nearest_node_idx, nearest_node = self.tree.GetNearestVertex(rnd_node)
            new_node = self.extend(nearest_node, rnd_node, sample_dist)
            if not self.planning_env.state_validity_checker((new_node.x, new_node.y), check_obst=False):
                continue

            if self.planning_env.edge_validity_checker(nearest_node, new_node):
                new_node_idx = self.tree.AddVertex(new_node)
                self.tree.AddEdge(nearest_node_idx, new_node_idx)
                new_node.parent = nearest_node

                if k is not None:
                    self.rewire(new_node, new_node_idx, k)

            print(i)
            if (i % 3) == 0:
                self.draw_graph(new_node)
            i += 1

            last_node = self.tree.vertices[-1]
            last_node_idx = len(self.tree.vertices) - 1
            goal_dist, _ = self.calc_distance_and_angle(last_node, goal_node)

            if goal_dist <= sample_dist:
                if self.planning_env.edge_validity_checker(last_node, goal_node):
                    goal_idx = self.tree.AddVertex(goal_node)
                    self.tree.AddEdge(last_node_idx, goal_idx)
                    goal_node.cost = last_node.cost + goal_dist
                    plan = self.extract_plan(start_idx, goal_idx)
                    self.draw_graph()
                    break

        # TODO (student): Implement  your planner here.
        return plan


    def rewire(self, x_child, x_child_idx, k):
        
        k = min(k, len(self.tree.vertices) - 1)
        knnIDs, k_nodes = self.tree.GetKNN(x_child, k)
        for pparent_id, x_pparent in zip(knnIDs, k_nodes):
            if pparent_id == x_child_idx:
                continue

            if self.planning_env.edge_validity_checker(x_pparent, x_child):
                R, _ = self.calc_distance_and_angle(x_pparent, x_child)
                if (x_pparent.cost + R < x_child.cost): # rewire
                    self.tree.AddEdge(pparent_id, x_child_idx)
                    x_child.cost = x_pparent.cost + R
                    x_child.parent = x_pparent
                    self.propagate_cost_to_leaves(x_child)

        for pparent_id, x_pparent in zip(knnIDs, k_nodes):
            if pparent_id == x_child_idx:
                continue

            if self.planning_env.edge_validity_checker(x_child, x_pparent):
                R, _ = self.calc_distance_and_angle(x_child, x_pparent)
                if (x_child.cost + R < x_pparent.cost): # rewire
                    self.tree.AddEdge(x_child_idx, pparent_id)
                    x_pparent.cost = x_child.cost + R
                    x_pparent.parent = x_child
                    self.propagate_cost_to_leaves(x_pparent)


    def extract_plan(self, start_idx, goal_idx):
        plan = []
        iter_node = self.tree.vertices[goal_idx]
        iter_idx = goal_idx
        plan.append((iter_idx, iter_node))

        while True:
            iter_idx = self.tree.edges[iter_idx]
            iter_node = self.tree.vertices[iter_idx]
            plan.append((iter_idx, iter_node))
            if start_idx == iter_idx:
                break
        plan = plan[::-1]
        return plan


    def extend(self, from_node, to_node, sample_dist):
        R, theta = self.calc_distance_and_angle(from_node, to_node)
        if sample_dist < R:
            R = sample_dist
            new_node = Node(from_node.x + R * np.cos(theta), from_node.y + R * np.sin(theta))
        else:
            new_node = to_node
        
        new_node.cost = R + from_node.cost
        return new_node


    def get_random_node(self):
        if np.random.rand() > self.goal_sample_rate:
            rnd = Node(np.random.random_integers(*self.planning_env.xlimit),
                       np.random.random_integers(*self.planning_env.ylimit))
        else:  # goal point sampling
            rnd = Node(*self.planning_env.goal)
        return rnd

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.tree.vertices:
            if node.parent == parent_node:
                R, _ = self.calc_distance_and_angle(node, parent_node)
                node.cost = parent_node.cost + R
                self.propagate_cost_to_leaves(node)


    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d  = ((dx**2 + dy**2)**0.5)
        theta = math.atan2(dy, dx)
        return d, theta


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


    def ShortenPath(self, path):
        # TODO (student): Postprocessing of the plan.
        start = 0
        while start < len(path):
            start_idx, start_node = path[start]
            end = len(path) - 1
            while start < end:
                end_idx, end_node = path[end]
                if not self.planning_env.edge_validity_checker(start_node, end_node):
                    end -= 1
                    continue

                R, _ = self.calc_distance_and_angle(start_node, end_node)
                if (start_node.cost + R < end_node.cost): # rewire
                    self.tree.AddEdge(start_idx, end_idx)
                    end_node.cost   = start_node.cost + R
                    end_node.parent = start_node
                    self.propagate_cost_to_leaves(end_node)
                    start = end - 1
                    break
                else:
                    end -= 1
            start += 1

        start_idx, _ = path[0]
        goal_idx,  _ = path[-1]
        plan = self.extract_plan(start_idx, goal_idx)
        return plan