import numpy as np
import math
from RRTTree import RRTTree
from matplotlib import pyplot as plt


class Node(object):
        """
        RRT Node
        """
        def __init__(self, x, y, cost=0):
            self.x = x
            self.y = y
            self.cost = cost

        @property
        def p(self):
            return (self.x, self.y)

        def __repr__(self):
            return str((self.x, self.y))


class RRTPlanner(object):

    def __init__(self, planning_env, goal_bias=0.2, step_size = 10):
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)
        self.goal_bias = goal_bias
        self.step_size = step_size


    def Plan(self, start_config, goal_config, max_iter = 1000):
        
        plan = []
        start_node = Node(*start_config)
        goal_node  = Node(*goal_config)

        # Start with adding the start configuration to the tree.
        start_idx = self.tree.AddVertex(start_node)
        i = 0
        while i < max_iter:
            rnd_node = self.get_random_node()
            nearest_node_idx, nearest_node = self.tree.GetNearestVertex(rnd_node)
            new_node = self.extend(nearest_node, rnd_node)
            if not self.planning_env.state_validity_checker(new_node.p, check_obst=False):
                continue

            if self.planning_env.edge_validity_checker(nearest_node.p, new_node.p):
                new_node_idx = self.tree.AddVertex(new_node)
                self.tree.AddEdge(nearest_node_idx, new_node_idx)
                if (new_node.p == goal_node.p):
                    plan = self.extract_plan(start_idx, new_node_idx)
                    plan_ = [self.tree.vertices[node_idx].p for node_idx in plan]
                    self.planning_env.draw_graph(tree=self.tree, plan=plan_)
                    break

            print(i)
            if (i % 3) == 0:
                self.planning_env.draw_graph(rnd_node=new_node, tree=self.tree)
            i += 1

        # TODO (student): Implement  your planner here.
        return plan

    def extract_plan(self, start_idx, goal_idx):
        plan = []
        iter_idx = goal_idx
        while True:
            plan.append(iter_idx)
            if start_idx == iter_idx:
                break

            # next
            iter_idx = self.tree.edges[iter_idx]

        plan = plan[::-1]
        return plan


    def extend(self, from_node, to_node):        
        from_node_cost = from_node.cost
        from_node = np.array((from_node.x, from_node.y))
        to_node   = np.array((to_node.x, to_node.y))

        delta = to_node - from_node
        norm  = self.planning_env.compute_distance(from_node, to_node)
        if norm > self.step_size:
            delta = delta / norm
            delta = delta * self.step_size
        
        new_node = from_node + delta
        new_node = Node(*new_node)
        new_node.cost = min(norm, self.step_size) + from_node_cost
        return new_node

    def get_random_node(self):
        if np.random.rand() > self.goal_bias:
            rnd = Node(np.random.random_integers(*self.planning_env.xlimit),
                       np.random.random_integers(*self.planning_env.ylimit))
        else:  # goal point sampling
            rnd = Node(*self.planning_env.goal)
        return rnd

    def propagate_cost_to_leaves(self, parent_node_idx):
        parent_node = self.tree.vertices[parent_node_idx]
        for node_idx, node in enumerate(self.tree.vertices):
            if node_idx in self.tree.edges:
                if self.tree.edges[node_idx] == parent_node_idx:
                    R = self.planning_env.compute_distance(parent_node.p, node.p)
                    node.cost = parent_node.cost + R
                    self.propagate_cost_to_leaves(node_idx)

    def ShortenPath(self, path):
        # TODO (student): Postprocessing of the plan.
        if path == []:
            return path
        
        start = 0
        while start < len(path):
            start_idx = path[start]
            start_node = self.tree.vertices[start_idx]

            end = len(path) - 1
            while start < end:
                end_idx = path[end]
                end_node = self.tree.vertices[end_idx]
                if not self.planning_env.edge_validity_checker(start_node.p, end_node.p):
                    end -= 1
                    continue

                R = self.planning_env.compute_distance(start_node.p, end_node.p)
                if (start_node.cost + R < end_node.cost): # rewire
                    self.tree.AddEdge(start_idx, end_idx)
                    end_node.cost   = start_node.cost + R
                    self.propagate_cost_to_leaves(end_idx)
                    start = end - 1
                    break
                else:
                    end -= 1
            start += 1

        start_idx = path[0]
        goal_idx  = path[-1]
        plan = self.extract_plan(start_idx, goal_idx)
        return plan