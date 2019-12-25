import numpy as np
import time
from RRTTree import RRTTree
from RRTPlanner import RRTPlanner, Node

class RRTStarPlanner(RRTPlanner):

    def __init__(self, planning_env, goal_bias=0.2, step_size = 10, k = None):
        super().__init__(planning_env, goal_bias, step_size)
        self.k = k     

    def Plan(self, start_config, goal_config, max_iter = 3000, use_cost_times=False):
        
        plan = []
        start_node = Node(*start_config)
        goal_node  = Node(*goal_config)
        goal_idx = None
        min_cost = float('inf')
        
        plot_plan = None
        title = 'cost = inf'
        cost_times = []

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
                k = self.k
                if k is None:
                    k = int(np.log(len(self.tree.vertices)))

                if len(self.tree.vertices) <= k:
                    knnIDs, k_nodes = list(range(len(self.tree.vertices))), self.tree.vertices
                else:
                    knnIDs, k_nodes = self.tree.GetKNN(new_node, k)

                parent_idx = self.choose_parent(new_node, knnIDs)
                if parent_idx is not None:
                    new_node_idx = self.tree.AddVertex(new_node)
                    self.tree.AddEdge(parent_idx, new_node_idx)
                    if new_node.p == goal_node.p:
                        goal_idx = new_node_idx
                    self.rewire(new_node_idx, knnIDs)

            print(i)
            if (i % 3) == 0:
                self.planning_env.draw_graph(rnd_node=new_node, tree=self.tree, plan=plot_plan, title=title)
                    
            if goal_idx is not None:
                cost = self.tree.vertices[goal_idx].cost
                if cost < min_cost:
                    min_cost = cost
                    plan = self.extract_plan(start_idx, goal_idx)
                    cost_times.append((time.time(), min_cost))
                    title = 'cost = {}'.format(min_cost)
                    plot_plan = [self.tree.vertices[node_idx].p for node_idx in plan]
            i += 1

        if goal_idx is not None:
            plan = self.extract_plan(start_idx, goal_idx)
            cost_times.append((time.time(), self.tree.vertices[goal_idx].cost))

        if use_cost_times:
            return (plan, cost_times)

        return plan

    def choose_parent(self, new_node, knnIDs):
        if knnIDs == []:
            return None

        # search nearest cost in knnIDs
        extended = []
        for near_node_idx in knnIDs:
            near_node = self.tree.vertices[near_node_idx]
            if self.planning_env.edge_validity_checker(near_node.p, new_node.p):
                extended.append((self.planning_env.compute_distance(near_node.p, new_node.p), 
                                 near_node_idx))

        if not extended:
            return None

        new_cost, parent_idx = min(extended, key= lambda node_t : node_t[0])
        new_node.cost = self.tree.vertices[parent_idx].cost + new_cost
        return parent_idx

    def rewire(self, new_node_idx, knnIDs):
        new_node = self.tree.vertices[new_node_idx]
        for near_node_idx in knnIDs:
            near_node = self.tree.vertices[near_node_idx]
            no_collision = self.planning_env.edge_validity_checker(new_node.p, near_node.p)
            R = self.planning_env.compute_distance(new_node.p, near_node.p)
            improved_cost = near_node.cost > new_node.cost + R

            if no_collision and improved_cost:
                self.tree.AddEdge(new_node_idx, near_node_idx)
                self.propagate_cost_to_leaves(new_node_idx)