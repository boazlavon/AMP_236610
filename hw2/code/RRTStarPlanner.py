import numpy
from RRTTree import RRTTree
from RRTPlanner import RRTPlanner, Node

class RRTStarPlanner(RRTPlanner):

    def __init__(self, planning_env, goal_sample_rate=0.2, sample_dist = 30, k = 2, finish_on_goal=True):
        super().__init__(planning_env, goal_sample_rate, sample_dist)
        self.k = k     
        self.finish_on_goal = finish_on_goal

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
            if not self.planning_env.state_validity_checker((new_node.x, new_node.y), check_obst=False):
                continue

            if self.planning_env.edge_validity_checker(nearest_node, new_node):
                if len(self.tree.vertices) <= self.k:
                    knnIDs, k_nodes = list(range(len(self.tree.vertices))), self.tree.vertices
                else:
                    knnIDs, k_nodes = self.tree.GetKNN(new_node, self.k)

                result = self.choose_parent(new_node, knnIDs)
                if result is not None:
                    new_node, parent_idx = result
                    parent_node  = self.tree.vertices[parent_idx]
                    if self.planning_env.edge_validity_checker(new_node, parent_node):
                        new_node_idx = self.tree.AddVertex(new_node)
                        self.tree.AddEdge(parent_idx, new_node_idx)
                        new_node.parent = parent_node
                        self.rewire(new_node_idx, knnIDs)

            print(i)
            if (i % 2) == 0:
                self.draw_graph(new_node)
                i = i
            i += 1

            if new_node and self.finish_on_goal:  # check reaching the goal
                result = self.search_best_goal_node(goal_node)
                if result is not None:
                    last_node_idx, goal_dist = result
                    last_node = self.tree.vertices[last_node_idx]
                    if self.planning_env.edge_validity_checker(last_node, goal_node):
                        goal_idx = self.tree.AddVertex(goal_node)
                        self.tree.AddEdge(last_node_idx, goal_idx)
                        goal_node.cost = self.tree.vertices[last_node_idx].cost + goal_dist
                        plan = self.extract_plan(start_idx, goal_idx)
                        self.draw_graph()
                        break

        if not self.finish_on_goal:
            result = self.search_best_goal_node(goal_node)
            if result is not None:
                last_node_idx, goal_dist = result
                last_node = self.tree.vertices[last_node_idx]
                if self.planning_env.edge_validity_checker(last_node, goal_node):
                    goal_idx = self.tree.AddVertex(goal_node)
                    self.tree.AddEdge(last_node_idx, goal_idx)
                    goal_node.cost = self.tree.vertices[last_node_idx].cost + goal_dist
                    plan = self.extract_plan(start_idx, goal_idx)
                    self.draw_graph()

        return plan

    def choose_parent(self, new_node, knnIDs):
        if knnIDs == []:
            return None

        # search nearest cost in knnIDs
        extended = []
        for near_node_idx in knnIDs:
            near_node = self.tree.vertices[near_node_idx]
            if self.planning_env.edge_validity_checker(near_node, new_node):
                t_node = self.extend(near_node, new_node)
                if ((t_node.x == new_node.x) and (t_node.y == new_node.y)):
                    extended.append((t_node, near_node_idx))

        if not extended:
            return None

        min_node, parent_idx = min(extended, key= lambda node_t : node_t[0].cost)
        return (min_node, parent_idx)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def rewire(self, new_node_idx, knnIDs):
        new_node = self.tree.vertices[new_node_idx]
        for near_node_idx in knnIDs:
            near_node = self.tree.vertices[near_node_idx]
            edge_node = self.extend(new_node, near_node)
            if not edge_node:
                continue

            if not ((edge_node.x == near_node.x) and (edge_node.x == near_node.x)):
                continue

            no_collision = self.planning_env.edge_validity_checker(new_node, near_node)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node.cost = edge_node.cost
                near_node.parent = new_node
                self.tree.AddEdge(new_node_idx, near_node_idx)
                self.propagate_cost_to_leaves(new_node)


    def search_best_goal_node(self, goal_node):
        dist_to_goal_list = [self.calc_distance_and_angle(n, goal_node)[0] for n in self.tree.vertices]
        goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.sample_dist]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            if self.planning_env.edge_validity_checker(self.tree.vertices[goal_ind], goal_node):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.tree.vertices[i].cost for i in safe_goal_inds])
        for node_idx in safe_goal_inds:
            if self.tree.vertices[node_idx].cost == min_cost:
                return (node_idx, min_cost)
        return None