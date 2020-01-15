import sys
import time
import numpy
import MapEnvironment
from matplotlib import pyplot as plt

class Node():
    def __init__(self, parent, position):
        self.parent = parent
        self.position = position
        self.g=0
        self.h=0
        self.f=0

    def __eq__(self, other):
        eq = set(self.position == other.position)
        return eq == {True}

class AStarPlanner(object):
    def __init__(self, planning_env: MapEnvironment.MapEnvironment):
        self.planning_env = planning_env
        self.nodes = dict()
        self.eps = 1

    def Plan(self, start_config, goal_config):
        plan = []
        start = Node(None, numpy.array(start_config))
        goal = Node(None, numpy.array(goal_config))
        open_list = [start]
        closed_list = []
        while len(open_list) > 0:
            current_node = min(open_list, key=lambda x: x.f)
            open_list.remove(current_node)
            closed_list.append(current_node)
            plt.plot(current_node.position[1],current_node.position[0], 'o', color='b')
            if current_node == goal:
                print("Total cost: ", current_node.g)
                print("Num of nodes expanded: ", len(closed_list))
                while current_node is not None:
                    plan.append(current_node.position)
                    current_node = current_node.parent
                return numpy.array(plan[::-1])

            moves = [(0,1),(1,0),(1,1),(0,-1),(-1,0),(-1,-1),(-1,1),(1,-1)]
            successors = [Node(current_node, current_node.position + move)
                           for move in moves if self.planning_env.state_validity_checker(current_node.position + move)]
                           
            for successor in successors:
                new_g = current_node.g + self.planning_env.compute_distance(current_node.position, successor.position)
                try:
                    old_node = open_list[open_list.index(successor)]
                    if new_g < old_node.g:
                        old_node.g = new_g
                        old_node.parent = current_node
                        old_node.f = old_node.g + self.eps * old_node.h
                except ValueError: #Not in open list
                    try:
                        old_node = closed_list[closed_list.index(successor)]
                        if new_g < old_node.g:
                            closed_list.remove(old_node)
                            old_node.g = new_g
                            old_node.parent = current_node
                            old_node.f = old_node.g + self.eps * old_node.h
                            open_list.append(old_node)
                    except ValueError: #Not in closed list
                        successor.g = new_g
                        successor.h = self.planning_env.compute_heuristic(successor.position)
                        successor.f = successor.g + self.eps * successor.h
                        open_list.append(successor)

        return numpy.array(plan)

    def ShortenPath(self, path):

        # TODO (student): Postprocess the planner.
        return path
