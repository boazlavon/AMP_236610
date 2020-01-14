#!/usr/bin/env python

import argparse

from AStarPlanner import AStarPlanner
from IPython import embed
from MapEnvironment import MapEnvironment
from MultiHeuristicPlanner import MultiHeuristicPlanner


def main(planning_env, planner, start, goal, w1, w2, mp):
    # Notify.
    # Plan.
    plan = planner.Plan(start, goal)
    nodes_count = len(planner.g)
    cost        = planner.g[tuple(goal)]
    if cost == float('inf'):
        cost = 'inf'
    planner_name = 'MHA_test1_{}'.format(mp)
    title =  'w1 = {}, w2 = {}, cost: {}, nodes_count: {}'.format(w1, w2, cost, nodes_count)
    filename = '{}_{}_{}_w1_{}_w2_{}'.format(planner_name, cost, nodes_count, w1, w2)

    # Visualize the final path.
    planning_env.draw_graph(bp=planner.bp, plan=plan, filename=filename, title=title, show=False, close=True)
	

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='script for testing planners')

    parser.add_argument('-m', '--map', type=str, default='map1.txt',
                        help='The environment to plan on')
    parser.add_argument('-p', '--planner', type=str, default='rrt',
                        help='The planner to run (star, rrt, rrtstar)')
    parser.add_argument('-s', '--start', nargs='+', type=int, required=True)
    parser.add_argument('-g', '--goal', nargs='+', type=int, required=True)

    # added
    parser.add_argument('-u', '--userGuidance', nargs='+', type=float,
                        help='User guidance points for the multiHeuristic AStar')

    args = parser.parse_args()

    # First setup the environment and the robot.
    
    # Next setup the planner
    W1 = (1,)
    W2 = (1,)
    for w1 in W1:
        for w2 in W2:
            print('w1={} W2={}'.format(w2, w2))
            planning_env = MapEnvironment(args.map, args.start, args.goal)
            planner = MultiHeuristicPlanner(planning_env, list(zip(args.userGuidance[::2], args.userGuidance[1::2])), w1, w2)
            main(planning_env, planner, args.start, args.goal, w1, w2, args.map)