#!/usr/bin/env python

import argparse

from AStarPlanner import AStarPlanner
from IPython import embed
from MapEnvironment import MapEnvironment
from MultiHeuristicPlannerSolution import MultiHeuristicPlanner


def main(planning_env, planner, start, goal):
    # Notify.
    input('Press any key to begin planning')
    planning_env.visualize_env()

    # Plan.
    plan = planner.Plan(start, goal)

    # Visualize the final path.
    planning_env.visualize_plan(plan)
    exit(0)
	

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
    planning_env = MapEnvironment(args.map, args.start, args.goal)

    # Next setup the planner
    if args.planner == 'astar':
        planner = AStarPlanner(planning_env)
    elif args.planner == 'mhastar':
        planner = MultiHeuristicPlanner(planning_env, list(zip(args.userGuidance[::2], args.userGuidance[1::2])))
    else:
        print('Unknown planner option: %s' % args.planner)
        exit(0)

    main(planning_env, planner, args.start, args.goal)
