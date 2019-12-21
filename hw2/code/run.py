#!/usr/bin/env python

import argparse, numpy, time

from MapEnvironment import MapEnvironment
from RRTPlanner import RRTPlanner
from RRTStarPlanner import RRTStarPlanner
from AStarPlanner import AStarPlanner

from IPython import embed

def main(planning_env, planner, start, goal, planner_name):

    # Notify.
    input('Press any key to begin planning')

    # Plan.
    start_time = time.time()
    plan = planner.Plan(start, goal)
    end_time   = time.time()

    # Shortcut the path.
    # TODO (student): Do not shortcut when comparing the performance of algorithms. 
    # Comment this line out when collecting data over performance metrics.
    plan_short = planner.ShortenPath(plan)

    # Visualize the final path.
    cost  = planning_env.calc_plan_cost(plan)
    exec_time = time.strftime('%M:%S', time.gmtime(end_time - start_time))
    title =  'planner: {}, cost: {}, exec_time: {}'.format(planner_name, cost, exec_time)
    filename = '{}_{}_{}_{}'.format(planner_name, int(cost), int(end_time - start_time), int(time.time()))
    planning_env.visualize_plan(plan, filename=filename, title=title)
    embed()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='script for testing planners')

    parser.add_argument('-m', '--map', type=str, default='map1.txt',
                        help='The environment to plan on')    
    parser.add_argument('-p', '--planner', type=str, default='rrt',
                        help='The planner to run (star, rrt, rrtstar)')
    parser.add_argument('-s', '--start', nargs='+', type=int, required=True)
    parser.add_argument('-g', '--goal', nargs='+', type=int, required=True)

    args = parser.parse_args()

    # First setup the environment and the robot.
    planning_env = MapEnvironment(args.map, args.start, args.goal)

    # Next setup the planner
    if args.planner == 'astar':
        planner = AStarPlanner(planning_env)
    elif args.planner == 'rrt':
        planner = RRTPlanner(planning_env)
    elif args.planner == 'rrtconnect':
        planner = RRTStarPlanner(planning_env)
    else:
        print('Unknown planner option: %s' % args.planner)
        exit(0)

    main(planning_env, planner, args.start, args.goal, args.planner)
