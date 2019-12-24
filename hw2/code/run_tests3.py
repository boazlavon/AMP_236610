#!/usr/bin/env python

import argparse, numpy, time

from MapEnvironment import MapEnvironment
from RRTPlanner import RRTPlanner
from RRTStarPlanner import RRTStarPlanner
from AStarPlanner import AStarPlanner

from IPython import embed

def main(planning_env, planner, start, goal, planner_name, extra_text='',):

    # Notify.
    # Plan.
    start_time = time.time()
    plan = planner.Plan(start, goal, max_iter=2000)

    # Shortcut the path.
    # TODO (student): Do not shortcut when comparing the performance of algorithms. 
    # Comment this line out when collecting data over performance metrics.
    plan_short = planner.ShortenPath(plan)
    end_time   = time.time()

    # Visualize the final path.
    plan_short = [(node.x, node.y) for _, node in plan_short]
    cost  = planning_env.calc_plan_cost(plan_short)
    exec_time = time.strftime('%M:%S', time.gmtime(end_time - start_time))
    title =  'planner: {}, cost: {}, exec_time: {}'.format(planner_name, cost, exec_time)
    filename = '{}_{}_{}_{}_{}'.format(extra_text, planner_name, int(cost), int(end_time - start_time), int(time.time()))
    planning_env.visualize_plan(plan_short, filename=filename, title=title)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='script for testing planners')

    parser.add_argument('-m', '--map', type=str, default='map1.txt',
                        help='The environment to plan on')    
    parser.add_argument('-p', '--planner', type=str, default='rrt',
                        help='The planner to run (star, rrt, rrtstar)')
    parser.add_argument('-s', '--start', nargs='+', type=int, required=True)
    parser.add_argument('-g', '--goal', nargs='+', type=int, required=True)

    args = parser.parse_args()

    for j, k in enumerate([2,4]):
        for i in range(2):
            planning_env = MapEnvironment(args.map, args.start, args.goal, map_resolution=1)
            planner = RRTStarPlanner(planning_env, goal_sample_rate=0.2, k=k)
            main(planning_env, planner, args.start, args.goal, args.planner, extra_text='exp3_j{}_k{}'.format(j, k))