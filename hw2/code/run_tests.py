#!/usr/bin/env python

import argparse, numpy, time

from MapEnvironment import MapEnvironment
from RRTPlanner import RRTPlanner
from RRTStarPlanner import RRTStarPlanner
from AStarPlanner import AStarPlanner

from IPython import embed

def main(planning_env, planner, start, goal, planner_name, sample_dist, extra_text='',):

    # Notify.
    input('Press any key to begin planning')

    # Plan.
    start_time = time.time()
    plan = planner.Plan(start, goal, sample_dist=sample_dist)

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
    for prob in [0.05, 0.2]:
        for i in range(5):
            planner = RRTPlanner(planning_env, goal_sample_rate=p)
            main(planning_env, planner, args.start, args.goal, args.planner, sample_dist=10, extra_text='exp1_p{}_i{}'.format(p, i))

    for j, dist in enumerate([4, 16, 64, 128, float('inf')]):
        for i in range(3):
            planner = RRTPlanner(planning_env, goal_sample_rate=0.2)
            main(planning_env, planner, args.start, args.goal, args.planner, sample_dist=dist, extra_text='exp2_j{}_i{}'.format(j, i))