#!/usr/bin/env python

import argparse, numpy, time

from MapEnvironment import MapEnvironment
from RRTPlanner import RRTPlanner
from RRTStarPlanner import RRTStarPlanner
from AStarPlanner import AStarPlanner

from IPython import embed

def main(planning_env, planner, start, goal, planner_name='rrtstar', extra_text=''):

    # Plan.
    start_time = time.time()
    plan, cost_times = planner.Plan(start, goal, use_cost_times=True, max_iter=3000)

    # Shortcut the path.
    # TODO (student): Do not shortcut when comparing the performance of algorithms. 
    # Comment this line out when collecting data over performance metrics.
    plan = planner.ShortenPath(plan)
    cost_times = [(start_time, float('inf'))] + cost_times
    end_time = time.time()

    # Visualize the final path.
    cost = 'inf'
    if planner_name == 'rrt' or planner_name == 'rrtstar':
        if len(plan):
            cost = planner.tree.vertices[plan[-1]].cost
        plan = [planner.tree.vertices[node_idx].p for node_idx in plan]
        tree = planner.tree
    else:
        if len(plan):
            cost = planning_env.calc_cost(plan)
        tree = None

    exec_time = time.strftime('%M:%S', time.gmtime(end_time - start_time))
    title =  'planner: {}, cost: {}, exec_time: {}'.format(planner_name, cost, exec_time)
    filename = '{}{}_{}_{}_{}'.format(extra_text, planner_name, cost, int(end_time - start_time), int(time.time()))
    
    planning_env.draw_graph(tree=tree, plan=plan, filename=filename, title=title, show=False, close=True)
    return cost_times

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='script for testing planners')

    parser.add_argument('-m', '--map', type=str, default='map1.txt',
                        help='The environment to plan on')    
    parser.add_argument('-s', '--start', nargs='+', type=int, required=True)
    parser.add_argument('-g', '--goal', nargs='+', type=int, required=True)

    args = parser.parse_args()

    # First setup the environment and the robot.

    # Next setup the planner
    PLANNER = {'rrt' : RRTPlanner, 'rrtstar' : RRTStarPlanner}
    K_d = {2 : '2', 8 : '8', 16 : '16', None : 'logn'}
    C = { 2 : [], 8 : [], 16 : [], None: []}
    for k in [2, 8, 16, None]:
        for i in range(5):
            planning_env = MapEnvironment(args.map, args.start, args.goal)
            planner = RRTStarPlanner(planning_env, goal_bias=0.2, step_size=10, k=k)
            extra_text = 'q3_k{}_i{}'.format(K_d[k], (i + 1))
            cost_times = main(planning_env, planner, args.start, args.goal, planner_name='rrtstar', extra_text=extra_text)
            C[k].append(cost_times)
    import pdb; pdb.set_trace()