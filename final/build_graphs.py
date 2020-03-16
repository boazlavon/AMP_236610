import argparse
import os
from os import listdir
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache

# col index
COST_INDEX       = 5
COVEREGE_INDEX   = 6
TOTAL_TIME_INDEX = 11

AXIS_LABELS  =  { COST_INDEX : 'Length', COVEREGE_INDEX : 'Coverage (%)', TOTAL_TIME_INDEX : 'Time (ms)' }
FNAME_LABELS  = { COST_INDEX : 'cost', COVEREGE_INDEX   : 'cov',          TOTAL_TIME_INDEX : 'time' }

ROBOT_PLANAR = 'planar'

def load_results(robot, exec_name):
    data_path = os.path.join('.', 'data', robot, exec_name)
    files = [f for f in listdir(data_path) if f.startswith('f') and not f.endswith('result')]
    experiments = {}
    for file in files:
        f = float(file[ file.find('_') +1 : file.find('eps')] )
        eps = float(file[ file.find('eps')+4 : file.find('p_')] )
        p = float(file[ file.find('p_') + 2 :])
        experiments[(f,eps,p)] = open(join(data_path, file)).read()
    return (experiments, files)

def build_time_vs_coverage_graph(experiments, x_axis, y_axis, in_f=None, in_eps=None, in_p=None, output_dir=None, robot=ROBOT_PLANAR):
    plt.clf()
    keys = experiments.keys()
    keys = [(f,eps,p) for f,eps,p in keys if ((in_f == f) or (in_f is None))]
    keys = [(f,eps,p) for f,eps,p in keys if ((in_eps == eps) or (in_eps is None))]
    keys = [(f,eps,p) for f,eps,p in keys if ((in_p == p) or (in_p is None))]
    max_cost = 0
    for params in keys:
        f, eps, p = params

        # raw btyes to table
        lines =   experiments[params].split('\n')
        words =   [line.split() for line in lines[:-1]]

        cost     = [       float(word[COST_INDEX])            for word in words ]
        max_cost = max(max_cost, max(cost))

        coverage = [ 100 * float(word[COVEREGE_INDEX]) for word in words ]
        time     = [       int(word[TOTAL_TIME_INDEX]) for word in words ]

        RESULTS  = { COST_INDEX : cost, COVEREGE_INDEX : coverage, TOTAL_TIME_INDEX : time }
        x_values = RESULTS[x_axis]
        x_values = [0.01 if x == 0 else x for x in x_values]

        y_values = RESULTS[y_axis]

        label = ''
        if in_f is None:
            label += f'f={f} '
        if in_eps is None:
            label += f'\u03B5={eps} '
        if in_p is None:
            label += f'p={p}'
        plt.plot(x_values, y_values, label=label)

    title = f'{robot} '
    if in_f is not None:
        title += f'f={in_f} '
    if in_eps is not None:
        title += f'\u03B5={in_eps} '
    if in_p is not None:
        title += f'p={in_p}'

    in_f   = 'iter'   if in_f     is None else in_f
    in_p   = 'iter'   if in_p   is None else in_p
    in_eps = 'iter'   if in_eps is None else in_eps

    filename = f'{robot} f_{in_f}eps_{in_p}p_{in_eps}_x{FNAME_LABELS[x_axis]}_y{FNAME_LABELS[y_axis]}.png'
    plt.xlabel(AXIS_LABELS[x_axis])
    plt.ylabel(AXIS_LABELS[y_axis])
    plt.title(title)
    plt.legend(loc='best')
    if x_axis == TOTAL_TIME_INDEX:
        plt.xscale('log')

    if output_dir is not None:
        filename = os.path.join(output_dir, filename)

    if y_axis == COST_INDEX:
        max_cost = int(max_cost)
        plt.ylim([0, ((max_cost / 10)) * 10])
        plt.yticks(np.arange(0, (1 + (max_cost / 10)) * 10, 10))
    plt.savefig(filename)

def build_graphs(experiments, files, robot):
    GRAPHS = [(COVEREGE_INDEX, COST_INDEX), (TOTAL_TIME_INDEX, COVEREGE_INDEX)]
    os.system(f'rm -rv ./output')
    for x_axis, y_axis in GRAPHS:
        PREFIX = join('output', f'{robot}_x{FNAME_LABELS[x_axis]}_y{FNAME_LABELS[y_axis]}')
        output_dir = PREFIX + '_iter_f'
        os.system(f'mkdir -pv {output_dir}')
        for in_f, in_eps, in_p in set([(None, in_eps, in_p) for in_f, in_eps, in_p in experiments.keys()]):
            build_time_vs_coverage_graph(experiments, x_axis, y_axis, in_f=in_f, in_eps=in_eps, in_p=in_p, output_dir=output_dir, robot=robot)
        
        output_dir = PREFIX + '_iter_eps'
        os.system(f'mkdir -pv {output_dir}')
        for in_f, in_eps, in_p in set([(in_f, None, in_p) for in_f, in_eps, in_p in experiments.keys()]):
            build_time_vs_coverage_graph(experiments, x_axis, y_axis, in_f=in_f, in_eps=in_eps, in_p=in_p, output_dir=output_dir, robot=robot)

        output_dir = PREFIX + '_iter_p'
        os.system(f'mkdir -pv {output_dir}')
        for in_f, in_eps, in_p in set([(in_f, in_eps, None) for in_f, in_eps, in_p in experiments.keys()]):
            build_time_vs_coverage_graph(experiments, x_axis, y_axis, in_f=in_f, in_eps=in_eps, in_p=in_p, output_dir=output_dir, robot=robot)

        output_dir = PREFIX + '_iter_f,p'
        os.system(f'mkdir -pv {output_dir}')
        for in_f, in_eps, in_p in set([(None, in_eps, None) for in_f, in_eps, in_p in experiments.keys()]):
            build_time_vs_coverage_graph(experiments, x_axis, y_axis, in_f=in_f, in_eps=in_eps, in_p=in_p, output_dir=output_dir, robot=robot)

        output_dir = PREFIX + '_iter_f,eps'
        os.system(f'mkdir -pv {output_dir}')
        for in_f, in_eps, in_p in set([(None, None, in_p) for in_f, in_eps, in_p in experiments.keys()]):
            build_time_vs_coverage_graph(experiments, x_axis, y_axis, in_f=in_f, in_eps=in_eps, in_p=in_p, output_dir=output_dir, robot=robot)

        output_dir = PREFIX + '_iter_p,eps'
        os.system(f'mkdir -pv {output_dir}')
        for in_f, in_eps, in_p in set([(in_f, None, None) for in_f, in_eps, in_p in experiments.keys()]):
            build_time_vs_coverage_graph(experiments, x_axis, y_axis, in_f=in_f, in_eps=in_eps, in_p=in_p, output_dir=output_dir, robot=robot)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot'    , default=ROBOT_PLANAR)
    parser.add_argument('--exec_name', default='seed_42_vertices_1000')
    args = parser.parse_args()
    robot, exec_name= args.robot, args.exec_name
    experiments, files = load_results(robot, exec_name)
    build_graphs(experiments, files, robot)

if __name__ == '__main__':
    main()