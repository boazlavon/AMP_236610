import math
import argparse
import os
from os import listdir
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

EXEC_NAME = 'seed_42_vertices_1000'

# col index
COST_INDEX       = 5
COVEREGE_INDEX   = 6
TOTAL_TIME_INDEX = 11

AXIS_LABELS  =  { COST_INDEX : 'Length', COVEREGE_INDEX : 'Coverage (%)', TOTAL_TIME_INDEX : 'Time (ms)' }
FNAME_LABELS  = { COST_INDEX : 'cost',   COVEREGE_INDEX : 'cov',          TOTAL_TIME_INDEX : 'time' }

ROBOT_PLANAR = 'planar'
ROBOT_CRISP  = 'crisp'
ROBOT_DRONE  = 'drone'
ALL_ROBOTS   = [ROBOT_PLANAR, ROBOT_CRISP, ROBOT_DRONE]
GOOD_COV     = {ROBOT_PLANAR : 90, ROBOT_DRONE :60, ROBOT_CRISP : 40}

def load_results(robot, exec_name):
    data_path = os.path.join('.', 'data', robot, exec_name)
    files = [f for f in listdir(data_path) if f.startswith('f') and not f.endswith('result')]
    experiments_raw = {}
    for file in files:
        f   = float(file[ file.find('_') +1   : file.find('eps')] )
        eps = float(file[ file.find('eps')+4  : file.find('p_')] )
        p   = float(file[ file.find('p_') + 2 :])
        experiments_raw[(f,eps,p)] = open(join(data_path, file)).read()

    experiments = {}
    for params in experiments_raw.keys():
        lines     = experiments_raw[params].split('\n')
        words     = [ line.split() for line in lines[:-1] ]
        cost      = [        float(word[COST_INDEX])       for word in words ]
        coverage  = [ 100 *  float(word[COVEREGE_INDEX])   for word in words ]
        time      = [          int(word[TOTAL_TIME_INDEX]) for word in words ]
        experiments[params] = { COST_INDEX : cost, COVEREGE_INDEX : coverage, TOTAL_TIME_INDEX : time }
    return experiments


def search_coverage(experiments, enough_cov=None, choose_key=TOTAL_TIME_INDEX):
    max_cov = 0
    max_cov_keys = {}
    if enough_cov is None:
        for key in experiments.keys():
            max_cov = max(max_cov, max(experiments[key][COVEREGE_INDEX]))
        enough_cov = max_cov

    for key in experiments.keys():
        if max(experiments[key][COVEREGE_INDEX]) >= enough_cov:
            max_cov_keys[key] = max(experiments[key][choose_key])
    if max_cov_keys:
        chosen_params, chosen_value = min(max_cov_keys.items(), key=lambda item: item[1])
    else:
        chosen_params, chosen_value = (None, None, None), None
    return chosen_params, enough_cov, chosen_value


def frange(start, stop=None, step=None):
    start = float(start)
    if stop == None:
        stop = start + 0.0
        start = 0.0
    if step == None:
        step = 1.0

    count = 0
    while True:
        temp = float(start + count * step)
        if step > 0 and temp >= stop:
            break
        elif step < 0 and temp <= stop:
            break
        yield temp
        count += 1

def build_graph(experiments, x_axis, y_axis, in_f=None, in_eps=None, in_p=None, output_dir=None, robot=ROBOT_PLANAR):
    plt.clf()
    keys = experiments.keys()
    keys = [(f, eps, p) for f,eps,p in keys if ((in_f == f) or (in_f is None))]
    keys = [(f, eps, p) for f,eps,p in keys if ((in_eps == eps) or (in_eps is None))]
    keys = [(f, eps, p) for f,eps,p in keys if ((in_p == p) or (in_p is None))]

    max_cost = 0
    for params in keys:
        f, eps, p = params
        x_values = experiments[params][x_axis]
        x_values = [min(min([x for x in x_values if x != 0]), 0.01)\
                    if x == 0 else x for x in x_values]
        y_values = experiments[params][y_axis]
        if y_axis == COST_INDEX:
            max_cost = max(max_cost, max(y_values))

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

    in_f   = 'iter'   if in_f   is None else in_f
    in_p   = 'iter'   if in_p   is None else in_p
    in_eps = 'iter'   if in_eps is None else in_eps

    plt.xlabel(AXIS_LABELS[x_axis])
    plt.ylabel(AXIS_LABELS[y_axis])
    plt.title(title)
    plt.legend(loc='best')
    if x_axis == TOTAL_TIME_INDEX:
        plt.xscale('log')

    if y_axis == COST_INDEX:
        step  = (max(y_values) - min(y_values)) / 10.0
        if max_cost < 1:
            y_max = 1.1 * max_cost
            yticks = list(frange(0, y_max, step=step)) + [y_max]
        else:
            max_cost = int(max_cost)
            step  = int(step) + (10 - (int(step) % 10))
            y_max = 5 + (max_cost / 10) * 10
            yticks = np.arange(0, y_max, step)

        plt.ylim([0, y_max])
        plt.yticks(yticks)

    filename = f'{robot} f_{in_f}eps_{in_p}p_{in_eps}_x{FNAME_LABELS[x_axis]}_y{FNAME_LABELS[y_axis]}.png'
    if output_dir is not None:
        filename = os.path.join(output_dir, filename)
    plt.savefig(filename)


ITER_BIT_FLAGS = {0b100 : 'f', 0b010 : 'eps', 0b001 : 'p'}
def mask_tuple(in_tuple, mask):
    output_tuple = list(in_tuple)
    for flag in ITER_BIT_FLAGS.keys():
        if flag & mask:
            output_tuple[int(math.log(flag,2))] = None
    return tuple(output_tuple)

def build_graphs(experiments, robot):
    GRAPHS = [(COVEREGE_INDEX, COST_INDEX), (TOTAL_TIME_INDEX, COVEREGE_INDEX)]
    os.system(f'rm -r ./output/{robot}')
    for x_axis, y_axis in GRAPHS:
        OUTPUTDIR_PREFIX = join('output', f'{robot}', f'{robot}_x{FNAME_LABELS[x_axis]}_y{FNAME_LABELS[y_axis]}')
        for option in range(1,7):
            suffix = ','.join([ITER_BIT_FLAGS[flag] for flag in ITER_BIT_FLAGS.keys() if (option & flag)])
            print(suffix)
            output_dir = OUTPUTDIR_PREFIX + f'_iter_{suffix}'
            os.system(f'mkdir -pv {output_dir}')
            masked_experiments = set([mask_tuple(params, option) for params in experiments.keys()])
            for in_f, in_eps, in_p in masked_experiments:
                build_graph(experiments, x_axis, y_axis, in_f=in_f, in_eps=in_eps, in_p=in_p, output_dir=output_dir, robot=robot)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphs',    dest='graphs', action='store_true')
    parser.add_argument('--no-graphs', dest='graphs', action='store_false')
    parser.set_defaults(graphs=True)
    args = parser.parse_args()
    do_build_graph = args.graphs

    for robot in ALL_ROBOTS:
        print('=============================================')
        experiments = load_results(robot, EXEC_NAME)
        print(f'ROBOT: {robot}')
        
        # search params having maximal coverage
        params, opt_cov, chosose_value = search_coverage(experiments, choose_key=TOTAL_TIME_INDEX)
        f, eps, p = params
        print(f'max coverage = {opt_cov} (min time = {chosose_value}): f={f}, \u03B5={eps}, p={p}')
        params, opt_cov, chosose_value = search_coverage(experiments, choose_key=COST_INDEX)
        f, eps, p = params
        print(f'max coverage = {opt_cov} (min length = {chosose_value}): f={f}, \u03B5={eps}, p={p}')

        params, opt_cov, chosose_value = search_coverage(experiments, enough_cov=GOOD_COV[robot], choose_key=TOTAL_TIME_INDEX)
        f, eps, p = params
        print(f'at least {opt_cov}% coverage (min time = {chosose_value}): f={f}, \u03B5={eps}, p={p}')
        params, opt_cov, chosose_value = search_coverage(experiments, enough_cov=GOOD_COV[robot], choose_key=COST_INDEX)
        f, eps, p = params
        print(f'at least {opt_cov}% coverage (min length = {chosose_value}): f={f}, \u03B5={eps}, p={p}')

        print()
        print('Build Graphs')
        if do_build_graph:
            build_graphs(experiments, robot)
        print('=============================================')
        
if __name__ == '__main__':
    main()