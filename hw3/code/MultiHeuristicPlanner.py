import queue as q
from heapq import heappush, heappop
import itertools

import numpy as np
from matplotlib import pyplot as plt

class PQueue(object):
    REMOVED = '<removed-task>'      # placeholder for a removed task

    def __init__(self):
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of tasks to entries
        self.counter = itertools.count()     # unique sequence count
        self.count = 0

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)
        self.counter += 1

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = entry_finder.pop(task)
        entry[-1] = PQueue.REMOVED
        self.counter -= 1

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(pq)
            if task is not PQueue.REMOVED:
                self.counter -= 1
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

    def isEmpty(self):
        return (self.counter == 0)

class Node(object):

        HAS_GENERATED = set()
        """
        RRT Node
        """
        def __init__(self, x, y, cost=0, is_ancestor_cache={}):
            self.x = x
            self.y = y
            #self.cost = cost # g
            self.is_ancestor_cache = {}
            self.is_ancestor_cache.update(is_ancestor_cache)
            Node.HAS_GENERATED.add((x, y))

        @property
        def p(self):
            return (self.x, self.y)

        def key(self, g, h, w, **kwargs):
            return g[self.p] + h(**kwargs) * w 

        def successors(self, planning_env):
            successors  = planning_env.successors(self.p)
            is_ancestor = dict(self.is_ancestor_cache)
            if self.p in is_ancestor:
                is_ancestor[self.p] = True

            successors = [Node(s_config[0], s_config[1],\
                               #cost=(self.cost + \
                               #      planning_env.compure_distance(self.p, s_config)),\
                               is_ancestor_cache=is_ancestor)
                            for s_config in successors]
            return successors

        def __eq__(self, other):
            return True

        def __lt__(self, other):
            return False

        def __repr__(self):
            return str((self.x, self.y))


class MultiHeuristicPlanner(object):
    def __init__(self, planning_env, guidance, w1=20, w2=1.35):
        """

        :param planning_env: The planning environment for the algorithm
        :param guidance: a list of tuples containing the user guidance points
        :param w1: inflation parameter of individual searches
        :param w2: The factor of comparison between anchor heuristic and inadmissible heuristics 
        """
        self.guidance = guidance
        self.planning_env = planning_env
        self.nodes = dict()

        hi = [self.planning_env.build_user_guided_heuristic(g_config) \
                for g_config in guidance]
        self.H = { (i + 1) : hi for i, hi in enumerate(hi) }
        self.H[0] = self.planning_env.compute_heuristic
        self.w1 = w1
        self.w2 = w2
        self.OPEN = []
        self.g    = {}
        self.bp   = {}
        self.CLOSED_anchor = []
        self.CLOSED_inad   = []

    def remove_state(self, s):
        for i, _ in enumerate(self.OPEN):
            new_openi = q.PriorityQueue()
            while not self.OPEN[i].empty():
                key, state = self.OPEN[i].get()
                if state.p == s.p:
                    continue
                new_openi.put((key, state))
            self.OPEN[i] = new_openi

    def insert_update(self, s, i):
        
        items = []
        kwargs = {'config' : s.p}
        if i > 0:
            kwargs.update({'is_ancestor' : s.is_ancestor_cache})
        key = s.key(self.g, self.H[i], self.w1, **kwargs)
        items.append((key, s))

        while True:
            if self.OPEN[i].empty():
                break

            key, node = self.OPEN[i].get()
            if node.p == s.p:
                break

            items.append((key, node))
        
        for key, node  in items:
            self.OPEN[i].put((key, node))

    def expand_state(self, s):
        self.remove_state(s)
        for n_s in Node.successors(s, self.planning_env):
            if n_s.p not in self.g:
                self.g[n_s.p] = float('inf')
            if n_s.p not in self.bp:
                self.bp[n_s.p] = None

            if self.g[n_s.p] > self.g[s.p] + self.planning_env.compute_distance(s.p, n_s.p):
                self.g[n_s.p]  = self.g[s.p] + self.planning_env.compute_distance(s.p, n_s.p)
                self.bp[n_s.p] = s.p

                if n_s.p not in self.CLOSED_anchor:
                    self.insert_update(n_s, 0)
                    if n_s.p not in self.CLOSED_inad:
                        kwargs = {'config'      : n_s.p}
                        key0 = n_s.key(self.g, self.H[0], self.w1, **kwargs)
                        for i, openi in enumerate(self.OPEN):
                            if i == 0:
                                continue
                            kwargs = {'config'      : n_s.p,
                                      'is_ancestor' : n_s.is_ancestor_cache }
                            keyi = n_s.key(self.g, self.H[i], self.w1, **kwargs)
                            if keyi <= self.w2 * key0:
                                self.insert_update(n_s, i)

    def extract_plan(self, goal_config):
        plan = []
        iter_config = goal_config
        while iter_config is not None:
            plan.append(iter_config)
            iter_config = self.bp[iter_config]

        plan = plan[::-1]
        return plan

    '''
    This function assume the input start_config and goal_config matches the start and goal
    in the environment object.
    '''
    def Plan(self, start_config, goal_config):
        start_config = tuple(start_config)
        goal_config  = tuple(goal_config)
        is_ancestor = { g : False for g in self.guidance}
        start_node = Node(start_config[0], start_config[1], 
                          is_ancestor_cache=is_ancestor)

        self.g[start_config]  = 0
        self.g[goal_config]   = float('inf')
        self.bp[start_config] = None
        self.bp[goal_config]  = None

        self.OPEN = [q.PriorityQueue()]
        for _ in self.guidance:
            self.OPEN.append(q.PriorityQueue())

        kwargs = {'config' : start_node.p}
        key = start_node.key(self.g, self.H[0], self.w1, **kwargs)
        self.OPEN[0].put((key, start_node))

        for i, open_i in enumerate(self.OPEN):
            if i == 0:
                continue
            kwargs = {'config'      : start_node.p,
                      'is_ancestor' : start_node.is_ancestor_cache }
            key = start_node.key(self.g, self.H[i], self.w1, **kwargs)
            open_i.put((key, start_node))

        iter_count = 0
        while True:
            if self.OPEN[0].empty():
                return self.extract_plan(goal_config)

            min_key0, min_node0 = self.OPEN[0].get() # take min and push back
            self.OPEN[0].put((min_key0, min_node0))
            if min_key0 >= float('inf'):
                break

            for (i, open_i) in enumerate(self.OPEN):
                if i == 0:
                    continue

                is_empty = self.OPEN[i].empty() 
                if not is_empty:
                    min_keyi, min_nodei = self.OPEN[i].get() # take min and push back
                    self.OPEN[i].put((min_keyi, min_nodei))

                if (not is_empty) and min_keyi <= self.w2 * min_key0:
                    choice = 'open1'
                    print(min_keyi, min_key0, self.w2*min_key0)
                    if self.g[goal_config] <= min_keyi:
                        if self.g[goal_config] < float('inf'):
                            return self.extract_plan(goal_config) 
                    else:
                        s = min_nodei
                        self.expand_state(s)
                        self.CLOSED_inad.append(s.p)

                else:
                    choice = 'open0'
                    if self.g[goal_config] <= min_key0:
                        if self.g[goal_config] < float('inf'):
                            return self.extract_plan(goal_config)
                    else:
                        s = min_node0
                        self.expand_state(s)
                        self.CLOSED_anchor.append(s.p)

                print(iter_count, s, choice, self.OPEN[0].qsize(), self.OPEN[1].qsize())
                if (iter_count % 3) == 0:
                    self.planning_env.draw_graph(bp=self.bp)
                iter_count += 1