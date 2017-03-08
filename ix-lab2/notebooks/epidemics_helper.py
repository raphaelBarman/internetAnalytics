"""
epidemic_helper.py: Helper module to simulate continuous-time stochastic 
SIR epidemics.

Copyright © 2017 — LCA 4
"""
import time
import bisect
import numpy as np
import networkx as nx
from numpy import random as rd


class OrderedProcessingList:
    """
    List of ('event','time') ordered by 'time' used for the cascades
    'time' is assumed to be a float

    The data structure is implemented as two lists: an ordered list of times
    and a list a corresponding event

    The ordering is maintained using the bisect library for items insertion
    """

    def __init__(self):
        self.time_list = list()  # List of times
        self.event_list = list()   # List of events
        self.event_set = set() # Set of events (for faster access)

    def __getitem__(self, event):
        idx = self.event_list.index(event)
        return self.time_list[idx]

    def _update_item(self, event, time):
        idx = bisect.bisect(self.time_list, time)
        self.event_list.insert(idx, event)
        self.time_list.insert(idx, time)
        self.event_set.add(event)

    def __setitem__(self, event, time):
        """
        Set item ('event', 'time') only if 'time' is prior to the existing one
        (if it exists)
        """
        if event in self.event_set:
            # If 'event' is already in the list, get its index
            old_idx = self.event_list.index(event)
            # If existing time is posterior to 'time', update it
            if self.time_list[old_idx] > time:
                # Remove old value
                self.time_list.pop(old_idx)
                self.event_list.pop(old_idx)
                # Add the new one
                self._update_item(event, time)
        else:
            # If 'event' is not in the list, add it
            self._update_item(event, time)

    def pop(self, index):
        event = self.event_list.pop(index)
        time = self.time_list.pop(index)
        self.event_set.remove(event)
        return event, time

    def __len__(self):
        assert len(self.time_list) == len(self.event_list)
        return len(self.time_list)

    def __str__(self):
        return str(zip(self.event_list, self.time_list))

    def __repr__(self):
        return repr(zip(self.event_list, self.time_list))   


class ProgressPrinter(object):
    """
    Helper object to print relevant information throughout the epidemic
    """
    PRINT_INTERVAL = 0.1
    _PRINT_MSG = ('Epidemic spreading... ' 
                  '{t:.2f} days elapsed | '
                  '{S:.1f}% susceptible, {I:.1f}% infected, '
                  '{R:.1f}% recovered')
    _PRINTLN_MSG = ('Epidemic stopped after {t:.2f} days | '
                    '{t:.2f} days elapsed | '
                    '{S:.1f}% susceptible, {I:.1f}% infected, '
                    '{R:.1f}% recovered')

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.last_print = time.time()

    def print(self, sir_obj, epitime, end='', force=False):
        if not self.verbose:
            return
        if (time.time() - self.last_print > self.PRINT_INTERVAL) or force:
            S = np.sum(sir_obj.status==0) * 100. / sir_obj.n_nodes
            I = np.sum(sir_obj.status==1) * 100. / sir_obj.n_nodes
            R = np.sum(sir_obj.status==2) * 100. / sir_obj.n_nodes
            print('\r', self._PRINT_MSG.format(t=epitime, S=S, I=I, R=R), 
                  sep='', end=end, flush=True)
            self.last_print = time.time()

    def println(self, sir_obj, epitime):
        if not self.verbose:
            return
        S = np.sum(sir_obj.status==0) * 100. / sir_obj.n_nodes
        I = np.sum(sir_obj.status==1) * 100. / sir_obj.n_nodes
        R = np.sum(sir_obj.status==2) * 100. / sir_obj.n_nodes
        print('\r', self._PRINTLN_MSG.format(t=epitime, S=S, I=I, R=R), 
            sep='', end='\n', flush=True)
        self.last_print = time.time()


class SimulationSIR(object):
    """
    Simulate continuous-time SIR epidemic with exponentially distributed 
    infection and recovery rates.

    Attributes:
    ----------
    G : networkx.Graph
        Propagation network, the ids of the nodes must be consecutive integers 
        starting at 0.
    n_nodes : int
        Number of nodes in the graph G
    beta : float
        Exponential infection rate (non-negative)
    gamma : float
        Exponential recovery rate (non-negative)
    status : numpy.ndarray
        Array of shape `n_nodes` indicating the status of each node,
        `0` stands for susceptible or healthy,
        `1` stands for infected, 
        `2` stands for recovered or dead
    inf_time : numpy.ndarray
        Array of shape `n_nodes` indicating the time of infection of each node,
        The default value for each not-infected nodes is infinity
    rec_time : numpy.ndarray
        Array of shape `n_nodes` indicating the time of recovery of each node,
        The default value for each not-recovered nodes is infinity
    infector :  numpy.ndarray
        Array of shape `n_nodes` indicating the who infected who
        The default value for each not-infected nodes is NaN
    """
    STATE_SPACE = [0, 1, 2]

    def __init__(self, G, beta, gamma, verbose=True):
        """
        Init an SIR cascade over a graph

        Arguments:
        ---------
        G : networkx.Graph()
                Graph over which the epidemic propagates
        beta : float
            Exponential infection rate (must be non-negative)
        gamma : float
            Exponential recovery rate (must be non-negative)
        verbose : bool (default: True)
            Indicate the print behavior, if set to False, nothing will be
            printed
        """
        # Propagatin network
        if not isinstance(G, nx.Graph):
            raise ValueError('Invalid graph type, must be networkx.Graph')
        if not set(G.nodes()) == set(range(len(G.nodes()))):
            raise ValueError('Invalid node ordering')
        self.G = G
        # Cache the number of nodes
        self.n_nodes = len(G.nodes())
        # Infection rate
        if beta < 0:
            raise ValueError('Invalid `beta` {} (must be non-negative)')
        self.beta = beta
        # Recovery rate
        if gamma < 0:
            raise ValueError('Invalid `gamma` {} (must be non-negative)')
        self.gamma = gamma
        # Printer for logging
        self._printer = ProgressPrinter(verbose=verbose)

    def get_node_status(self, node, time):
        """
        Get the status of a node at a given time
        """
        try:
            if self.inf_time[node] > time:
                return 0
            elif self.rec_time[node] > time:
                return 1
            else:
                return 2
        except IndexError:
            raise ValueError('Invalid node `{}`'.format(node))

    def _draw_edge_delay(self):
        """
        Draw the infection delay of every edge
        """
        edge_list = self.G.edges()
        n_edges = len(edge_list)
        edge_delay = rd.exponential(1./self.beta, size=n_edges)
        self._edge_delay = {}
        for (u, v), d in zip(edge_list, edge_delay):
            self._edge_delay[(u, v)] = d
            self._edge_delay[(v, u)] = d

    def _draw_node_delay(self):
        """
        Draw the recovery delay of every node
        """
        node_list = self.G.nodes()
        n_nodes = len(node_list)
        node_delay = rd.exponential(1./self.gamma, size=n_nodes)
        self._node_delay = np.zeros(self.n_nodes)
        for n, d in zip(node_list, node_delay):
            self._node_delay[n] = d

    def _process_child_infection(self, node, recovery_time, child, time):
        """Deal with neighbors infections"""
        infection_time = time + self._edge_delay[(node, child)]
        if infection_time <= self.max_time:
            if infection_time < recovery_time:
                if self.inf_time[child] > infection_time:
                    self.inf_time[child] = infection_time
                    self.infector[child] = node
                    self.processing[(child,'inf')] = infection_time

    def _process_infection_event(self, node, time):
        """
        Mark node `node` as infected at time `time`, then set its recovery 
        time and neighbors infection times to the processing list
        """
        # Mark node as infected
        self.status[node] = 1
        # Set recovery event
        recovery_time = time + self._node_delay[node]
        self.processing[(node, 'rec')] = recovery_time
        # Set neighbors infection events
        for child in self.G.neighbors(node):
            if self.status[child] == 0:
                self._process_child_infection(node, recovery_time, child, time)

    def _process_recovery_event(self, node, time):
        """
        Mark node `node` as recovered at time `time`
        """
        if time <= self.max_time:
            self.rec_time[node] = time
            self.status[node] = 2

    def _init_run(self, source, max_time):
        """
        Initialize the run of the epidemic, starting at node `source` for at
        most `max_time` units of time
        """
        if source not in self.G.nodes():
            raise ValueError('The source node {} is not in graph'.format(
                source))
        # Edge delays for infections
        self._draw_edge_delay()
        # Node delays for recoveries
        self._draw_node_delay()
        # Nodes status (0: Susceptible, 1: Infected, 2: Recovered)
        self.status = np.zeros(self.n_nodes, dtype='int8')
        # Infection times (inf by default)
        self.inf_time = np.inf * np.ones(self.n_nodes, dtype='float')
        # Keep track of who infected who (nan by default)
        self.infector = np.nan * np.ones(self.n_nodes, dtype='int')
        # Recovery times (inf by default)
        self.rec_time = np.inf * np.ones(self.n_nodes, dtype='float')
        # Maximum epidemic time
        self.max_time = max_time
        # Set the source node
        self.source = source
        # Infect the source
        self.inf_time[self.source] = 0
        # Events to process in order
        self.processing = OrderedProcessingList()
        self.processing[(self.source, 'inf')] = 0

    def launch_epidemic(self, source, max_time=np.inf):
        """
        Run the epidemic, starting from node 'source', for at most `max_time` 
        units of time
        """
        self._init_run(source, max_time)
        # Init epidemic time to 0
        time = 0
        last_print = 0
        while self.processing:
            # Get the next event to process
            (node, event_type), time = self.processing.pop(0)
            if time > self.max_time:
                break  # Stop at then end of the observation window
            # Process the event
            if event_type == 'inf':
                self._process_infection_event(node, time)
            elif event_type == 'rec':
                self._process_recovery_event(node, time)
            else:
                raise ValueError("Invalid event type")
            self._printer.print(self, time)
        self._printer.println(self, time)
        self.max_time = time
        # Free memory
        del self.processing
