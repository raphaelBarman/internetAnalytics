import networkx as nx
import numpy as np
import numpy.linalg as la

# Creates a transition matrix from a graph
def get_transition_matrix(graph):
    N = nx.number_of_nodes(graph)
    # First every cell of the transition matrix is initialized to 0
    transition_matrix = np.zeros((N, N), dtype=float)
    # Compute the transition matrix :
    for n in graph.nodes():
        out_deg = graph.out_degree(n)
        if(out_deg != 0):
            # If the out degree of node n is not 0,
            # we compute the fraction 1/outdegree(u) and
            # replace this value in the cell for all successors of n
            out_deg_fraction = 1./float(out_deg)
            for s in graph.successors(n):
                transition_matrix[int(n)][int(s)] = out_deg_fraction
    return transition_matrix
# Computes the new transition matrix that takes into
# account the dangling vector
def get_new_transition_matrix(graph):
    transition_matrix = get_transition_matrix(graph)
    N = transition_matrix.shape[0]
    # Select rows with only zeros
    zeroes = np.where(~transition_matrix.any(axis=1))[0]
    # We create an indicator vector from the rows with
    # only zeroes
    w = [0.] * N
    for i in zeroes:
        w[i] = 1.
    ones = np.ones(N,)
    new_transition_matrix = transition_matrix + np.dot(w, ones) * 1 / N
    return new_transition_matrix
# Computes the google matrix for the given graph and theta
def google_matrix(graph, theta):
    new_transition_matrix = get_new_transition_matrix(graph)
    N = new_transition_matrix.shape[0]
    google_matrix = theta * new_transition_matrix + (1 - theta) * np.ones((N,N)) * (1. / N)
    return google_matrix
# Performs a power iteration using the given matrix,
# initial vector and stop when the difference of the norm
# between two iterations is smaller than threshold
def power_iter(G,init, threshold):
    diff = la.norm(init)
    prev = init
    curr = init
    while(diff > threshold):
        curr = np.dot(prev,G)
        diff = np.abs(la.norm(curr)-la.norm(prev))
        prev = curr
    return curr
