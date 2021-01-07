import numpy as np
from scipy.spatial.distance import squareform, pdist
cimport numpy as npc
cimport cython
from numpy.random import Generator, SFC64

cdef class PrintGrid:
    cdef:
        int [:,:] coordinates
        Py_ssize_t size
        npc.float64_t[:,:] distance_matrix
        npc.float64_t [:] compressed_distance_matrix

    def __init__(self, coordinates):
        self.coordinates = coordinates.astype(np.int32)
        self.size = coordinates.shape[0]
        self.distance_matrix = squareform(pdist(self.coordinates, metric='sqeuclidean'))


cdef class PrintOrder:
    cdef:
        Py_ssize_t print_size, fixed_size, empty_size, var_size
        npc.float64_t [:] times, condensed_time_matrix
        npc.float64_t [:,:] time_matrix

    def __init__(self, times, fixed_size, grid_size):
        self.fixed_size = fixed_size
        self.empty_size = grid_size - times.size
        self.var_size = grid_size - fixed_size
        times = np.concatenate((
            times[:-fixed_size],
            np.full(self.empty_size, np.nan),
            times[-fixed_size:]
        ))
        self.times = times
        self.time_matrix = np.abs(times[:,None] - times[None,:])
        self.condensed_time_matrix = squareform(self.time_matrix)


cdef class PrintLayout:
    cdef:
        PrintGrid grid
        PrintOrder order
        Py_ssize_t var_size
        npc.npy_intp [:] idx
        npc.float64_t [:,:] _euclidean_matrix
        npc.float64_t [:,:] _distance_matrix
        npc.float64_t [:] _variable_distances
        npc.float64_t [:] _compressed_distances


    def __init__(self, PrintGrid grid, PrintOrder order, npc.npy_intp [:] idx):
        self.grid = grid
        self.order = order
        self.var_size = order.var_size
        self.idx = idx
        self._compressed_distances = None
        self._euclidean_matrix = None
        self._distance_matrix = None
        self._variable_distances = None

    def euclidean_matrix(self):
        if self._euclidean_matrix is None:
            dist = np.asarray(self.grid.distance_matrix)[self.idx,:][:,self.idx]
            dist[np.eye(dist.shape[0])]=np.nan
            dist[self.var_size:dist.shape[0], self.var_size:dist.shape[1]] = np.nan
            self._euclidean_matrix = dist
        return self._euclidean_matrix

    def compressed_distances(self):
        if self._compressed_distances is None:
            self._compressed_distances = squareform(self.euclidean_matrix()) * self.order.condensed_time_matrix
        return self._compressed_distances

    def minimum_variable_distance(self):
        return np.nanmin(self.compressed_distances())


def random_initialize(PrintGrid grid, PrintOrder order,
                      int iterations, npc.float64_t threshold=0, PrintLayout best=None):
    if best is None:
        best = PrintLayout(grid, order, np.arange(order.print_size), order.var_size)
    cdef PrintLayout testing

    shuffle = Generator(SFC64()).permutation

    cdef npc.float64_t[:, :] distance_matrix = grid.distance_matrix
    cdef npc.float64_t [:] times = order.times

    cdef npc.npy_intp i, j, k, temp
    cdef Py_ssize_t total_size = order.print_size

    for i in range(iterations):
        arr = shuffle(var_size)
        if meets_threshold(threshold, arr, var_size, total_size, times, distance_matrix):
            testing = PrintLayout(grid, order, arr, var_size)
            if best is None or compare_ranked_layouts(testing, best):
                best = testing
                threshold = testing.minimum_variable_distance()
    return best

def iterative_swap(PrintLayout pl, int iterations):
    cdef:
        int var_size = pl.var_size
        int i = 0
        int j,k
        npc.npy_intp worst1, worst2
        npc.float64_t worst_distance, worst_
        npc.float64_t [:] worstv1, worstv2
        npc.float64_t [:] vec1 = np.triu_indices(pl.order.size)[0]
    for i in range(iterations):
        # Get the worst spot index and distance
        worst1 = np.argmin(pl.compressed_distances())
        worst_distance = pl.compressed_distances()[worst1]
        worst1 = vec1[worst1]
        worstv1  = pl.euclidean_matrix()[vec1[worst1]]
        # For every other spot it could move to:
        for j in range(worstv1.size):

            # For every distance that would be altered:
                # If it's less than the worst distance, break to the next position
                # If every distance is greater than the distance, update the PrintLayout pl



cdef bint meets_threshold(npc.float64_t threshold, npc.npy_intp [:] idx,
                           Py_ssize_t sample_size, Py_ssize_t total_size,
                           npc.float64_t [:] times,
                           npc.float64_t [:,:] pop_distance_matrix):
    cdef npc.float64_t time, distance
    cdef npc.float64_t [:] d1
    cdef Py_ssize_t i, j

    for i in range(sample_size):
        time = times[i]
        d1 = pop_distance_matrix[idx[i]]
        for j in range(i+1, total_size):
            distance = d1[idx[j]] * abs(times[j] - time)
            if distance < threshold:
                return False
    return True


cdef bint compare_ranked_layouts(PrintLayout A, PrintLayout B):
    cdef npc.float64_t [:] a_order = np.sort(A.variable_distances())
    cdef npc.float64_t [:] b_order = np.sort(B.variable_distances())
    cdef int i
    cdef npc.float64_t a, b
    for i in range(a_order.size):
        a = a_order[i]
        b = b_order[i]
        if a>b: return True
        if a<b: return False
    return False
