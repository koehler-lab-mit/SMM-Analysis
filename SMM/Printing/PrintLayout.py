import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

class PrintGenerator:

    def __init__(self, coordinates, times, fixed_coordinates=None, fixed_times=None):
        if fixed_coordinates is None or fixed_times is None:
            fixed_coordinates = np.empty((0,2))
            fixed_times = np.empty(0)

        self.pop_coordinates = coordinates.astype(np.int32)
        self.fixed_coordinates = fixed_coordinates.astype(np.int32)
        self.total_coordinates = np.vstack([coordinates, fixed_coordinates]).astype(np.int32)

        self.times = np.concatenate([times, fixed_times]).astype(np.float64)

        self.pop_size = coordinates.shape[0]
        self.fixed_size = fixed_coordinates.shape[0]
        self.sample_size = self.times.size
        self.print_size = self.sample_size + self.fixed_size
        self.total_size = self.total_coordinates.shape[0]

        self.sample_idx = np.arange(self.sample_size, dtype=int)

        self.time_matrix = np.abs(self.times[:,None] - self.times[None,:])
        self.compressed_time_matrix = squareform(self.time_matrix)

        self.pop_distance_matrix = squareform(pdist(self.total_coordinates, metric='sqeuclidean'))
        mask = np.identity(self.print_size, dtype=bool)
        a = np.arange(self.sample_size, self.print_size)
        mask[a,a] = True
        self.sample_mask = ~mask
        self.compressed_sample_mask = squareform(self.sample_mask)

        self.minimum_distance = self.find_minimum_distance()

    def get_distance_matrix(self, idx=None):
        if idx is None:
            idx = self.sample_idx
        idx = np.concatenate((idx, np.arange(self.pop_size, self.total_size)))
        return np.asarray(self.pop_distance_matrix)[idx,:][:,idx]

    def find_pairwise_distances(self, idx=None):
        return squareform(self.get_distance_matrix(idx)) * self.compressed_time_matrix

    def find_minimum_distance(self, idx=None):
        a = self.find_pairwise_distances(idx=idx)
        a = a[self.compressed_sample_mask]
        return np.min(a)

    def find_worst_index(self, idx=None):
        if idx is None:
            idx = self.sample_idx
        dists = self.get_distance_matrix(idx=idx)[0:idx.size,:]
        return np.unravel_index(np.argmin(dists.ravel()), dists.shape)[0]

    def iter_swap(self, iterations):
        idx = self.sample_idx
        for _ in range(iterations):
            #distance_matrix = self.get_distance_matrix(idx) * self.time_matrix
            # Find the worst pair and distance
            worst = np.argmin()
            #   Only test the 1st index, in case the 2nd is fixed
            # For each non-fixed coordinate, swap indices
                # Check if swap improves minimum distance
                # If so, perform swap and break; more iterations > better swaps

    def random_initialize(self, iterations):
        size = self.sample_size
        cdef npc.npy_intp *arr = <npc.npy_intp *> malloc(size * sizeof(npc.npy_intp))
        if not arr:
            raise MemoryError()
        for i in range(iterations):
            for j in range(size):
                arr[j] = j
            for j in reversed(range(1, size)):
                k = rand() % size
                temp = arr[k]
                arr[k] = arr[j]
                arr[j] = temp
            if self.test_improvement(arr):
                output = np.empty(size, dtype=int)
                for j in range(size):
                    output[j] = arr[j]
                self.sample_idx = output
                self.minimum_distance = self.find_minimum_distance()
        free(arr)

    cdef bint test_improvement(self, npc.npy_intp* idx):
        cdef npc.float64_t time, distance
        cdef npc.float64_t [:] d1
        cdef Py_ssize_t i, j
        cdef npc.float64_t threshold = self.minimum_distance

        for i in range(self.sample_size):
            time = self.times[i]
            d1 = self.pop_distance_matrix[idx[i]]
            for j in range(i+1, self.total_size):
                distance = d1[idx[j]] * abs(self.times[j] - time)
                if distance < threshold:
                    return False
        return True


cdef inline npc.npy_intp[:] _swap_index(npc.npy_intp [:] arr, npc.npy_intp a, npc.npy_intp b):
    cdef npc.npy_intp temp = arr[b]
    arr = arr.copy()
    arr[b] = arr[a]
    arr[a] = temp
    return arr
