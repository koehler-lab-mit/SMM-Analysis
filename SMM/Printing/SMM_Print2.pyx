import numpy as np
from scipy.spatial.distance import squareform, pdist
cimport numpy as cnp
from numpy cimport float32_t, npy_intp
cimport cython
from numpy.random import Generator, SFC64
from cython cimport view


_generator = Generator(SFC64())
shuffle = _generator.shuffle

cdef inline npy_intp idx_condensed(npy_intp i, npy_intp j, int n):
    return n*j - j*(j+1)//2 + i - 1 - j

cdef inline float32_t print_distance(float32_t x1, float32_t y1, float32_t t1, float32_t x2, float32_t y2, float32_t t2):
    return ((x1-x2)**2 + (y1-y2)**2) * abs(t1-t2)

cdef int size_of_condensed_matrix(var_size, fixed_size): # Not correct
    return (var_size * (var_size-1) / 2) + (var_size * fixed_size)

cdef float32_t [:] print_distance_matrix(float32_t[:,:] ddt, int var_size, int total_size):
    cdef:
        int i, j, k=0
        Py_ssize_t size = ddt.shape[0]
        float32_t [:] arr = np.empty(shape, dtype=np.float32)
        float32_t [:] temp, temp2
        float32_t x1, y1, t1

    for i in range(var_size):
        temp = ddt[i]
        x1 = temp[0]; y1 = temp[1], t1 = temp[2]
        for j in range(i, total_size):
            temp2 = ddt[j]
            arr[k] = print_distance(temp, x1, y1, t1, temp2[0], temp2[1], temp2[2])
            k+=1
    return arr

cdef class PrintGrid:
    cdef:
        int [:,:] coordinates
        Py_ssize_t size, var_size, fixed_size
        float32_t[:,:] distance_matrix
        float32_t [:] compressed_distance_matrix
        npy_intp [:] compressed_row, compressed_col

    def __init__(self, var_coordinates, fixed_coordinates):
        self.coordinates = np.ascontiguousarray(
            np.concatenate((var_coordinates, fixed_coordinates))
        ).astype(np.int32)
        self.var_size = var_coordinates.shape[0]
        self.fixed_size = fixed_coordinates.shape[0]
        self.size = self.var_size + self.fixed_size

        distance_matrix = squareform(pdist(self.coordinates, metric='sqeuclidean')).astype(np.float32)
        distance_matrix[np.eye(self.size, dtype=np.bool)] = np.nan
        distance_matrix[self.var_size:, self.var_size:] = np.nan
        self.distance_matrix = distance_matrix
        self.compressed_distance_matrix = squareform(distance_matrix, checks=False)
        n = np.triu_indices(self.size, 1)
        self.compressed_row = n[0]
        self.compressed_col = n[1]


cdef class PrintLayout:
    cdef:
        PrintGrid grid
        public float32_t [:] times
        float32_t [:,:] _distance_matrix, _time_matrix
        float32_t [:] _compressed_distances, _compressed_times
        int var_size


    def __init__(self, PrintGrid grid, float32_t [:] times):
        self.grid = grid
        self.times = times
        self.var_size = grid.var_size
        self._compressed_times = None
        self._time_matrix = None
        self._compressed_distances = None
        self._distance_matrix = None

    cdef float32_t[:,:] time_matrix(self):
        if self._time_matrix is None:
            times = np.asarray(self.times)
            self._time_matrix = np.abs(times[:,None] - times[None,:])
        return self._time_matrix

    cdef float32_t[:] compressed_time_matrix(self):
        if self._compressed_times is None:
            self._compressed_times = squareform(self.time_matrix(), checks=False)
        return self._compressed_times

    cpdef float32_t[:] compressed_distance_matrix(self):
        if self._compressed_distances is None:
            self._compressed_distances = np.asarray(self.grid.compressed_distance_matrix) * np.asarray(self.compressed_time_matrix())
        return self._compressed_distances

    cdef float32_t[:,:] distance_matrix(self):
        if self._distance_matrix is None:
            self._distance_matrix = squareform(self.compressed_distance_matrix())
        return self._distance_matrix

    cdef PrintLayout shuffle(self):
        cdef cnp.ndarray[float32_t, ndim=1] t2 = np.asarray(self.times).copy()
        shuffle(t2[0:self.var_size])
        return PrintLayout(self.grid, t2)

    def eval_landscape(self, int iterations):
        cdef int i = 0
        cdef PrintLayout pl
        cdef float32_t [:] output = np.empty(iterations, np.float32)
        for i in range(iterations):
            pl = self.shuffle()
            output[i] = np.nanmin(pl.compressed_distance_matrix())
        return output

    cdef bint meets_threshold(self, float32_t threshold):
        cdef:
            int i, j
            float32_t distance, time
            float32_t [:] d1
            float32_t [:] times = self.times
            float32_t [:,:] distance_matrix = self.grid.distance_matrix
            PrintGrid grid = self.grid

        for i in range(self.var_size):
            time = times[i]
            d1 = distance_matrix[i]
            for j in range(i+1, grid.size):
                if (d1[j] * abs(times[j] - time)) < threshold:
                    return False
        return True

    cdef bint beats_worst(self, PrintLayout pl):
        cdef float32_t self_min = np.nanargmin(self.compressed_distance_matrix())
        cdef float32_t pl_min = np.nanargmin(pl.compressed_distance_matrix())
        if self_min > pl_min:
            return True
        if self_min < pl_min:
            return False

        cdef:
            float32_t [:] a_order = np.sort(self.compressed_distance_matrix())
            float32_t [:] b_order = np.sort(pl.compressed_distance_matrix())
            int i
            float32_t a, b
        for i in range(self.var_size):
            a = a_order[i]
            b = b_order[i]
            if a>b: return True
            if a<b: return False
        return False


def random_initialize(PrintLayout pl, int iterations, int swap_iterations, float32_t threshold=0):
    cdef:
        PrintLayout testing
        PrintLayout best = pl
        float32_t[:, :] distance_matrix = pl.grid.distance_matrix
        float32_t [:] times = pl.times
        npy_intp i, j, k

    for i in range(iterations):
        testing = pl.shuffle()
        if testing.meets_threshold(threshold):
            testing = iterative_swap(testing, swap_iterations)
            if testing.beats_worst(best):
                print("Improved")
                best = testing
                #threshold = np.nanmin(best.compressed_distance_matrix())
    return best

def random_initialize_test(PrintLayout pl, int iterations, int swap_iterations, float32_t threshold=0):
    cdef:
        PrintLayout testing
        PrintLayout best = pl
        float32_t[:, :] distance_matrix = pl.grid.distance_matrix
        float32_t [:] times = pl.times
        npy_intp i, j, k

    output = np.empty(iterations, dtype=np.float32)
    for i in range(iterations):
        testing = pl.shuffle()
        if testing.meets_threshold(threshold):
            testing = iterative_swap(testing, swap_iterations)
            # if testing.beats_worst(best):
            #     best = testing
                #threshold = np.nanmin(best.compressed_distance_matrix())
        output[i] = np.nanmin(testing.compressed_distance_matrix())
    return output


cpdef iterative_swap(PrintLayout pl, int iterations):
    cdef:
        int var_size = pl.var_size
        int total_size = pl.grid.size
        int i, j, k
        npy_intp worst_index, worst_compressed_index
        float32_t worst_distance, worst_time

    times = np.asarray(pl.times).copy()
    vec = np.triu_indices(pl.grid.size)[0]
    vec2 = np.triu_indices(pl.grid.size)[1]
    euclid = pl.grid.distance_matrix
    for i in range(iterations):
        pl2 = swap_search(pl)
        if np.array_equal(pl.times, pl2.times, equal_nan=True):
            return pl
        pl = pl2
    return pl

cdef PrintLayout swap_search(PrintLayout pl):
    cdef:
        float32_t[:] d1
        float32_t[:] times = pl.times.copy()
        float32_t[:,:] euclid = pl.grid.distance_matrix
        int worst_compressed_index = np.nanargmin(pl.compressed_distance_matrix())
        float32_t worst_distance = pl.compressed_distance_matrix()[worst_compressed_index]
        npy_intp worst_index = pl.grid.compressed_row[worst_compressed_index]
        npy_intp worst_pair = pl.grid.compressed_col[worst_compressed_index]
        float32_t worst_time = times[worst_index]
        int i

    worst_index_distances = euclid[worst_index,:]
    coor = np.asarray(pl.grid.coordinates)

    for i in range(pl.var_size):
        if worst_index == i:
            continue
        test_time = times[i]
        times[worst_index] = test_time
        times[i] = worst_time

        d1 = euclid[i,:] * np.abs(np.asarray(times) - worst_time)
        if np.nanmin(d1) <= worst_distance:
            times[worst_index] = worst_time
            times[i] = test_time
            continue
        if np.isnan(test_time):
            return PrintLayout(pl.grid, times)
        d1 = worst_index_distances * np.abs(np.asarray(times) - test_time)
        if np.nanmin(d1) <= worst_distance:
            times[worst_index] = worst_time
            times[i] = test_time
            continue
        return PrintLayout(pl.grid, times)

    t = worst_index
    worst_index = worst_pair
    worst_pair = t
    if worst_index >= pl.var_size:
        return pl
    worst_time = times[worst_index]
    worst_index_distances = euclid[worst_index,:]

    for i in range(pl.var_size):
        if worst_index == i:
            continue
        test_time = times[i]
        times[worst_index] = test_time
        times[i] = worst_time

        d1 = euclid[i,:] * np.abs(np.asarray(times) - worst_time)
        if np.nanmin(d1) <= worst_distance:
            times[worst_index] = worst_time
            times[i] = test_time
            continue
        if np.isnan(test_time):
            return PrintLayout(pl.grid, times)
        d1 = worst_index_distances * np.abs(np.asarray(times) - test_time)
        if np.nanmin(d1) <= worst_distance:
            times[worst_index] = worst_time
            times[i] = test_time
            continue
        return PrintLayout(pl.grid, times)
    return pl

cdef exceeds_distance(npy_intp idx, float32_t [:] arr, float32_t [:] times, float32_t value):
    cdef:
        float32_t dist = arr[idx]
        float32_t time = times[idx]
        int i

    for i in range(arr.size):
        if i==idx: continue
        pass
