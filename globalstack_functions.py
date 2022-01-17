import numpy as np
from numba import jit
from numba import int64, float64
import math

from numba.experimental import jitclass

spec2 = [
    ('__nn', int64),
    ('__numel', int64),
    ('__u', int64),
    ('__uu', int64),
    ('__ul', int64),
    ('__indices', int64[:]),
    ('__z', float64[:])
]


@jitclass(spec2)
class pq:
    """
    A class for jit enables priority queue.  This serves a fairly specific role at the moment in that
    it allows for an elevation vector (z) to be added.  However it remains unsorted until all of its indices are pushed onto the queue.
    Additionally, the returned values are only the indices.
    For example if z = [3,2,3], add each element sequetially to the vector for i in range(len(z)): pq = pq.push(i) will sort to
    2,3,3.  But then pa.top() will return (1) instead of (2) in order to be used in the Barnes (2014) algorithm.
    In the future it should be more generalized.

    """

    def __init__(self, z):
        """
        initiate all values to 0

        :nn: The indices of z
        :numel: number of elements currently in the queue
        :u:

        """
        self.__nn = np.int64(len(z))
        self.__numel = np.int64(0)
        self.__u = np.int64(0)  # End of the right side of the queue
        self.__uu = np.int64(0)  # Top value of the queue
        self.__ul = np.int64(0)  # End of the left side of the queue
        self.__indices = np.full(len(z) + 1, 0)  # This is the main vector containing indices to be sorted
        self.__z = np.concatenate((np.zeros(1).ravel(), z.ravel()))  # contains the values to be sorted upward

    def top(self):
        """
        Get the top value of the queue (lowest value)

        :return:  self
        """
        return self.__indices[1] - 1

    def get(self):
        """
        Get the ordered z values, not necessarily perfectly sorted due to the nature of pq

        :return: ordered z values (lowest to highest)
        """
        return self.__z[self.__indices]

    # @property
    def pop(self):
        """
        Pop lowest value off the queue and re-sort

        :return: self
        """
        self.__uu = self.__indices[1]  # Absolute Top value of the queue
        self.__indices[1] = self.__indices[self.__numel]  # Move the last value to the top and re-sort
        self.__indices[self.__numel] = 0
        self.__u = 2  # End of right hand side (initially we just have 2 sides with 1 element each)
        self.__ul = np.int(self.__u / 2)  # end of left hand side
        while self.__u < self.__numel - 2:
            # Is the end of the current right side less than the end of the next left side? If so, we stay with the current set of sides
            if self.__z[self.__indices[self.__u]] <= self.__z[self.__indices[self.__u + 1]]:
                # If right side is greater than the left side, flip values and move onto the next set of sides
                if self.__z[self.__indices[self.__ul]] >= self.__z[self.__indices[self.__u]]:
                    t = self.__indices[self.__ul]
                    self.__indices[self.__ul] = self.__indices[self.__u]
                    self.__indices[self.__u] = t

                    self.__ul = self.__u
                    self.__u *= 2
                else:
                    break
            # If end of the right side is greater than the next set of left sides, flip values and move onto the next set of sides
            elif self.__z[self.__indices[self.__ul]] > self.__z[self.__indices[self.__u + 1]]:

                t = self.__indices[self.__ul]
                self.__indices[self.__ul] = self.__indices[self.__u + 1]
                self.__indices[self.__u + 1] = t
                self.__u = 2 * (self.__u + 1)
                self.__ul = np.int(self.__u / 2)

            else:

                break

        self.__numel -= 1
        return self

    def push(self, i):
        """
        Push a value onto the queue (and sort)

        :param i: value to add
        :return: self
        """

        i += 1
        self.__numel += 1

        self.__u = self.__numel  # The end of the right side of the queue
        self.__ul = np.int(self.__u / 2)  # The end of the left side of the queue

        self.__indices[self.__u] = i  # initially add index to the end of the right-hand side
        while self.__ul > 0:
            # If end left is greater than end right, switch end left and end right.
            if self.__z[self.__indices[self.__ul]] >= self.__z[self.__indices[self.__u]]:

                t = self.__indices[self.__ul]
                self.__indices[self.__ul] = self.__indices[self.__u]
                self.__indices[self.__u] = t

            else:
                break
            # Now break up the current left hand side into new halves, repeat).
            self.__u = np.int(self.__u / 2)
            self.__ul = np.int(self.__u / 2)
        return self


#
@jit(nopython=True)
def sinkfill(Z):
    """
    Fill pits using the priority flood method of Barnes et al., 2014.

    :param Z: Input elevation grid
    :return: output elevation, pit filled
    """

    c = int(0)
    ny, nx = np.shape(Z)
    nn = ny * nx
    p = int(0)
    closed = np.full(nn, False)
    idx = [1, -1, ny, -ny, -ny + 1, -ny - 1, ny + 1, ny - 1]  # Linear indices of neighbors
    open = pq(Z.transpose().flatten())

    for i in range(0, ny):  ##
        for j in range(0, nx):
            if (i == 0) or (j == 0) or (j == nx - 1) or (i == nx - 1) or (Z[i, j] <= 0):
                # In this case only edge cells, and those below sea level (base level) are added
                ij = j * ny + i
                if not closed[ij]:
                    closed[ij] = True
                    open = open.push(ij)
                    c += 1

    pit = np.zeros(nn)
    pittop = int(-9999)
    while (c > 0) or (p > 0):

        if (p > 0) and (c > 0) and (pit[p - 1] == -9999):
            s = open.top()
            open = open.pop()  # The pq class (above) has seperate methods for pop and top, although (others may combine both functions)
            c -= 1
            pittop = -9999
        elif p > 0:
            s = int(pit[p - 1])
            pit[p - 1] = -9999
            p -= 1
            if pittop == -9999:
                si, sj = lind(s, ny)
                pittop = Z[si, sj]
        else:
            s = int(open.top())
            open = open.pop()
            c -= 1
            pittop = -9999

        for i in range(8):

            ij = idx[i] + s
            si, sj = lind(s, ny) #Current
            ii, jj = lind(ij, ny) #Neighbor

            if (ii >= 0) and (jj >= 0) and (ii < ny) and (jj < nx) and not closed[ij]:
                closed[ij] = True

                if Z[ii, jj] <= Z[si, sj]:

                    Z[ii, jj] = Z[si, sj] + 1e-8  # This (e) is sufficiently small for most DEMs but it's not the lowest possible.  In case we are using 32 bit, I keep it here.

                    pit[p] = ij

                    p += 1
                else:
                    open = open.push(ij)
                    c += 1
            if np.mod(ij, 1e8) == 0:
                print(ij)
    return Z


def zero_edges(y):
    """
    Trivial but widely used function to zero out edges of a matrix

    :param y: input matrix
    "returns: Matrix with edges zeroed out
    """
    y[:, 0] = 0
    y[0, :] = 0
    y[:, -1] = 0
    y[-1, :] = 0

    return y


@jit(nopython=True)
def h_flowdir(d):
    """
    Translate flow dir from hydrosheds into flow dir from simplem format.

    :param d: flow direction grid from hydrosheds
    :return: sx and sy, the flow direction in x and y directions ( i.e. -1, 0 , 1)
    """
    ny, nx = np.shape(d)
    sy = np.zeros(np.shape(d), dtype=np.int8)  # slopes formatted correctly for simplem
    sx = np.zeros(np.shape(d), dtype=np.int8)

    for i in range(ny):
        for j in range(nx):
            d1 = d[i, j]
            if (d1 == 0) or (d1 == 255):  # 0 is outlet to ocean, 255 is internally drained pour point
                sx[i, j] = 0
                sy[i, j] = 0
            elif d1 == 1:  # East
                sx[i, j] = 1
                sy[i, j] = 0
            elif d1 == 2:  # SE
                sx[i, j] = 1
                sy[i, j] = 1
            elif d1 == 4:  # S
                sx[i, j] = 0
                sy[i, j] = 1
            elif d1 == 8:  # SW
                sx[i, j] = -1
                sy[i, j] = 1
            elif d1 == 16:  # W
                sx[i, j] = -1
                sy[i, j] = 0
            elif d1 == 32:  # NW
                sx[i, j] = -1
                sy[i, j] = -1
            elif d1 == 64:  # N
                sx[i, j] = 0
                sy[i, j] = -1
            elif d1 == 128:  # NE
                sx[i, j] = 1
                sy[i, j] = -1

    return sx, sy


@jit(nopython=True)
def lind(xy, n):
    """
    compute bilinear index from linear indices - trivial but widely used (hence the separate function)

    :param xy:  linear index
    :param n: ny or nx (depending on row-major or col-major indexing)
    :return:
    """
    # Compute linear index from 2 points
    x = math.floor(xy / n)  # type: float
    y = xy % n
    return int(y), int(x)


@jit(nopython=True)
def stack_bilinear(sx, sy):
    """
    takes the input flordirs sx sy and makes the topologically ordered
     stack of the stream network in O(n) time.  This is a slightly different approach from the
     Fastscape algorithm which uses a recursive function - instead this sues a while loop, which is more efficient.
     The bilinear indices use less memory than the linear indices.  In memory limited systems, that is important, and that is why it is done here

    :param sx: x flow direction grid
    :param sy: y flow direction grid
    :return: topologically ordered stack, Ix and Iy indices
    """

    c = 0
    k = 0
    ny, nx = np.shape(sx)
    Ix = np.zeros(ny * nx, dtype=np.uint32)
    Iy = np.zeros(ny * nx, dtype=np.uint32)
    for i in range(ny):
        for j in range(nx):

            ij = j * ny + i
            i2 = i
            j2 = j
            if sx[i, j] == 0 and sy[i, j] == 0:  # if we find a sink, begin searching upstream

                Ix[c] = int(ij / ny)
                Iy[c] = int(ij % ny)
                c += 1

                while c > k and c < ny * nx - 1:  # While we have unexplored members on the stack
                    for i1 in range(-1, 2):
                        for j1 in range(-1, 2):
                            if j2 + j1 > 0 and i2 + i1 > 0 and j2 + j1 < nx - 1 and i2 + i1 < ny - 1:
                                ij2 = (j2 + j1) * ny + i2 + i1
                                # print(s[i2 + i1, j2 + j1])

                                if (i1 != 0 or j1 != 0) and sy[int(i2 + i1),
                                                               int(j2 + j1)] + i1 == 0 and sx[int(i2 + i1),
                                                                                              int(j2 + j1)] + j1 == 0:  # Is each upstream neighbor a donor?
                                    # I[c] = ij2
                                    Ix[c] = int(ij2 / ny)
                                    Iy[c] = int(ij2 % ny)
                                    c += 1

                    k = k + 1
                    ij = Ix[k] * ny + Iy[k]
                    i2, j2 = lind(ij, ny)
        if np.mod(i, 1000) == 0:
            print('##')
            print(c)
            print(i / ny)
    return Ix, Iy


## normal
@jit(nopython=True)
def acc(sx, sy, Ix, Iy, size):
    """
    Takes the stack and receiver grids and computes drainage area.

    :param sx: x flow direction grid
    :param sy: y flow direction grid
    :param Ix: x flow direction grid
    :param Iy: y flow direction grid
    :return: Drainage area grid
    """

    A = np.ones(size, dtype=np.uint32)
    ny, nx = size
    c = 0
    # Drainage area calculation is simply a downstream sum - i.e. sum in the reverse order of the stack
    for ij in range(len(Ix) - 1, -1, -1):
        j = Ix[ij]
        i = Iy[ij]
        ij1 = i + j * ny

        j2 = int(sx[i, j]) + j
        i2 = int(sy[i, j]) + i

        if sy[i, j] != 0 or sx[i, j] != 0:
            A[i2, j2] += A[i, j]

        if ij % 1000000 == 0:
            print(ij / len(Ix))
            c += 1

            #
    return A


@jit(nopython=True)
def slp(Z):
    """
    Calculates the flow direction receiver grid of a DEM based on the D8 algorithm

    :return: Flow direction receiver grid values sy and sx ( both D8 directions -1,0,1)
    """
    ij = 0
    c = 0
    sx = np.zeros(np.shape(Z), dtype=np.int8)
    sy = np.zeros(np.shape(Z), dtype=np.int8)
    ny, nx = np.shape(Z)
    for i in range(0, ny):
        for j in range(0, nx):
            mxi = 0  # Max slope between cell and all neighbors
            sx[i, j] = 0
            sy[i, j] = 0
            if (i > 0 and i < ny and j > 0 and j < nx - 1 and i < ny - 1):
                for i1 in range(-1, 2):
                    for j1 in range(-1, 2):
                        if (Z[i, j] - Z[i + i1, j + j1]) / np.sqrt((float(i1) ** 2) + float(j1) ** 2 + 1e-10) > mxi:
                            mxi = (Z[i, j] - Z[i + i1, j + j1]) / np.sqrt((float(i1) ** 2) + float(j1) ** 2) # If the max slope is less than current... we deem this neighbor the new max
                            sx[i, j] = j1
                            sy[i, j] = i1
                if mxi == 0: # keeping track of number of sinks, for debugging purposes
                    c += 1
                    # fnd[i,j] = 1
        if np.mod(i, 1000) == 0:
            print(i / ny)
    print(c)
    return sx, sy
