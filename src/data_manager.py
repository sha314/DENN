import imports
import tensorflow as tf


class DataManager_v2:
    def __init__(self, lower_bound, upper_bound, dtype=tf.float32):
        """
        lower_bound : lower bounds. [x1i, x2i, x3i, ...]. initial value
        upper_bound : upper bounds. [x1f, x2f, x3f, ...]. final value
        dtype       : data type. tensorflow is data type sensetive.
        """
        self.dtype = dtype
        self.lower_bound = tf.cast(lower_bound, dtype=self.dtype)
        self.upper_bound = tf.cast(upper_bound, dtype=self.dtype)
        # print(self.lower_bound)
        pass

    def generate_random(self, N):
        """
        N : number of data points to be generated
        """
        d = tf.random.uniform((N, self.lower_bound.shape[0]), self.lower_bound, self.upper_bound)
        return d

    def generate_uniform(self, N):
        """
        N : number of unique data points along each direction
        """
        dd = []
        for i in range(self.lower_bound.shape[0]):
            d = tf.linspace(self.lower_bound[i], self.upper_bound[i], N)
            dd.append(d)
            pass

        xx = tf.meshgrid(*[a for a in dd])
        # print(len(xx))
        # print(xx[0].shape)

        X = tf.concat([tf.reshape(a, (-1, 1)) for a in xx], axis=1)
        # print(X.shape)
        return X

    @staticmethod
    def generate_random_s(lower_bound, upper_bound, N, dtype=tf.float32):
        a = DataManager_v2(lower_bound, upper_bound, dtype)
        return a.generate_random(N)

    @staticmethod
    def generate_uniform_s(lower_bound, upper_bound, N, dtype=tf.float32):
        print("lower_bound ", lower_bound, ". upper_bound ", upper_bound)
        a = DataManager_v2(lower_bound, upper_bound, dtype)
        return a.generate_random(N)
        pass

    # def generate_dense(self, N, ranges, densities=None):
    #     """
    #     generate data randomly from given range respecting the density of data in that range.
    #     N         : int. total number of data points.
    #     ranges    : list of range. nested list. [[a, b], [b, c]]
    #     densities : None, list or float.
    #                     if `None` then all data is generated from given ranges.
    #                     if a list of density of points in given ranges or . length of `densities` must be the same as `ranges`.
    #                 sum of all elements of densities must never exceed unity.
    #                     if float then that percentage of data will be generated from ranges and other data is from random selection.
    #     """
    #     data_count = tf.constant(densities) * N
    #     data_count = tf.cast(data_count, dtype=tf.int32)
    #     ranges = tf.cast(ranges, dtype=self.dtype)
    #     Ng = np.sum(data_count)
    #     if Ng > N:
    #         data_count /= N
    #         print("generate_dense")
    #         # print(data_count)
    #     all_data = []
    #     if N > Ng:
    #         # other points will be chosen randomly from lower and upper bound. which was set during construction
    #         rest = self.generate_random(N - Ng)
    #         all_data.append(rest)
    #         pass
    #
    #     # print(data_count.shape)
    #     if len(data_count.shape) == 0:  # scalar tensor
    #         data_count = tf.cast(data_count, tf.float32)
    #         a = data_count / len(ranges)
    #         # print(a)
    #         if a == 0:
    #             a = 1
    #             pass
    #         data_count = tf.ones(len(ranges)) * a
    #         data_count = tf.cast(data_count, tf.int32)
    #         # print(data_count)
    #         pass
    #
    #     for rng, Nd in zip(ranges, data_count):
    #         a = DataManager_v2.generate_random_s(rng[0], rng[1], Nd, self.dtype)
    #         all_data.append(a)
    #         pass
    #     d = tf.concat(all_data, axis=0)
    #     return d

    def scale_data(self, X):
        """
        scaling data before feeding to neural net. after scaling data spans [-1,1]
        X  : input data
        lb : lower bounds
        up : upper bounds
        """
        lb = self.lower_bound
        ub = self.upper_bound
        H = 2.0 * (X - lb) / (ub - lb) - 1.0
        return H

    def get_ranges_from_loss_X(self, X, loss, tolerance=None):
        """
        for multidimensional data.

        X      : input data
        loss   : loss for input data `X`
        return : a list of range. first element of the list is the lower bounds and
                second element is the upper bounds
        """

        N = X.shape[0]
        #         print(X.shape)
        loss = loss / np.max(loss)
        if tolerance is None:
            tolerance = np.mean(loss)
            pass
        step = 0.2 * (self.upper_bound - self.lower_bound) / N
        mask = loss > tolerance
        #         print(mask.shape)
        # mask = np.reshape(mask, (-1,1))
        Xm = X[tf.reshape(mask, (-1,))]  # only select rows not columns
        #         print(step.shape)
        #         print(Xm.shape)
        ranges = [Xm - step, Xm + step]
        return ranges

    def get_data_from_loss_X(self, X, loss, tolerance=None):
        """
        for multidimensional data. If called multiple times with certain interval, you will have more data
        in the region where loss is higher than tolerance.

        X         : input data. usually uniformly distributed over the domain of inpur variables.
        loss      : loss for input data `X`
        tolerance : default None. then average of normalized loss is used a tolerance.
        return    : data points where loss is higher than tolerance.
        """
        loss /= np.sum(loss) # normalized
        N = X.shape[0]
        if tolerance is None:
            tolerance = np.mean(loss)
            pass
        mask = loss > tolerance
        #         print(mask.shape)
        # mask = np.reshape(mask, (-1,1))
        Xm = X[tf.reshape(mask, (-1,))]  # only select rows not columns
        #         print(step.shape)
        print(Xm.shape)

        return Xm

    @staticmethod
    def get_ranges_from_loss_X_1D(lb, ub, X, loss, tolerance=None):
        """
        for 1D data, i.e., one variable data only
        lb        : lower bound
        ub        : upper bound
        X         : input data
        loss      : loss due to `X` as input. converted to loss -> loss/max(loss)
        tolerance : default None. in that case mean loss is used.
                    if loss is greater than tolerance then more data in that range is generated
        """
        # TODO : upgrade this program so that it works for multi-dimensional problem, i.e.,
        # more than one variable in picture
        N = X.shape[0]
        loss = loss / np.max(loss)
        if tolerance is None:
            tolerance = np.mean(loss)
            pass
        step = (ub - lb) / N
        mask = loss > tolerance
        Xm = X[mask]

        Xm_sorted = tf.sort(Xm)
        print(Xm_sorted.shape)
        X_diff = np.diff(Xm_sorted)
        #     print(mu)
        #     print(X_diff)
        idx = X_diff > step
        indices = np.arange(0, X_diff.shape[0], 1, dtype=int)
        #     print(indices)
        #     print(indices.shape)
        #     print(idx.shape)
        idx2 = indices[idx]
        idx2 = tf.reshape(idx2, (-1,))
        # Xm_sorted = tf.reshape(Xm_sorted, (-1,1))
        # print(Xm_sorted)
        # print(Xm_sorted[idx2])
        ranges = []
        for i in idx2:
            ranges.append([Xm_sorted[i].numpy(), Xm_sorted[i + 1].numpy()])
        # for rng in zip(Xm_sorted[idx2], Xm_sorted[idx2 + 1]):
        #     ranges.append(rng)
        #             print(rng)
        return tf.constant(ranges)
