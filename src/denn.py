import tensorflow as tf
import data_manager
import neural_net

DataManager = data_manager.DataManager_v2
NeuralNet = neural_net.NeuralNet_v3

import numpy as np

import time

import imports


class APL_v11:
    """
    Uses tensorflow optimizer.
    Built for flexibility.

    Model extended from this class can be saved as a pickle file and
    can be loaded from that file. only importnat information are saved.

    For example : weights, biases, layers, lower and upper bounds, variable order,
                 last history, date, time, classname, etc.
    """

    def __init__(self, layers, learning_rate=0.001):
        """
        layers : layer configuration list
        eta    : learning rate
        """
        self.nnet = NeuralNet(layers, learning_rate)  # neural net

        self.history = []
        self.epoch = 0
        self.classname = "APL_v11"
        self.var_order = None  # order of variables
        self.ext = None  # external parameter
        self.batch_size = 32
        self.X_batch = None
        self.debug = False
        self.d_manager = None
        self.steep_density = 0.3  # 30% data comes from steep region
        self.dtype = np.float32
        pass

    @staticmethod
    def load_from_file(filename):
        print("TODO. load models from file directly without create an object of this class.")
        pass

    def schedule_learning_rate(self, decay_steps, decay_rate):
        self.nnet.schedule_learning_rate(decay_steps, decay_rate)
        pass

    def current_learning_rate(self):
        return self.nnet.current_learning_rate(self.epoch)

    def loss_function(self):
        """
        takes no argument.
        for passing to tensorflow optimizer
        """
        print("Must be implemented by the subclasses")
        return tf.constant(0.)

    def pde_loss_vector(self, X):
        """
        takes the same input as predict method.
        returns the loss vector for each data point of X
        """
        print("Must be implemented by the subclasses")
        return X * tf.constant(0.)

    def pde_loss(self):
        return tf.constant(0.)

    def boundary_loss(self):
        return tf.constant(0.)

    def set_learning_rate(self, learning_rate):
        self.nnet.set_learning_rate(learning_rate)

    def rechedule_learning_rate(self, new_learning_rate, decay_steps, decay_rate):
        self.nnet.rechedule_learning_rate(new_learning_rate, decay_steps, decay_rate)

    def set_data_manager(self, d_manager):
        """
        d_manager : a DataManager object
        """
        self.d_manager = d_manager  # data manager

    def set_steep_density(self, density):
        self.steep_density = density
        pass

    def set_neural_net(self, nnet):
        """
        nnet :  a NeuralNet_v3 object
        """
        self.nnet = nnet

    def set_mode(self, debug=False):
        """
        if debug mode is True then some extra messages will be printed
        """
        self.debug = debug

    def get_classname(self):
        return self.classname

    def set_variable_order(self, order):
        """
        for multivariable case this is important.
        in which order the variable appear in the columns
        """
        self.var_order = order
        print("variable order ", self.var_order)

    def set_ext_param(self, ext_const={}, ext_var={}):
        """

        ext_const : external parameter that must remain constant throughout the training
        ext_var   : external parameter that must change constantly while training

        """
        pass

    def get_dict(self):
        """
        returns the following data as dict
                1. history. last element
                2. iteration
                3. date
                4. time
                5. classname
                6. lower boundary
                7. upper boundary
                8. external parameter
                """
        dct = self.nnet.get_dict()
        dct['history'] = self.history[-1]
        dct['iteration'] = self.epoch
        dct['epoch'] = self.epoch
        today = datetime.datetime.today()
        dct['date'] = today.strftime('%Y.%m.%d')
        dct['time'] = today.strftime('%H:%M:%S')
        dct['classname'] = self.get_classname()
        dct['lower_boundary'] = self.lower_boundary
        dct['upper_boundary'] = self.upper_boundary
        dct['var_order'] = self.var_order
        dct['ext_params'] = self.ext

        return dct

    def save(self, filename, extension='.pkl'):
        """
        saving model.


        filename    : full name of the file that the model will be saved
        extension   : .pkl for pickle extension
        """

        dct = self.get_dict()
        f_name = filename + extension
        with open(f_name, 'wb') as outfile:
            pickle.dump(dct, outfile)
        pass

        # f_name = filename + '.json'
        # dct2 = to_pure_python(dct)
        # print(dct2)
        # with open(f_name, 'w', encoding='utf-8') as outfile:
        #     json.dump(dct2, outfile)
        #     pass

    def load_weights(self, weights, biases):
        print("loading weights and biases directly")
        self.nnet.load_weights(weights, biases)
        pass

    def load(self, filename, extension='.pkl'):
        """
        loading model from file
        """
        filename = filename + extension
        with open(filename, 'rb') as file:
            dct = pickle.load(file)
        pass
        keys = dct.keys()
        # weights = dct['weights']
        # biases = dct['biases']
        self.epoch = dct['iteration']
        if "epoch" not in keys:
            dct['epoch'] = self.epoch
        self.nnet.load_dct(dct)
        # self.nnet.layers = dct['layers']
        # self.nnet.load_weights(weights, biases)
        self.history.append(dct['history'])
        self.lower_boundary = dct['lower_boundary']
        self.upper_boundary = dct['upper_boundary']
        self.var_order = dct['var_order']

        # check if the current configuration is ok

        pass

    def set_dtype(self, dtype):
        self.dtype = dtype

    def set_boundary(self, lower_boundary, upper_boundary):
        """
        :param lower_boundary: list of lower boundary. allowed data type is native list or numpy.ndarray
        :param upper_boundary: list of upper boundary. allowed data type is native list or numpy.ndarray
        :return:
        """
        tpe = type(lower_boundary)
        if tpe == np.ndarray:
            self.lower_boundary = lower_boundary.astype(dtype=self.dtype)
            self.upper_boundary = upper_boundary.astype(dtype=self.dtype)
        elif tpe == list:
            self.lower_boundary = np.array(lower_boundary, self.dtype)
            self.upper_boundary = np.array(upper_boundary, self.dtype)
            pass
        else:
            print("data type must be a list or numpy.ndarray")
            return
        self.ub = tf.constant(self.upper_boundary)  # tensorflow object
        self.lb = tf.constant(self.lower_boundary)  # tensorflow object
        pass

    @tf.function
    def scale_data(self, X):
        """
        scaling data before feeding to neural net. this function must be called internally by `APL.predict` method
        :param X: input data.
        :return:
        """
        ## lb : lower bounds
        ## up : upper bounds
        # lb = self.lower_boundary
        # ub = self.upper_boundary
        # H = 2.0*(X - lb)/(ub - lb) - 1.0
        H = 2.0 * (X - self.lb)
        H /= (self.ub - self.lb)
        H -= 1.0
        return H

    @tf.function
    def predict(self, X):
        """
        predicted method. rescale the input in range[-1,1] and feed it to the input of the neural net.
        Convention :- new row means new data point and new column means new feature.

        :param X: input data. must be a `tf.tensor`.
        :return: evaluated result by the ANN
        """

        X = self.scale_data(X)
        return self.nnet.evaluate(X)

    @staticmethod
    def get_elapsed_time(start_time, formating="%H:%M:%S"):
        elapsed_time = time.time() - start_time
        return time.strftime(formating, time.gmtime(elapsed_time))

    def rar_deep_xde(self, epsilon=1e-3):
        """
        inspired from DEEPXDE paper. will probably be enabled in APL11.

        sometimes solution has steeper region and more data is needed to learn
        it. once every interval this method will be called if appropriate argument
        is provided to accomplish this task.
        since only pde loss term depends on the data not on the boundary.

        returns : ranges where more data is needed.
        """

        loss = self.pde_loss(True)
        print(epsilon)
        mask = loss > epsilon
        mask = tf.reshape(mask, (-1,))
        # print(mask.shape)
        # print(self.X.shape)
        denser = tf.boolean_mask(self.X, mask)
        if self.debug:
            print("rar data shape ", denser.shape)
            pass
        return denser

    def get_data_steep(self):
        """
        get more data where error is higher
        steep_density probability to get data from dense region.
                     dense means the area where error is high. TODO set it from subclass.
        """
        # print("get_data_steep")
        N = self.X.shape[0]
        X = self.d_manager.generate_uniform(N)
        X_var = tf.Variable(X)
        loss = self.pde_loss_vector(X_var)

        ranges = self.d_manager.get_ranges_from_loss_X(X, loss, None)
        X_dense = self.d_manager.generate_dense(N, ranges, self.steep_density)
        return X_dense

    def train(self, epoch, interval=100, batch_size=32, generate_data=False, rar=False):
        """

        epoch      : number of iteration
        interval   : interval of recording history and calling 'set_data_external' method
        batch_size : number of data to train at each iteration or interval?. default is 0, means take whole data set
        generate_data :
        rar        : DEEPXDE paper. enabled in APL11
        """
        # self.train_v1a(epoch, interval, batch_size, generate_data, rar)
        self.train_v1b(epoch, interval, batch_size, generate_data, rar)
        # self.train_v2(epoch, interval, batch_size, generate_data, rar)

    def train_v1a(self, epoch, interval=100, batch_size=32, generate_data=False, rar=False):
        start_time = time.time()
        interval_time = start_time
        if batch_size > 32 or batch_size != self.batch_size:
            print("changing batch size to ", batch_size," before training")
            self.batch_size = batch_size
            X_batch = self.get_batch_data()  # for batch training.
            self.X_batch = tf.Variable(X_batch,
                                       name="X_batch")  # for new batch size assign will not work. so resetting X_batch
            pass

        if self.d_manager is None:
            print("set up data manager first")
            return

        for e in range(epoch):

            if e % interval == 0:
                interval_time = self.after_each_interval(e, generate_data, interval_time, rar)

            # self.optimizer.minimize(self.loss_function, var_list=self.nnet.weights)
            # X_batch = self.get_batch_data()  # for batch training.
            X_batch = self.get_batch_data_v2()  # for batch training.
            self.X_batch.assign(X_batch)
            self.nnet.minimize(self.loss_function)
            self.epoch += 1
            pass

        self.history.append([self.epoch, self.loss_function().numpy()])
        # elapsed_time = time.time() - start_time
        print("Elapsed time, ", self.get_elapsed_time(start_time))
        pass

    def train_v1b(self, epoch, interval=100, batch_size=32, generate_data=False, rar=False):
        """
        Will be highly considered in APL_v12
        epoch      : number of iteration
        interval   : interval of recording history and calling 'set_data_external' method
        batch_size : number of data to train at each iteration or interval?. default is 0, means take whole data set
        generate_data :
        rar        : DEEPXDE paper. enabled in APL11
        """
        print("Warning : for this to work loss_function and everything inside it must be tf.function decorated")
        start_time = time.time()
        interval_time = start_time
        if batch_size > 32 or batch_size != self.batch_size:
            print("changing batch size to ", batch_size," before training")
            self.batch_size = batch_size
            X_batch = self.get_batch_data()  # for batch training.
            self.X_batch = tf.Variable(X_batch, name="X_batch") # for new batch size assign will not work. so resetting X_batch
            pass

        if self.d_manager is None:
            print("set up data manager first")
            return
        num_interval = epoch // interval
        print(num_interval)
        for e in tf.range(num_interval):
            self.one_interval(interval)
            self.epoch += interval
            interval_time = self.after_each_interval(self.epoch, generate_data, interval_time, rar)

            # self.optimizer.minimize(self.loss_function, var_list=self.nnet.weights)
            # X_batch = self.get_batch_data()  # for batch training.


            pass

        # self.history.append([self.epoch, self.loss_function().numpy()])
        # elapsed_time = time.time() - start_time
        print("Elapsed time, ", self.get_elapsed_time(start_time))
        pass

    # @tf.function # it will be a tf function in future
    def one_interval(self, interval):
        # print("Warning : one_interval is not tf.function decorated")
        for i in tf.range(interval):
            # print("Tracing get_batch_data_v2")
            X_batch = self.get_batch_data_v2()  # for batch training.
            # print("Tracing assign")
            self.X_batch.assign(X_batch)
            # print("Tracing minimize")
            self.nnet.minimize(self.loss_function)
            pass

    def after_each_interval(self, e, generate_data, interval_time, rar):
        loss = self.loss_function().numpy()
        tf.print("Iteration ", e, ". Elapsed time ", APL_v11.get_elapsed_time(interval_time), " => loss ", loss)

        self.history.append([self.epoch, loss])
        if generate_data:
            self.set_data_external(rar)
            pass
        return time.time()

    def train_v2(self, epoch, interval=100, batch_size=32, generate_data=False, rar=False):
        """
        uses tensorflow data. will be made defult in APLv12 if proven efficient.
        epoch      : number of iteration
        interval   : interval of recording history and calling 'set_data_external' method
        batch_size : number of data to train at each iteration or interval?. default is 0, means take whole data set
        generate_data :
        rar        : DEEPXDE paper. enabled in APL11
        """
        start_time = time.time()
        interval_time = start_time
        if batch_size > 32:
            self.batch_size = batch_size
            pass

        if self.d_manager is None:
            print("set up data manager first")
            return

        # self.X_batch = self.get_batch_data()  # for batch training. TODO : replace it with data manager
        # print(self.X_batch)
        dataset = self.data_set.repeat(epoch)
        dataset = dataset.shuffle(self.data_set_size * epoch, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        for e in range(epoch):

            if e % interval == 0:
                interval_time = self.after_each_interval(e, generate_data, interval_time, rar)
                pass

            # self.optimizer.minimize(self.loss_function, var_list=self.nnet.weights)
            X_batch = next(iter(dataset)) # for batch training. TODO : replace it with data manager
            self.X_batch.assign(X_batch)
            self.nnet.minimize(self.loss_function)
            self.epoch += 1
            pass

        self.history.append([self.epoch, self.loss_function().numpy()])
        # elapsed_time = time.time() - start_time
        print("Elapsed time, ", self.get_elapsed_time(start_time))
        pass

    def get_history(self):
        return np.array(self.history)

    def iteration_count(self):
        # N = self.history[-1][0]
        # if N != self.epoch:
        #     return N + self.epoch
        return self.epoch

    def set_training_data_size(self, N_data_points):
        self.X = self.d_manager.generate_random(N_data_points)
        self.X_batch = tf.Variable(self.get_batch_data(), name='X_batch')

        # for v12
        X = self.d_manager.generate_random(N_data_points) # tf.data will randomize it
        self.data_set_size = X.shape[0]
        self.data_set = tf.data.Dataset.from_tensor_slices(X)
        # self.X = tf.Variable(X)
        # print(self.X.shape)
        pass

    # def set_boundary_data_points(self, X, y):
    #     self.X_boundary = X
    #     self.y_boundary = y
    #     pass

    def get_batch_data(self):
        """
        Useful for batch training.
        batch_size : batch size. default is 0 which represents whole data set
        return     : a batch of data randomly selected from full data_set self.X
        """
        # print("get_batch_data")

        # print("self.X.shape ", self.X.shape)
        idx = np.random.choice(self.X.shape[0], self.batch_size)
        X = tf.gather(self.X, idx)
        # pass
        # if self.denser is not None:
        #     X = tf.concat([X, self.denser], axis=0)
        # if self.debug:
        #     print("batch shape ", X.shape)
        # X = self.d_manager.generate_random(self.batch_size)
        # return tf.Variable(X) # creating new variable is expensive. try assigining to old one
        return X

    @tf.function
    def get_batch_data_v2(self):
        """
        will be default in APLv12
        Useful for batch training.
        batch_size : batch size. default is 0 which represents whole data set
        return     : a batch of data randomly selected from full data_set self.X
        """
        # print("get_batch_data")

        # print("self.X.shape ", self.X.shape)
        shape = (self.batch_size,)
        idx = tf.random.uniform(
            shape, minval=0, maxval=self.X.shape[0], dtype=tf.dtypes.int32, seed=None, name=None
        )
        # print(idx)
        X = tf.gather(self.X, idx)
        # pass
        # if self.denser is not None:
        #     X = tf.concat([X, self.denser], axis=0)
        # if self.debug:
        #     print("batch shape ", X.shape)
        # X = self.d_manager.generate_random(self.batch_size)
        # return tf.Variable(X) # creating new variable is expensive. try assigining to old one
        return X

    def set_data_external(self, rar=False):
        """
        must be overriden by sub classes.
        If data is to be obtained from and external source, this method might come handy
        """
        if rar:
            self.X = self.get_data_steep()
            return
        if self.d_manager is not None:
            self.X = self.d_manager.generate_random(self.X.shape[0])
        pass

    def set_external_parameter(self, ext):
        self.ext = ext
        print(self.ext)
        pass

    def info(self):
        self.nnet.info()
        print("Lower Boundary, lb=", self.lower_boundary)
        print("Lower Boundary, lb=", self.upper_boundary)

    def set_regularization_l2_param(self, wp, bp=0):
        """

        :param wp: hyper parameter for weights
        :param bp: hyper parameter for weights. if zero then both weights and biases have same hyper parameter value
        :return:
        """
        self.nnet.set_regularization_hyper_param(wp, bp)
        pass

    @tf.function
    def regularization_l2(self):
        return self.nnet.regularization_l2_loss()

    def accuracy_test(self, N=5000):
        print("pde loss and boundary loss and all other loss function for "
              "uniform point distribution must be implemented in the subclass")
        print("Super class might not give the accurate result")
        print("Pde loss on ", N, " data points")
        X = self.d_manager.generate_uniform(N)
        X = tf.Variable(X)
        vec = self.pde_loss_vector(X)
        pde_error = tf.sqrt(tf.reduce_mean(tf.square(vec)))
        print("RMSE error is ", pde_error.numpy())
        print("Boundary loss ", self.boundary_loss())

    def set_activation_func(self, func, name):
        self.nnet.set_activation_func(func, name)
        pass
