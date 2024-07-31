
import imports
import data_manager
import tensorflow as tf


class NeuralNet_v3:
    """
    X1 = X0 W0 + b0
    multiplying from right. Weights goes with features not data points.

    Once you run the `evaluate` method, you cannot reassign weights and biases. If you
    do, expect an error. If this is the situation, try constructing a  object.

    Layer freezing is implemented
    """

    def __init__(self, layers=[1, 2, 1], learning_rate=0.001, filename=None):
        """
        layers  : layer configuration
        eta     : learning rate
        filename: default None. file name of saved NeuralNet without `.pkl` extension
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.optimizer_name = 'adam'
        self.num_layer = len(self.layers)
        if self.num_layer < 3:
            print("at least 3 layer is required")
            pass
        self.classname = "NeuralNet_v3"
        self.weights = []
        self.biases = []
        self.freeze_index = None  # for freezing layer
        if filename is not None:
            print("loading from file")
            self.load(filename)
            pass
        else:
            self.init_weights_biases()
            pass
        self.activation_func = tf.nn.swish
        self.activation_func_name = 'swish'
        self.schedule_name = None
        self.hyper_param_l2_loss = None
        pass

    def get_classname(self):
        return self.classname

    def __get_new(self, layers, weights, biases):
        a = NeuralNet_v3(layers)
        a.weights = weights
        a.biases = biases
        return a

    def set_learning_rate(self, learning_rate):
        self.schedule_name = None
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def schedule_learning_rate(self, decay_steps, decay_rate):
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate)
        # self.callbacks = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule, verbose=0)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.schedule_name='exponential_decay'
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        print("schedule ", self.schedule_name)

    def rechedule_learning_rate(self, new_learning_rate, decay_steps, decay_rate):
        print("Warning !!! changing learning rate might affect learning")
        self.learning_rate = new_learning_rate
        self.schedule_learning_rate(decay_steps, decay_rate)
        pass

    def set_optimizer(self, optimizer, name):
        self.optimizer = optimizer
        self.optimizer_name = name

    def set_activation_func(self, func, name):
        """

        :param func: tf.nn.* . activation function
        :param name: str. name of the activation function
        :return: None
        """
        self.activation_func = func
        self.activation_func_name = name

    def info(self):
        """
        Display information about the NeuralNet object
        """
        print("Optimizer     ", self.optimizer_name)
        print("Learning rate ", self.learning_rate)
        if self.schedule_name is not None:
            print("Learning Schedule ", self.schedule_name)
            print("decay_steps ", self.decay_steps)
            print("decay_rate ", self.decay_rate)
        print("Layers        ", self.layers)
        print("Number of hidden layers        ", len(self.layers)-2)
        print("Activation Function   ", self.activation_func_name)
        print("Weights and Biases")
        parameters = 0
        for i in range(len(self.weights)):
            w_shape = self.weights[i].shape
            b_shape = self.biases[i].shape
            parameters += w_shape[0] * w_shape[1] + b_shape[0] * b_shape[1]
            print("W({0})={1} , b({0})={2} ".format(i, w_shape, b_shape))
            pass
        print("Number of Parameters   ", parameters)
        if self.hyper_param_l2_loss is not None:
            print("Regularization Parameters   ", self.hyper_param_l2_loss)
            pass
        pass

    def init_weights_biases(self):
        for i in range(self.num_layer - 1):
            in_dim, out_dim = self.layers[i], self.layers[i + 1]
            w = self.weight_init_val_xavier(in_dim, out_dim)
            self.weights.append(tf.Variable(w, name="w{}".format(i)))

            b = tf.ones((1, out_dim))
            self.biases.append(tf.Variable(b, name="b{}".format(i)))
        pass

    def weight_init_val_xavier(self, in_dim, out_dim):
        """
        Y = W X
        xavier method starts with weights so that varience of `W` is 1,
        and it would mean `X` and `Y` have the same variance
        """
        xavier_stddev = tf.sqrt(2 / (in_dim + out_dim))
        #         print(xavier_stddev)
        w = tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev)
        #         print(w.shape)
        return w

    def set_regularization_hyper_param(self, wp, bp=0):
        """

        :param wp: hyper parameter for weights
        :param bp: hyper parameter for weights. if zero then both weights and biases have same hyper parameter value
        :return:
        """
        self.hyper_param_l2_loss = dict()
        self.hyper_param_l2_loss['w_hype'] = wp
        self.hyper_param_l2_loss['b_hype'] = bp if bp != 0 else wp
        pass

    @tf.function
    def regularization_l2_loss(self):
        """
        L2 regulatization
        :return: a \sum_i Wi + b \sum_i Bi
        """
        print("Tracing > regularization_l2_loss")
        if self.hyper_param_l2_loss is None:
            print("hyper_param_l2_loss not set")
            return 0.
        else:
            wa = tf.nn.l2_loss(self.weights[0])
            wb = tf.nn.l2_loss(self.biases[0])
            loop_count = len(self.weights)
            # print("loop_count ", loop_count)
            for i in range(1, loop_count):
                # print("got here")
                wa += tf.nn.l2_loss(self.weights[i])
                wb += tf.nn.l2_loss(self.biases[i])
                pass
            # print("summing complete")
            loss = wa*self.hyper_param_l2_loss['w_hype']
            loss += wb*self.hyper_param_l2_loss['b_hype']
            return loss
        pass

    @tf.function
    def evaluate(self, X):
        # print(type(X))
        # print(X.shape)
        # m, n = X.shape
        # print("data points ", m)
        # print("features    ", n)

        X0 = X
        # print(self.weights)
        # print(self.biases)
        for i in range(len(self.weights) - 1):
            W = self.weights[i]
            b = self.biases[i]
            # print(X0.shape)
            # print(W.shape)
            # print(b.shape)
            X0 = tf.matmul(X0, W) + b
            X0 = self.activation_func(X0)  # activation function

        W = self.weights[-1]
        b = self.biases[-1]
        # print(X0.shape)
        # print(W.shape)
        Y = tf.matmul(X0, W) + b
        return Y

    def evaluate_v2(self, X):
        # print(X.shape)
        # m, n = X.shape
        # print("data points ", m)
        # print("features    ", n)
        X0 = X
        for i in range(len(self.weights)):
            W = self.weights[i]
            b = self.biases[i]
            # print(X0.shape)
            # print(W.shape)
            X0 = tf.matmul(X0, W) + b
            X0 = self.activation[i](X0)  # different layer can have different activation function activation function
            pass
        return X0

    @tf.function
    def minimize(self, loss_function):
        """
        Weights are in this class. Optimizer is also in this class.
        Just providing a loss_function will get the work done

        loss_function : a function that takes no arguments and returns a value
                 computed using weights of this class.
        """
        if self.freeze_index is None:
            self.optimizer.minimize(loss_function, var_list=[self.weights, self.biases])
        else:
            self.optimizer.minimize(loss_function, var_list=[self.weights[self.freeze_index:],
                                                             self.biases[self.freeze_index:]])
            print("Need testing")
            pass

    def get_weights_biases(self):
        return [self.weights, self.biases]

    @staticmethod
    def join_nnet(self, nnet_a, layer_a, nnet_b, layer_b):
        """
        Joing two neural net object. by combining weights and biases according to specified layer_indices
        learning rate is kept the same as the self object. Only last few layers are modified generally since all layers in the begining
        are kinda trained for extract information.

        nnet_a  : first  NeuralNet object
        layer_a : indices of the layer for `nnet_a`?
        nnet_b  : second NeuralNet object
        layer_b : indices of the layer for `nnet_b`?
        returns : new NeuralNet object of given configuration
        """

        pass

    def split(self, left_layer_number):
        """
        TODO : needs testing

        split the neural net into two parts, i.e. left and right.

        left_layer_number : number of layers on the left side.
        return            : weight_L, weight_R

        """
        if left_layer_number >= len(self.layers):
            print("unfit splitting configuration")
            return
        weights_L = self.weights[:left_layer_number]
        biases_L = self.biases[:left_layer_number]
        weights_R = self.weights[left_layer_number:]
        biases_R = self.biases[left_layer_number:]
        return [weights_L, biases_L], [weights_R, biases_R]

    def append_layers(self, ignore=0, layers=[]):
        """
        TODO : needs testing

        append additional layers to the right of the neural net.
        input layer remains the same but output layer changes.
        ignore : how many layer will be ignored. usually the outpur layer is replaced with.
        layers : appending layer configuration
        """

        n_keep = len(self.layers) - ignore  # number of layers to keep
        n_count = n_keep + len(layers)  # number of layers after appending
        layers_new = []
        weights_new = []
        biases_new = []
        i = 0
        for ll in range(n_count):
            if ll < n_keep:
                layers_new.append(self.layers[ll])
            else:
                layers_new.append(layers[i])
                i += 1
            pass
        n_count -= 1
        n_keep -= 1
        in_dim = self.layers[n_keep - 1]
        i = 0
        for lw in range(n_count):
            if lw < n_keep:
                weights_new.append(self.weights[lw])
                biases_new.append(self.biases[lw])
            else:
                out_dim = layers[i]
                i += 1
                w = self.weight_init_val_xavier(in_dim, out_dim)
                weights_new.append(tf.Variable(w))
                b = tf.ones((1, out_dim))
                biases_new.append(tf.Variable(b))

                in_dim = out_dim

            pass
        # self.layers = layers_new
        # self.weights = weights_new
        return self.__get_new(layers_new, weights_new, biases_new)

    def get_dict(self):
        """
                1. weights
                2. biases
                2. layers
                4. date
                5. time
                6. classname
                9. external parameter
                10. activation_func
                11. optimizer
                11. learning_rate
                12. schedule_name
                13. decay_steps
                14. decay_rate
                15. hyper_param_l2_loss : regularization l2 loss hyper parameter
        """
        dct = {}
        dct['weights'] = [w.numpy() for w in self.weights]
        dct['biases'] = [b.numpy() for b in self.biases]
        dct['layers'] = self.layers

        today = datetime.datetime.today()
        dct['date'] = today.strftime('%Y.%m.%d')
        dct['time'] = today.strftime('%H:%M:%S')
        dct['classname'] = self.get_classname()
        dct['activation_func'] = self.activation_func_name
        dct['optimizer'] = self.optimizer_name
        dct['learning_rate'] = self.learning_rate
        dct['schedule_name'] = self.schedule_name
        if self.schedule_name is not None:
            dct['decay_steps'] = self.decay_steps
            dct['decay_rate'] = self.decay_rate
            pass
        if self.hyper_param_l2_loss is not None:
            dct['hyper_param_l2_loss'] = self.hyper_param_l2_loss
            pass
        return dct

    def current_learning_rate(self, epoch):
        """
        If the scheduler is defined, this comes handy.
        :return: current learning rate
        """
        if self.schedule_name is not None:
            # b = self.lr_schedule.__call__(epoch)
            # print("current learning rate is ", b)
            b = self.lr_schedule(epoch)
            # print("current learning rate is ", b)
            return b.numpy()
            pass
        return self.learning_rate

    def save(self, filename, extension='.pkl'):
        """
        saving model.
        saves the following data :

        filename    : full name of the file that the model will be saved.
        extension   : .pkl for pickle extension
        """
        dct = self.get_dict()
        f_name = filename + extension

        with open(f_name, 'wb') as outfile:
            pickle.dump(dct, outfile)
            pass

        # f_name = filename + '.json'
        # dct2 = to_pure_python(dct)
        # with open(f_name, 'w', encoding='utf-8') as outfile:
        #     json.dump(dct2, outfile)
        #     pass
        # TODO save in json format also

        pass

    def load_weights(self, weights, biases):
        """
        if you happen to have weights and biases in hand and not in a pkl file.
        In this case, weights and biases must be compatible with the given layer configuration.
        """
        new_weights = []
        new_biases = []
        for i in range(len(weights)):
            w = weights[i]
            b = biases[i]
            if (w.shape[0] != self.layers[i]) or (w.shape[1] != self.layers[i + 1]):
                print("got ", w.shape, " needed ", self.layers[i], self.layers[i + 1])
                print("problem with weights")
                pass
            if (b.shape[1] != self.layers[i + 1]):
                print("got ", b.shape, " needed ", 1, self.layers[i + 1])
                print("problem with biases")
                pass
            new_weights.append(tf.Variable(w, name="w{}".format(i)))
            new_biases.append(tf.Variable(b, name="b{}".format(i)))
            pass
        self.weights = new_weights
        self.biases = new_biases
        pass

    def load_dct(self, dct):

        self.layers = dct['layers']
        # print("last layer weights before ")
        # print(self.weights[-1])
        self.load_weights(dct['weights'], dct['biases'])
        # print("last layer weights after ")
        # print(self.weights[-1])
        # self.weights = []
        # self.biases = []
        # for i in range(len(dct['weights'])):
        #     w = dct['weights'][i]
        #     b = dct['biases'][i]
        #     self.weights.append(tf.Variable(w, name="w{}".format(i)))
        #     self.biases.append(tf.Variable(b, name="b{}".format(i)))
        #     pass
        # self.weights = dct['weights']
        # self.biases  = dct['biases']
        keys = dct.keys()
        key = 'activation_func'
        print("keys ", keys)
        print(dct[key])
        if key not in keys:
            print(key, " not found")
            self.activation_func = tf.nn.tanh
            self.activation_func_name = 'tanh'
        elif dct[key] == 'swish':
            print(key, " found swish")
            self.activation_func = tf.nn.swish
            self.activation_func_name = 'swish'
            pass
        elif dct[key] == 'tanh':
            print(key, " found tanh")
            self.activation_func = tf.nn.tanh
            self.activation_func_name = 'tanh'
            pass
        else:
            print("else")
            self.activation_func = tf.nn.tanh
            self.activation_func_name = 'tanh'
            pass

        key = 'optimizer'
        self.learning_rate = 1e-3
        # print(dct)
        if key not in keys:
            if 'learning_rate' in keys:
                self.learning_rate = dct['learning_rate']
                pass
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif dct[key] is 'adam':
            if 'learning_rate' in keys:
                self.learning_rate = dct['learning_rate']
                pass
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            self.learning_rate = 1e-3
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # print(self.weights)
        # print(self.biases)
        if "schedule_name" in keys and dct['schedule_name'] is not None:
            print("loading scheduler")
            epoch = dct['epoch']
            decay_steps = dct['decay_steps']
            decay_rate = dct['decay_rate']
            self.schedule_learning_rate(decay_steps, decay_rate)  # for initial learning_rate
            self.learning_rate = self.current_learning_rate(epoch)
            self.schedule_learning_rate(decay_steps, decay_rate)  # after epoch the initial learning rate is different
            pass
        if 'hyper_param_l2_loss' in keys:
            print("loading regularization hyper parameter")
            self.hyper_param_l2_loss = dct['hyper_param_l2_loss']
            pass
        pass

    def load(self, filename, extension='.pkl'):
        """
        loading model from file
        """
        filename = filename + extension
        with open(filename, 'rb') as file:
            dct = pickle.load(file)
        pass
        self.load_dct(dct)
        pass

    def freeze_layers(self, first_n_layer):
        """
        Freezes specified layers
        first_n_layer : first n layer is kept frozen
        """
        self.freeze_index = first_n_layer
        pass

    pass




def test_nn(f_exact):
    
    lb, ub = [-2], [2]
    d_manager = data_manager.DataManager_v2(lb, ub)

    X_train = d_manager.generate_random(150)
    y_train = f_exact(X_train)

    X_test = d_manager.generate_uniform(100)
    y_test = f_exact(X_test)

    layers = [1, 20, 20, 1]
    learning_rate = 1e-3
    model = NeuralNet_v3(layers=layers, learning_rate=learning_rate)

    X_train = tf.Variable(X_train)
    y_train = tf.constant(y_train)

    def loss_func():
        y = model.evaluate(X_train)
        loss = tf.square(y - y_train)
        return tf.reduce_mean(loss)

    y = model.evaluate(X_test)
    loss = tf.square(y - y_test)
    print("before training loss = ", tf.reduce_mean(loss))

    epoch = 1000
    for e in range(epoch):
        model.minimize(loss_func)
        pass

    y = model.evaluate(X_test)
    loss = tf.square(y - y_test)
    print("after training loss = ", tf.reduce_mean(loss))

    # from main_py.utils import directories
    # model_dir, figure_dir = directories.test_dir_create()

    import matplotlib.pyplot as plt
    plt.plot(X_test, y_test, label="exact")
    plt.plot(X_test, y, label="NN")
    plt.legend()
    plt.xlabel("X")
    filename = "nn_test"
    plt.show()
    # plt.savefig(figure_dir + filename)



# I want to learn these function
def f1(X):
    return X**2

def f2(X):
    return 3*X

if  __name__ == "__main__":
    # test_nn(f1)

    test_nn(f2)

    pass
    