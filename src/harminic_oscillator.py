import data_manager
import denn

import tensorflow as tf

DataManager = data_manager.DataManager_v2
APL = denn.APL_v11


class Oscillator(APL):
    def get_classname(self):
        a = super().get_classname()
        a += "_Oscillator"
        return a

    def psi_t(self, X):
        """
        the predictor that we will call while not training.
        return : evaluated function and it's derivative. useful for phase space diagram
        """
        X = tf.Variable(X)
        print(X.dtype)
        y, dy_dX = self.solution_and_derivative(X)
        print(y.dtype)
        return y, dy_dX

    @tf.function
    def pde_loss_vector(self, X):
        with tf.GradientTape() as g2:
            y, dy_dX = self.solution_and_derivative(X)
            pass
        # print(dy_dX)
        d2y_dX2 = g2.gradient(dy_dX, X)
        # print(d2y_dX2)

        loss_f = d2y_dX2 + y * self.ext['w'] ** 2
        return loss_f

    @tf.function
    def solution_and_derivative(self, X):
        with tf.GradientTape() as g:
            y = self.predict(X)
            pass
        dy_dX = g.gradient(y, X)
        return y, dy_dX

    @tf.function
    def pde_loss(self):
        print("Tracing > pde_loss")
        # X = self.get_batch_data()
        X = self.X_batch
        # print(X)
        loss = self.pde_loss_vector(X)
        loss = tf.square(loss)
        loss = tf.reduce_mean(loss)
        loss = tf.sqrt(loss)
        return loss

    def set_boundary(self, lower_boundary, upper_boundary):
        super(Oscillator, self).set_boundary(lower_boundary, upper_boundary)
        self.set_boundary_train_data()

    def set_boundary_train_data(self):
        self.X_b_train = tf.Variable([self.lower_boundary])
        self.position = 1.
        self.velocity = 0.
        pass

    @tf.function
    def boundary_loss(self):
        y, dy_dx = self.solution_and_derivative(self.X_b_train)
        #         print("x= ", x, ", y=", y)
        loss = tf.square(y - self.position)
        loss += tf.square(dy_dx - self.velocity)
        return tf.reduce_mean(loss)

    @tf.function
    def loss_function(self):
        Lp = self.pde_loss()
        Lb = self.boundary_loss()
        return Lp + Lb

    pass




import numpy as np

def test(epoch=1000):
    a = 20
    layers = [1, a, a, 1]

    model  = Oscillator(layers=layers, learning_rate=4e-4)
    ext = {'w':2}
    model.set_external_parameter(ext)
    model.set_variable_order(['t'])
    y0, v0 = 1.0, 0.0

    lb = np.array([0], dtype=np.float32)
    ub = np.array([10], dtype=np.float32)

    model.set_boundary(lb, ub)

    lb, ub = [0], [10]
    model.set_boundary(lb, ub)

    d_manager = DataManager(lb, ub)
    model.set_data_manager(d_manager)
    model.set_training_data_size(5000)

    a = DataManager.generate_uniform_s([0], [5], 100)
    b = DataManager.generate_uniform_s([5], [10], 100)
    # model.set_training_data([a, b])
    # model.set_steep_density(0.65)
    model.info()
    # model.train(epoch, interval=200, batch_size=100, generate_data=True, rar=True)
    model.train(epoch, interval=500, batch_size=512, rar=0.4)

    signature = model.get_classname()
    print(signature)
    filename = signature + "_oscillator_freq{}_test".format(ext['w'])
    print(filename)

    # from main_py.utils import directories
    # model_dir, figure_dir = directories.test_dir_create()

    # model.save(model_dir + filename)
    # return d_manager, model

    # model.load(dir_saved_model + "APLv10_oscillator")


    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2, figsize=(2*5, 2*3.5), dpi=100)
    axs = axes.flatten()

    X = tf.linspace(lb, ub, 300)
    X = tf.cast(X, tf.float32)
    y, v = model.psi_t(X)
    axs[0].plot(X, y, label='position')
    axs[0].legend()
    axs[1].plot(X, v, label='velocity')
    axs[1].legend()

    axs[2].plot(y[0], v[0], 'o', label='initial')
    axs[2].plot(y, v, label='phase space', linewidth=0.5)
    axs[2].set_xlabel(r'$x$')
    axs[2].set_ylabel(r'$v$')
    axs[2].legend()

    # axs[2].set_xlim([-omega, omega])
    # axs[2].set_ylim([-omega, omega])


    hist= model.get_history()
    print("loss ", hist[-1])
    axs[3].plot(hist[-200:,0], hist[-200:,1], label='loss')
    axs[3].set_xlabel('epoch')
    axs[3].set_ylabel('loss')
    axs[3].text(0.2, 0.6, "iteration={}".format(hist[-1][0]), transform=axs[3].transAxes)
    axs[3].text(0.2, 0.8, "loss={:e}".format(hist[-1][1]), transform=axs[3].transAxes)
    axs[3].ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    # plt.show()
    filename = "APLv10_oscillator_high_freq{}.png".format(6)
    plt.savefig(filename)






if __name__ == "__main__":
    test(5_000)
    # test_straight_line()