import numpy as np
from data import LinearlySeparableClasses, NonlinearlySeparableClasses
from visualization_utils import inspect_data, plot_data, x_data_from_grid, visualize_activation_function, \
    plot_two_layer_activations


# Example activation functions

def relu(logits):
    return np.maximum(logits, 0)


def sigmoid(logits):
    return 1. / (1. + np.exp(-logits))
    # return np.exp(-np.logaddexp(0, -logits))


def hardlim(logits):
    return (logits > 0).astype(np.float32)
    # return np.round(sigmoid(logits))


def linear(logits):
    return logits


def zad1_single_neuron(student_id):
    gen = LinearlySeparableClasses()
    x, y = gen.generate_data(seed=student_id)
    n_samples, n_features = x.shape

    inspect_data(x, y)
    plot_data(x, y, plot_xy_range=[-1, 2])

    # single neuron model
    class SingleNeuron:
        def __init__(self, n_in, f_act):
            self.W = 0.01 * np.random.randn(n_in, 1)  # size of W: [n_in, 1]
            self.b = 0.01 * np.random.randn(1)  # size of b: [1]
            self.f_act = f_act

        def forward(self, x_data):
            """
            :param x_data: neuron input: np.array of size [n_samples, n_in]
            :return: neuron output: np.array of size [n_samples, 1]
            """
            return hardlim(np.matmul(x_data, self.W) + self.b)

    # neuron initialized with random weights
    model = SingleNeuron(n_in=n_features, f_act=hardlim)

    # TODO: set appropriate weights (0.5 point)
    model.W[:, 0] = [-0.5614813161248267, -0.8851094477846504]
    model.b[:] = [0.3196431327430931]


    y_pred = model.forward(x)
    # model performance and evaluation
    print(f'Accuracy = {np.mean(y == y_pred) * 100}%')

    # test on the entire input space (with visualization)
    x_grid = x_data_from_grid(min_xy=-1, max_xy=2, grid_size=1000)
    y_pred_grid = model.forward(x_grid)
    plot_data(x, y, plot_xy_range=[-1, 2], x_grid=x_grid, y_grid=y_pred_grid, title='Neuron decision boundary')


def zad2_two_layer_net(student_id):
    gen = NonlinearlySeparableClasses()
    x, y = gen.generate_data(seed=student_id)
    n_samples, n_features = x.shape

    inspect_data(x, y)
    plot_data(x, y, plot_xy_range=[-1, 2])

    # layer, i.e., n_out independent neurons operating on the same input
    # (the i-th neuron has its parameters in the i-th column of the W matrix and the i-th position of the b vector)
    class DenseLayer:
        def __init__(self, n_in, n_out, f_act):
            self.W = 0.01 * np.random.randn(n_in, n_out)  # size of W: ([n_in, n_out])
            self.b = 0.01 * np.random.randn(n_out)  # size of b ([n_out])
            self.f_act = f_act

        def forward(self, x_data):
            return self.f_act(np.matmul(x_data, self.W) + self.b)

    class SimpleTwoLayerNetwork:
        def __init__(self, n_in, n_hidden, n_out):
            self.hidden_layer = DenseLayer(n_in, n_hidden, hardlim)
            self.output_layer = DenseLayer(n_hidden, n_out, hardlim)

        def forward(self, x_data):
            return self.output_layer.forward(self.hidden_layer.forward(x_data))


    # model initialized with random weights
    model = SimpleTwoLayerNetwork(n_in=n_features, n_hidden=2, n_out=1)

    a, b, c = 0, 0, 0
    for i in range(0, 50000):
        a = np.random.rand() * 2 - 1
        b = np.random.rand() * 2 - 1
        c = np.random.rand() * 2 - 1
        d = np.random.rand() * 2 - 1
        e = np.random.rand() * 2 - 1
        f = np.random.rand() * 2 - 1
        model.hidden_layer.W[:, 0] = [a, b]  # weights of neuron h1
        model.hidden_layer.W[:, 1] = [c, d]  # weights of neuron h2
        model.hidden_layer.b[:] = [e, f]
        model.output_layer.W[:, 0] = [0.5, 0.5]  # weights of the output neuron
        model.output_layer.b[:] = [-0.8]
        y_pred = model.forward(x)

        if np.mean(y == y_pred) * 100 == 100.0:
            print(a, b, c, d, e, f)
            break

    # model performance and evaluation
    print(f'Accuracy = {np.mean(y == y_pred) * 100}%')

    plot_two_layer_activations(model, x, y)


if __name__ == '__main__':
    # visualize_activation_function(relu)

    student_id = 188778

    zad1_single_neuron(student_id)
    zad2_two_layer_net(student_id)
