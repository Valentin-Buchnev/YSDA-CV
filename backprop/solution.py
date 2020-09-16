from interface import *
import numpy as np

# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values,
                n - batch size, ... - arbitrary input shape
        :return: np.array((n, ...)), output values,
                n - batch size, ... - arbitrary output shape (same as input)
        """

        return inputs * (inputs > 0)

    def backward(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs,
                n - batch size, ... - arbitrary output shape
        :return: np.array((n, ...)), dLoss/dInputs,
                n - batch size, ... - arbitrary input shape (same as output)
        """
        # your code here \/
        return  grad_outputs * (self.forward_inputs > 0)
        # your code here /\


# ============================== 2.1.2 Softmax ===============================
class Softmax(Layer):
    def forward(self, inputs):
        """
        :param inputs: np.array((n, d)), input values,
                n - batch size, d - number of units
        :return: np.array((n, d)), output values,
                n - batch size, d - number of units
        """
        # your code here \/
        result = np.exp(inputs)
        return (result.T / np.sum(result, axis=1)).T
        # your code here /\

    def backward(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs,
                n - batch size, d - number of units
        :return: np.array((n, d)), dLoss/dInputs,
                n - batch size, d - number of units
        """
        # your code here \/
        res = np.zeros_like(grad_outputs)
        out = self.forward_outputs
        for r, o, go in zip(res, out, grad_outputs):
            r[:] = (np.diag(o) - np.matrix(o).T @ np.matrix(o)) @ go
        return res
        # your code here /\


# =============================== 2.1.3 Dense ================================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_shape = (units,)
        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units, = self.output_shape

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

    def forward(self, inputs):
        """
        :param inputs: np.array((n, d)), input values,
                n - batch size, d - number of input units
        :return: np.array((n, c)), output values,
                n - batch size, c - number of output units
        """
        # your code here \/
        out_shape, = self.output_shape
        batch_size, in_shape = inputs.shape
        res = np.zeros((batch_size, out_shape))
        for r, i in zip(res, inputs):
            r[:] = self.biases + i @ self.weights
        return res
        # your code here /\

    def backward(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs,
                n - batch size, c - number of output units
        :return: np.array((n, d)), dLoss/dInputs,
                n - batch size, d - number of input units
        """
        # your code here \/
        inputs = self.forward_inputs
        input_shape, = self.input_shape
        batch_size, output_shape = grad_outputs.shape

        # Don't forget to update current gradients:
        # dLoss/dWeights
        self.weights_grad[...] = sum(map(lambda x, y: np.matrix(x).T @ np.matrix(y), inputs, grad_outputs)) / batch_size
        # dLoss/dBiases
        self.biases_grad[...] = np.mean(grad_outputs, axis=0)
        
        res = np.zeros((batch_size, input_shape))
        for r, go in zip(res, grad_outputs):
            r[:] = go @ self.weights.T
            
        return res
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def __call__(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values
        :return: np.array((n,)), loss scalars for batch
        """
        # your code here \/
        return np.sum(-y_gt * np.log(y_pred), axis=1)
        # your code here /\

    def gradient(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values
        :return: np.array((n, d)), gradient loss to y_pred
        """
        # your code here \/
        return -y_gt / (y_pred)
        # your code here /\


# ================================ 2.3.1 SGD =================================
class SGD(Optimizer):
    def __init__(self, lr):
        self._lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter
        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam
            :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self._lr * parameter_grad
            # your code here /\

        return updater


# ============================ 2.3.2 SGDMomentum =============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self._lr = lr
        self._momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter
        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam
            :return: np.array, new parameter values
            """
            # your code here \/

            # Don't forget to update the current inertia tensor:
            updater.inertia[...] = updater.inertia * self._momentum + self._lr * parameter_grad
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ======================= 2.4 Train and test on MNIST ========================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    # 2) Add layers to the model
    # (don't forget to specify the input shape for the first layer)

    model = Model(CategoricalCrossentropy(), SGDMomentum(0.1, 0.9))

    shape = x_train.shape[1]
    model.add(Dense(shape, input_shape=(shape,)))
    model.add(ReLU())

    model.add(Dense(3 * y_train.shape[1]))
    model.add(ReLU())

    model.add(Dense(y_train.shape[1]))
    model.add(Softmax())

    # 3) Train and validate the model using the provided data

    model.fit(x_train, y_train, 64, 1, shuffle=True, verbose=True, x_valid=x_valid, y_valid=y_valid)
    # your code here /\
    return model

# ============================================================================
