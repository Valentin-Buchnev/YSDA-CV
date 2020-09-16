import abc

import numpy as np

np.seterr(all='raise', under='ignore')

try:
    import tqdm
except ImportError:
    tqdm = None


# region Abstract base classes
class Layer(abc.ABC):
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.output_shape = None

        self.forward_inputs = None
        self.forward_outputs = None

        self._parameter_updaters = {}
        self._optimizer = None
        self._is_built = False

    def build(self, optimizer, prev_layer=None):
        self._optimizer = optimizer
        if prev_layer is not None:
            self.input_shape = prev_layer.output_shape
        elif self.input_shape is None:
            raise ValueError(
                'Unable to infer the input shape for '
                f'layer {self.__class__.__name__}. '
                'If this is the first layer in the model, '
                'please specify the "input_shape" parameter.'
            )
        self.output_shape = (
            self.input_shape if self.output_shape is None
            else self.output_shape
        )
        self._is_built = True

    def add_parameter(self, name, shape, initializer):
        if not self._is_built:
            raise RuntimeError(
                "add_parameter must be called after build "
                "(or after super().build inside custom build method)"
            )
        self._parameter_updaters[name] = \
            self._optimizer.get_parameter_updater(shape)
        param = initializer(shape)
        grad = np.zeros(shape)
        return param, grad

    def update_parameters(self):
        for name, updater in self._parameter_updaters.items():
            for k in (name, name + '_grad'):
                if not hasattr(self, k):
                    raise AttributeError(
                        f"Parameter {name} was registered for "
                        f"{self.__class__.__name__}, but attribute self.{k} "
                        "doesn't exits."
                    )
            parameter = getattr(self, name)
            parameter_grad = getattr(self, name + '_grad')
            parameter[...] = updater(parameter, parameter_grad)

    def __call__(self, inputs):
        self.forward_inputs = inputs
        outputs = self.forward(inputs)
        self.forward_outputs = outputs

        inputs.flags.writeable = False
        outputs.flags.writeable = False

        return outputs

    @abc.abstractmethod
    def forward(self, inputs):
        pass

    @abc.abstractmethod
    def backward(self, grad_outputs):
        pass


class Loss(abc.ABC):
    @abc.abstractmethod
    def __call__(self, y_gt, y_pred):
        pass

    @abc.abstractmethod
    def gradient(self, y_gt, y_pred):
        pass


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def get_parameter_updater(self, shape):
        pass


# endregion


# region Boilerplate
def he_initializer(input_dim):
    def _he_initializer(shape):
        return np.random.randn(*shape) * np.sqrt(2.0 / input_dim)

    return _he_initializer


def range_fn(*args, **kwargs):
    if tqdm is None:
        return range(*args), print, lambda desc: None
    else:
        progress = tqdm.trange(*args, **kwargs)

        def set_desc(desc):
            progress.desc = desc

        return progress, progress.write, set_desc


class Model(object):
    def __init__(self, loss, optimizer):
        if not isinstance(loss, Loss):
            raise RuntimeError(
                "Model loss should be an instance of Loss class. "
                f"Instead got: {loss} "
                f"of type {loss.__class__.__name__}."
            )
        if not isinstance(optimizer, Optimizer):
            raise RuntimeError(
                "Model optimizer should be an instance of Optimizer class. "
                f"Instead got: {optimizer} "
                f"of type {optimizer.__class__.__name__}."
            )
        self._layers = []
        self._loss = loss
        self._optimizer = optimizer
        self._last_y_pred = None

        self.loss_train_history = []
        self.loss_valid_history = []
        self.accuracy_train_history = []
        self.accuracy_valid_history = []

    def add(self, layer):
        if not self._layers:
            layer.build(self._optimizer)
        else:
            layer.build(self._optimizer, prev_layer=self._layers[-1])
        self._layers.append(layer)

    def forward(self, x_gt):
        output = x_gt
        output_meaning = "the network input shape"
        for idx, layer in enumerate(self._layers):
            if layer.input_shape != output.shape[1:]:
                raise ValueError(
                    "In forward pass, the input shape of "
                    f"Layer {self.__class__.__name__} "
                    f"doesn't match {output_meaning}:\n\t"
                    f"layer_expected_input.shape: {layer.input_shape}, "
                    f"layer_actual_input.shape: {output.shape[1:]}"
                )
            output_meaning = "the output shape of previous layer"
            output = layer(output)
            if layer.output_shape != output.shape[1:]:
                raise ValueError(
                    "In forward pass, the output shape of "
                    f"Layer {self.__class__.__name__} "
                    "doesn't match the declared output shape:\n\t"
                    f"layer_expected_output.shape: {layer.output_shape}, "
                    f"layer_actual_output.shape: {output.shape[1:]}"
                )
        self._last_y_pred = output
        return self._last_y_pred

    def backward(self, y_gt):
        if self._loss is None:
            raise ValueError("Loss is not defined")
        if self._last_y_pred.shape != y_gt.shape:
            raise ValueError(
                "Network output shape doesn't match ground truth shape:\n\t"
                f"output.shape: {self._last_y_pred.shape}, "
                f"y_gt.shape: {y_gt.shape}"
            )

        grad_outputs = self._loss.gradient(y_gt, self._last_y_pred)
        output_meaning = "the network output shape"

        for layer in self._layers[::-1]:
            if layer.output_shape != grad_outputs.shape[1:]:
                raise ValueError(
                    "In backward pass, the gradient of the output shape of "
                    f"Layer {self.__class__.__name__} "
                    f"doesn't match {output_meaning}:\n\t"
                    f"layer_expected_grad_output.shape: {layer.output_shape}, "
                    f"layer_actual_grad_output.shape: {grad_outputs.shape[1:]}"
                )
            output_meaning = "output shape of previous layer"
            grad_outputs = layer.backward(grad_outputs)
            if layer.input_shape != grad_outputs.shape[1:]:
                raise ValueError(
                    "In backward pass, the gradient of the input shape of "
                    f"Layer {self.__class__.__name__} "
                    "doesn't match the declared input shape:\n\t"
                    f"layer_expected_grad_input.shape: {layer.output_shape}, "
                    f"layer_actual_grad_input.shape: {grad_outputs.shape[1:]}"
                )

    def fit_batch(self, x_batch, y_batch):
        if self._optimizer is None:
            raise ValueError("Optimizer is not defined")
        y_batch_pred = self.forward(x_batch)
        self.backward(y_batch)
        for layer in self._layers[::-1]:
            layer.update_parameters()
        return self.get_metrics(y_batch, y_batch_pred)

    def fit(
            self, x_train, y_train, batch_size, epochs,
            shuffle=True, verbose=True,
            x_valid=None, y_valid=None
    ):
        size = x_train.shape[0]
        x_gt, y_gt = x_train[:], y_train[:]

        start_epoch = len(self.loss_train_history) + 1
        epochs_range, display, description = range_fn(
            start_epoch, start_epoch + epochs
        )
        description('Training')
        for epoch in epochs_range:
            if shuffle:
                p = np.random.permutation(size)
                x_gt, y_gt = x_train[p], y_train[p]

            train_metrics = np.empty((size // batch_size, 2))
            for step in range(size // batch_size):
                ind_slice = slice(step * batch_size, (step + 1) * batch_size)
                train_metrics[step] = self.fit_batch(
                    x_gt[ind_slice], y_gt[ind_slice]
                )
            train_loss, train_acc = np.mean(train_metrics, axis=0)

            metrics = [
                ("Epoch", f"{epoch: >3}"),
                ("train loss", f"{train_loss:#.6f}"),
                ("train accuracy", f"{train_acc:.2%}"),
            ]

            if (x_valid is not None) and (y_valid is not None):
                valid_loss, valid_acc = self.evaluate(
                    x_valid, y_valid, batch_size
                )
                metrics.extend([
                    ("validation loss", f"{valid_loss:#.6f}"),
                    ("validation accuracy", f"{valid_acc:.2%}"),
                ])
            else:
                valid_loss, valid_acc = float('nan'), float('nan')

            if verbose:
                display(
                    ', '.join(
                        f"{name}: {value}"
                        for name, value in metrics
                    )
                )

            self.loss_valid_history.append(valid_loss)
            self.loss_train_history.append(train_loss)
            self.accuracy_valid_history.append(valid_acc)
            self.accuracy_train_history.append(train_acc)
        if verbose:
            print()

    def get_metrics(self, y_gt, y_pred):
        losses = self._loss(y_gt, y_pred)
        matches = np.argmax(y_gt, axis=-1) == np.argmax(y_pred, axis=-1)
        return np.mean(losses), np.mean(matches)

    def evaluate(self, x_gt, y_gt, batch_size):
        if self._loss is None:
            raise ValueError("Loss is not defined")
        if x_gt.shape[0] != y_gt.shape[0]:
            raise ValueError("x and y must have equal size")

        y_pred = np.empty(y_gt.shape)
        size = x_gt.shape[0]
        for step in range(size // batch_size + 1):
            ind_slice = slice(step * batch_size, (step + 1) * batch_size)
            y_pred[ind_slice] = self.forward(x_gt[ind_slice])
        return self.get_metrics(y_gt, y_pred)
# endregion
