import pickle

import numpy as np

import interface


def load_test_data(test_name, test_path):
    return pickle.load(open(f'{test_path}/{test_name}.pickle', 'rb'))


# region Generic tests
def check_interface(impl, interface_base):
    this_impl = f"The {impl.__name__} class "
    this_interface = f" {interface_base.__name__} abstract base class."
    assert issubclass(impl, interface_base), (
            this_impl + "should inherit from the" + this_interface
    )
    for method_name in interface_base.__abstractmethods__:
        assert hasattr(impl, method_name), (
                this_impl +
                f"doesn't have the {method_name} method, required by" +
                this_interface
        )
        method = getattr(impl, method_name)
        assert not getattr(method, '__isabstractmethod__', False), (
                this_impl +
                f"doesn't implement the {method_name} method, required by" +
                this_interface
        )


def init_layer(layer_impl, test_data):
    # Build layer
    layer = layer_impl(**test_data['kwargs'])
    layer.build(MockOptimizer())
    values = {}

    # Set parameters of layer
    for k, v in test_data.items():
        if k.startswith('parameter_'):
            parameter_name = k[len('parameter_'):]
            setattr(layer, parameter_name, v)
            values[f"layer.{parameter_name}"] = v

    # Test forward pass
    values["inputs"] = test_data['inputs']
    return layer, values


def forward_layer(layer_impl, test_data):
    layer, values = init_layer(layer_impl, test_data)

    # Test forward pass
    with SubTest(
            x="layer.forward(inputs)",
            values=values
    ):
        assert_ndarray_equal(
            actual=layer.forward(test_data['inputs']),
            correct=test_data['outputs']
        )


def backward_layer(layer_impl, test_data):
    layer, values = init_layer(layer_impl, test_data)
    layer(test_data['inputs'])

    # Test backward pass
    values["grad_outputs"] = test_data['grad_outputs']
    with SubTest(
            x="layer.backward(grad_outputs)",
            values=values
    ):
        assert_ndarray_equal(
            actual=layer.backward(test_data['grad_outputs']),
            correct=test_data['grad_inputs']
        )

    # Test, that parameter gradients were calculated correctly
    for k, grad_value in test_data.items():
        if k.startswith('param_grad_'):
            grad_name = k[len('param_grad_'):] + '_grad'
            with SubTest(
                    x=f"layer.{grad_name}",
                    values=values
            ):
                assert_ndarray_equal(
                    actual=getattr(layer, grad_name),
                    correct=grad_value
                )


def simulate_optimizer(optimizer_impl, test_data):
    optimizer = optimizer_impl(**test_data['kwargs'])
    updaters = [
        optimizer.get_parameter_updater(sh)
        for sh in test_data['parameter_shapes']
    ]
    for step in test_data['steps']:
        for param_data, updater in zip(step, updaters):
            with SubTest(
                    x="updater(parameter, parameter_grad)",
                    values={
                        "parameter": param_data['value'],
                        "parameter_grad": param_data['grad']
                    }
            ):
                assert_ndarray_equal(
                    actual=updater(param_data['value'], param_data['grad']),
                    correct=param_data['new_value']
                )


# endregion


# region Generic asserts
class SubTest(object):
    def __init__(self, x=None, values=None):
        xname = "" if x is None else (x + " ")
        self.x = f"actual  {xname}value:"
        self.y = f"correct {xname}value:"
        self.values = {} if values is None else values

    def get_message(self):
        if self.values:
            return (
                    "\n\n" +
                    "\n".join(
                        f" {k}: {repr(v)}"
                        for k, v in self.values.items()
                    )
            )
        else:
            return ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        __tracebackhide__ = True
        if exc_type is not None and issubclass(exc_type, AssertionError):
            if len(exc_val.args) == 1:
                original = exc_val.args[0]
                original = original.replace('x:', self.x)
                original = original.replace('y:', self.y)
                exc_val.args = (original + self.get_message(),)
            elif len(exc_val.args) > 1:
                exc_val.args = exc_val.args + (self.get_message(),)
            else:
                exc_val.args = (self.get_message(),)
            return False


def assert_value_is_ndarray(value):
    __tracebackhide__ = True
    assert isinstance(value, (np.ndarray, np.generic)), (
        f"Value should be an instance of np.ndarray, but it is {type(value)}."
    )


def assert_dtypes_compatible(actual_dtype, correct_dtype):
    __tracebackhide__ = True
    assert (
            np.can_cast(actual_dtype, correct_dtype, casting='same_kind') and
            np.can_cast(correct_dtype, actual_dtype, casting='same_kind')
    ), (
        "The dtypes of actual value and correct value are not the same "
        "and can't be safely converted.\n"
        f"actual.dtype={actual_dtype}, correct.dtype={correct_dtype}"
    )


def assert_shapes_match(actual_shape, correct_shape):
    __tracebackhide__ = True
    assert (
            len(actual_shape) == len(correct_shape) and
            actual_shape == correct_shape
    ), (
        "The shapes of actual value and correct value are not the same.\n"
        f"actual.shape={actual_shape}, correct.shape={correct_shape}"
    )


def assert_ndarray_equal(*, actual, correct):
    __tracebackhide__ = True
    assert_value_is_ndarray(actual)
    assert_dtypes_compatible(actual.dtype, correct.dtype)
    assert_shapes_match(actual.shape, correct.shape)
    np.testing.assert_allclose(actual, correct, rtol=1e-5, verbose=True)


# endregion


# region Mocks
class MockOptimizer(interface.Optimizer):
    """Fake optimizer, that doesn't update the parameters"""

    def get_parameter_updater(self, shape):
        def update(parameter, parameter_grad):
            return parameter

        return update

# endregion
