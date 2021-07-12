import onnx
from onnx import numpy_helper, helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np  # type: ignore


def _batchnorm_test_mode(x, s, bias, mean, var, epsilon=1e-5): 
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias

# input size: (2, 3, 4, 5)
x = np.random.randn(2, 3, 4, 5).astype(np.float32)
s = np.random.randn(3).astype(np.float32)
bias = np.random.randn(3).astype(np.float32)
mean = np.random.randn(3).astype(np.float32)
var = np.random.rand(3).astype(np.float32)
y = _batchnorm_test_mode(x, s, bias, mean, var).astype(np.float32)

i_x = numpy_helper.from_array(x, 'x')
i_s = numpy_helper.from_array(s, 's')
i_b = numpy_helper.from_array(bias, 'bias')
i_m = numpy_helper.from_array(mean, 'mean')
i_v = numpy_helper.from_array(var, 'var')
i_y = numpy_helper.from_array(y, 'y')

t_x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2,3,4,5])
t_s = helper.make_tensor_value_info('s', TensorProto.FLOAT, [1,1,1,3])
t_b = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1,1,1,3])
t_m = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [1,1,1,3])
t_v = helper.make_tensor_value_info('var', TensorProto.FLOAT, [1,1,1,3])
t_y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2,3,4,5])

node = onnx.helper.make_node(
    'BatchNormalization',
    inputs=['x', 's', 'bias', 'mean', 'var'],
    outputs=['y'],
)

graph = helper.make_graph(
    [node],
    'test-model',
    [t_x, t_s, t_b, t_m, t_v],
    [t_y],
    [i_s, i_b, i_m, i_v]
)

model = helper.make_model(graph, producer_name='python_script')
onnx.checker.check_model(model)

onnx.save(model, 'testCaseBatchNorm_1.onnx')
print("model saved")

print("Input array saved to input_arr")
np.savetxt("input_arr.csv", np.ravel(x), delimiter=",")

print("Expected value of y")
print(np.ravel(y))