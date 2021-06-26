import onnx
from onnx import numpy_helper, helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np  # type: ignore





x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                [5., 6., 7., 8., 9.],
                [10., 11., 12., 13., 14.],
                [15., 16., 17., 18., 19.],
                [20., 21., 22., 23., 24.],
                [25., 26., 27., 28., 29.],
                [30., 31., 32., 33., 34.]]]]).astype(np.float32)
W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
    [1., 1., 1.],
    [1., 1., 1.]]]]).astype(np.float32)


i_w = numpy_helper.from_array(W, 'W')
t_x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1,1,7,5])
t_w = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1,1,3,3])
t_y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1,1,5,5])

# Convolution with strides=2 and padding
node_with_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    pads=[1, 1, 1, 1],
    strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
)

# Convolution with strides=2 and no padding
node_without_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    pads=[0, 0, 0, 0],
    strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
)

# Convolution with strides=2 and padding only along one dimension (the H dimension in NxCxHxW tensor)
node_with_asymmetric_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    pads=[1, 0, 1, 0],
    strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
)

graph = helper.make_graph(
    [node_with_asymmetric_padding],
    'test-model',
    [t_x, t_w],
    [t_y],
    [i_w]
)

model = helper.make_model(graph, producer_name='python_script')
onnx.checker.check_model(model)

onnx.save(model, 'testCaseConv_2.onnx')
