import onnx
from onnx import numpy_helper, helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np  # type: ignore





x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
    [5., 6., 7., 8., 9.],
    [10., 11., 12., 13., 14.],
    [15., 16., 17., 18., 19.],
    [20., 21., 22., 23., 24.]]]]).astype(np.float32)
W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
    [1., 1., 1.],
    [1., 1., 1.]]]]).astype(np.float32)


i_w = numpy_helper.from_array(W, 'W')
t_x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1,1,5,5])
t_w = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1,1,3,3])
t_y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1,1,5,5])

# Convolution with padding
node_with_padding = onnx.helper.make_node(
'Conv',
inputs=['x', 'W'],
outputs=['y'],
kernel_shape=[3, 3],
# Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
pads=[1, 1, 1, 1],
)

# Convolution without padding
node_without_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
    pads=[0, 0, 0, 0],
)

# Convolution with auto_pad='SAME_LOWER' and strides=2
node = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    auto_pad='SAME_LOWER',
    kernel_shape=[3, 3],
    strides=[2, 2],
)

graph = helper.make_graph(
    [node],
    'test-model',
    [t_x, t_w],
    [t_y],
    [i_w]
)

model = helper.make_model(graph, producer_name='python_script')
onnx.checker.check_model(model)

onnx.save(model, 'testCaseConv_1.onnx')
