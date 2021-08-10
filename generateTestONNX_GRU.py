import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker
import onnxruntime

# Test 1: GRU defaults
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 2])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 15, 2])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 15, 5])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 3, 5])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [1, 3, 5])

node = helper.make_node(
    'GRU',
    ['X', 'W', 'R'],
    ['Y', 'Y_h'],
    direction='forward',
    hidden_size=5,
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 15, 2],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 15, 5],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)

graph = onnx.helper.make_graph(
    [node],
    'gru_defaults',
    [X, W, R],
    [Y, Y_h],
    [tensor_W, tensor_R]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'gru_defaults.onnx')


# Test 2: GRU initial_bias
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 3])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 9, 3])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 9, 3])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 18])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 3, 3])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [1, 3, 3])

node = helper.make_node(
    'GRU',
    ['X', 'W', 'R', 'B'],
    ['Y', 'Y_h'],
    direction='forward',
    hidden_size=3,
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 9, 3],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 9, 3],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_B = helper.make_tensor(
    'B',
    TensorProto.FLOAT,
    [1, 18],
    [1., 1., 1., 1., 1., 1., 1., 1., 1.,
     0., 0., 0., 0., 0., 0., 0., 0., 0.]
)

graph = onnx.helper.make_graph(
    [node],
    'gru_initial_bias',
    [X, W, R, B],
    [Y, Y_h],
    [tensor_W, tensor_R, tensor_B]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'gru_initial_bias.onnx')

# Test 3: GRU seq_length
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3, 3])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 15, 3])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 15, 5])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 30])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 1, 3, 5])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [1, 3, 5])

node = helper.make_node(
    'GRU',
    ['X', 'W', 'R', 'B'],
    ['Y', 'Y_h'],
    direction='forward',
    hidden_size=5,
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 15, 3],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 15, 5],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_B = helper.make_tensor(
    'B',
    TensorProto.FLOAT,
    [1, 30],
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
)

graph = onnx.helper.make_graph(
    [node],
    'gru_seq_length',
    [X, W, R, B],
    [Y, Y_h],
    [tensor_W, tensor_R, tensor_B]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'gru_seq_length.onnx')

# Test 4: GRU batchwise
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 2])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 18, 2])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 18, 6])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 3, 6])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [1, 3, 6])

node = helper.make_node(
    'GRU',
    ['X', 'W', 'R'],
    ['Y', 'Y_h'],
    direction='forward',
    hidden_size=6
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 18, 2],
    [.2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2,
     .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 18, 6],
    [.2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2,
     .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2,
     .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2,
     .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2,
     .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2,
     .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2]
)

graph = onnx.helper.make_graph(
    [node],
    'gru_batchwise',
    [X, W, R],
    [Y, Y_h],
    [tensor_W, tensor_R]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'gru_batchwise.onnx')

# Test 5: GRU bidirectional
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 2])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [2, 15, 2])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [2, 15, 5])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 3, 5])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [2, 3, 5])

node = helper.make_node(
    'GRU',
    ['X', 'W', 'R'],
    ['Y', 'Y_h'],
    direction='bidirectional',
    hidden_size=5,
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [2, 15, 2],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
     0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [2, 15, 5],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)

graph = onnx.helper.make_graph(
    [node],
    'gru_bidirectional',
    [X, W, R],
    [Y, Y_h],
    [tensor_W, tensor_R]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'gru_bidirectional.onnx')
