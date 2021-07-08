import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker

# Test 1: lstm defaults
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 1, 2])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 12, 2])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 12, 3])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 1, 1, 3])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [1, 1, 3])

node = helper.make_node(
    'LSTM',
    ['X', 'W', 'R'],
    ['Y', 'Y_h'],
    "",
    activations=['Sigmoid', 'Tanh', 'Tanh'],
    clip=0.,
    direction='forward',
    hidden_size=3,
    layout=0
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 12, 2],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 12, 3],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)

graph = onnx.helper.make_graph(
    [node],
    'lstm_defaults',
    [X, W, R],
    [Y, Y_h],
    [tensor_W, tensor_R]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'lstm_defaults.onnx')

# Test 2: lstm initial_bias
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 1, 3])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 16, 3])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 16, 4])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 32])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 1, 1, 4])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [1, 1, 4])

node = helper.make_node(
    'LSTM',
    ['X', 'W', 'R', 'B'],
    ['Y', 'Y_h'],
    "",
    activations=['Sigmoid', 'Tanh', 'Tanh'],
    clip=0.,
    direction='forward',
    hidden_size=4,
    layout=0
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 16, 3],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 16, 4],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_B = helper.make_tensor(
    'B',
    TensorProto.FLOAT,
    [1, 32],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0.]
)

graph = onnx.helper.make_graph(
    [node],
    'lstm_initial_bias',
    [X, W, R, B],
    [Y, Y_h],
    [tensor_W, tensor_R, tensor_B]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'lstm_initial_bias.onnx')

# Test 3: lstm peepholes
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2, 4])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 12, 4])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 12, 3])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 24])
seq = helper.make_tensor_value_info('sequence_lens', TensorProto.FLOAT, [2])
initial_h = helper.make_tensor_value_info('initial_h', TensorProto.FLOAT, [1, 2, 3])
initial_c = helper.make_tensor_value_info('initial_c', TensorProto.FLOAT, [1, 2, 3])
P = helper.make_tensor_value_info('P', TensorProto.FLOAT, [1, 9])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 2, 3])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [1, 2, 3])

node = helper.make_node(
    'LSTM',
    ['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h', 'initial_c', 'P'],
    ['Y', 'Y_h'],
    "",
    activations=['Sigmoid', 'Tanh', 'Tanh'],
    clip=0.,
    direction='forward',
    hidden_size=3,
    layout=0
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 12, 4],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 12, 3],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_B = helper.make_tensor(
    'B',
    TensorProto.FLOAT,
    [1, 24],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
)
tensor_seq = helper.make_tensor(
    'sequence_lens',
    TensorProto.FLOAT,
    [2],
    [1, 1]
)
tensor_initial_h = helper.make_tensor(
    'initial_h',
    TensorProto.FLOAT,
    [1, 2, 3],
    [0., 0., 0., 0., 0., 0.]
)
tensor_initial_c = helper.make_tensor(
    'initial_c',
    TensorProto.FLOAT,
    [1, 2, 3],
    [0., 0., 0., 0., 0., 0.]
)
tensor_P = helper.make_tensor(
    'P',
    TensorProto.FLOAT,
    [1, 9],
    [0.1, 0.1, 0.1,
     0.1, 0.1, 0.1,
     0.1, 0.1, 0.1]
)
graph = onnx.helper.make_graph(
    [node],
    'lstm_peepholes',
    [X, W, R, B, seq, initial_h, initial_c, P],
    [Y, Y_h],
    [tensor_W, tensor_R, tensor_B, tensor_seq, tensor_initial_h, tensor_initial_c, tensor_P]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'lstm_peepholes.onnx')

# Test 4: lstm batchwise
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 1, 2])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 28, 2])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 28, 7])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 1, 1, 7])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [3, 1, 7])

node = helper.make_node(
    'LSTM',
    ['X', 'W', 'R'],
    ['Y', 'Y_h'],
    "",
    activations=['Sigmoid', 'Tanh', 'Tanh'],
    clip=0.,
    direction='forward',
    hidden_size=7,
    layout=1
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 28, 2],
    [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 28, 7],
    [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
)

graph = onnx.helper.make_graph(
    [node],
    'lstm_batchwise',
    [X, W, R],
    [Y, Y_h],
    [tensor_W, tensor_R]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'lstm_batchwise.onnx')


# Test 5: lstm bidirectional
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 1, 2])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [2, 12, 2])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [2, 12, 3])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 2, 1, 3])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [2, 1, 3])
Y_c = helper.make_tensor_value_info('Y_c', TensorProto.FLOAT, [2, 1, 3])
node = helper.make_node(
    'LSTM',
    ['X', 'W', 'R'],
    ['Y', 'Y_h', 'Y_c'],
    "",
    activations=['Sigmoid', 'Tanh', 'Tanh', 'Sigmoid', 'Tanh', 'Tanh'],
    clip=0.,
    direction='bidirectional',
    hidden_size=3,
    layout=0
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [2, 12, 2],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [2, 12, 3],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)

graph = onnx.helper.make_graph(
    [node],
    'lstm_bidirectional',
    [X, W, R],
    [Y, Y_h, Y_c],
    [tensor_W, tensor_R]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'lstm_bidirectional.onnx')

