import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker

# Test 1: rnn defaults
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 1, 3])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 5, 3])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 5, 5])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 10])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 1, 1, 5])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [1, 1, 5])

node = helper.make_node(
    'RNN',
    ['X', 'W', 'R', 'B'],
    ['Y', 'Y_h'],
    "",
    activations=['Tanh'],
    clip=0.,
    direction='forward',
    hidden_size=5,
    layout=0
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 5, 3],
    [0.01, 0.01, 0.01, 0.01, 0.01,
     0.01, 0.01, 0.01, 0.01, 0.01,
     0.01, 0.01, 0.01, 0.01, 0.01]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 5, 5],
    [0.01, 0.01, 0.01, 0.01, 0.01,
     0.01, 0.01, 0.01, 0.01, 0.01,
     0.01, 0.01, 0.01, 0.01, 0.01,
     0.01, 0.01, 0.01, 0.01, 0.01,
     0.01, 0.01, 0.01, 0.01, 0.01]
)
tensor_B = helper.make_tensor(
    'B',
    TensorProto.FLOAT,
    [1, 10],
    [0.01, 0.01, 0.01, 0.01, 0.01,
     0., 0., 0., 0., 0.]
)

graph = onnx.helper.make_graph(
    [node],
    'rnn_defaults',
    [X, W, R, B],
    [Y, Y_h],
    [tensor_W, tensor_R, tensor_B]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'rnn_defaults.onnx')

# Test 2: rnn seq_length
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3, 3])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 5, 3])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 5, 5])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 10])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 1, 3, 5])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [1, 3, 5])

node = helper.make_node(
    'RNN',
    ['X', 'W', 'R', 'B'],
    ['Y', 'Y_h'],
    "",
    activations=['Tanh'],
    clip=0.,
    direction='forward',
    hidden_size=5,
    layout=0
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 5, 3],
    [0.02, 0.02, 0.02, 0.02, 0.02,
     0.02, 0.02, 0.02, 0.02, 0.02,
     0.02, 0.02, 0.02, 0.02, 0.02]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 5, 5],
    [0.02, 0.02, 0.02, 0.02, 0.02,
     0.02, 0.02, 0.02, 0.02, 0.02,
     0.02, 0.02, 0.02, 0.02, 0.02,
     0.02, 0.02, 0.02, 0.02, 0.02,
     0.02, 0.02, 0.02, 0.02, 0.02]
)
tensor_B = helper.make_tensor(
    'B',
    TensorProto.FLOAT,
    [1, 10],
    [0.031, 0.031, 0.031, 0.031, 0.031,
     0.021, 0.021, 0.021, 0.021, 0.021]
)

graph = onnx.helper.make_graph(
    [node],
    'rnn_seq_length',
    [X, W, R, B],
    [Y, Y_h],
    [tensor_W, tensor_R, tensor_B]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'rnn_seq_length.onnx')

# Test 3: rnn batchwise
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 1, 2])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 4, 2])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 4, 4])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 1, 1, 4])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [3, 1, 4])

node = helper.make_node(
    'RNN',
    ['X', 'W', 'R'],
    ['Y', 'Y_h'],
    "",
    activations=['Tanh'],
    clip=0.,
    direction='forward',
    hidden_size=4,
    layout=1
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 4, 2],
    [0.05, 0.05, 0.05, 0.05,
     0.05, 0.05, 0.05, 0.05]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 4, 4],
    [0.05, 0.05, 0.05, 0.05,
     0.05, 0.05, 0.05, 0.05,
     0.05, 0.05, 0.05, 0.05,
     0.05, 0.05, 0.05, 0.05]
)

graph = onnx.helper.make_graph(
    [node],
    'rnn_batchwise',
    [X, W, R],
    [Y, Y_h],
    [tensor_W, tensor_R]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'rnn_batchwise.onnx')

# Test 4: rnn bidirectional
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 3, 2])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [2, 4, 2])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [2, 4, 4])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 8])
sequence_lens = helper.make_tensor_value_info('sequence_lens', TensorProto.FLOAT, [3])
initial_h = helper.make_tensor_value_info('initial_h', TensorProto.FLOAT, [2, 3, 4])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 2, 3, 4])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [2, 3, 4])

node = helper.make_node(
    'RNN',
    ['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h'],
    ['Y', 'Y_h'],
    "",
    activations=['Tanh', 'Tanh'],
    clip=0.,
    direction='bidirectional',
    hidden_size=4,
    layout=0
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [2, 4, 2],
    [1.16308,    2.21221,
     0.483805,   0.774004,
     0.299563,   1.04344,
     0.153025,   1.18393,
     -1.16881,   1.89171,
     1.55807,   -1.23474,
    -0.545945, -1.77103,
    -2.35563,  -0.451384]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [2, 4, 4],
    [-0.264848, -1.30311,   0.0712087,  0.64198,
     -2.76538,  -0.652074, -0.784275,  -1.76749,
     -0.450673, -0.917929, -0.966654,   0.650856,
     0.285538,  -0.909848, -1.90459,   -0.140926,
     -1.37131,   0.780644,  0.441009,   1.15856,
     0.313298,   1.96766,  -1.11991,   -0.00440959,
     0.407622,   2.60569,  -0.840986,   0.585658,
    0.823292,  -0.696818,  1.15115,    0.150269]
)
tensor_B = helper.make_tensor(
    'B',
    TensorProto.FLOAT,
    [2, 8],
    [-0.161029, -2.58991, 0.339721,  -0.31664,
     0.049053, -1.89795, -0.327121, -0.159628,
     -0.183054, -0.977459, -1.08309, -0.0165881,
     1.99349,   1.35513, -0.697978, -0.708618]
)
tensor_seq = helper.make_tensor(
    'sequence_lens',
    TensorProto.FLOAT,
    [3],
    [3, 3, 3]
)
tensor_initial_h = helper.make_tensor(
    'initial_h',
    TensorProto.FLOAT,
    [2, 3, 4],
    [-0.371075, 0.252533, -1.42195, 0.39303,
     -0.463112, -1.02438, -0.538399, -2.21508,
     -1.4221, -0.149365, 1.2587, 1.38294,
     -0.0841612, 1.45697, 0.0679387, 2.11548,
     -1.51051, 1.50948, 0.206351, -0.981445,
     -0.221477, -0.230484, 0.453313, 0.795476]
)

graph = onnx.helper.make_graph(
    [node],
    'rnn_bidirectional',
    [X, W, R, B, sequence_lens, initial_h],
    [Y, Y_h],
    [tensor_W, tensor_R, tensor_B, tensor_seq, tensor_initial_h]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'rnn_bidirectional.onnx')

# Test 5: rnn bidirectional batchwise
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 3, 2])
initial_h = helper.make_tensor_value_info('initial_h', TensorProto.FLOAT, [3, 2, 4])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 3, 2, 4])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [3, 2, 4])

node = helper.make_node(
    'RNN',
    ['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h'],
    ['Y', 'Y_h'],
    "",
    activations=['Tanh', 'Tanh'],
    clip=0.,
    direction='bidirectional',
    hidden_size=4,
    layout=1
)

tensor_initial_h = helper.make_tensor(
    'initial_h',
    TensorProto.FLOAT,
    [2, 3, 4],
    [-0.371075,   0.252533, -1.42195,    0.39303,
     -0.0841612,  1.45697,   0.0679387,  2.11548,
     -0.463112,  -1.02438,  -0.538399,  -2.21508,
     -1.51051,    1.50948,   0.206351,  -0.981445,
     -1.4221,    -0.149365,  1.2587,     1.38294,
    -0.221477,  -0.230484,  0.453313,   0.795476]
)

graph = onnx.helper.make_graph(
    [node],
    'rnn_bidirectional_batchwise',
    [X, W, R, B, sequence_lens, initial_h],
    [Y, Y_h],
    [tensor_W, tensor_R, tensor_B, tensor_seq, tensor_initial_h]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'rnn_bidirectional_batchwise.onnx')

# Test 6: rnn sequence
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 3, 5])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 6, 5])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 6, 6])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 12])
sequence_lens = helper.make_tensor_value_info('sequence_lens', TensorProto.FLOAT, [3])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 1, 3, 6])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [1, 3, 6])

node = helper.make_node(
    'RNN',
    ['X', 'W', 'R', 'B', 'sequence_lens'],
    ['Y', 'Y_h'],
    "",
    activations=['Tanh'],
    clip=0.,
    direction='forward',
    hidden_size=6,
    layout=0
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 6, 5],
    [0.2369,  0.1346,  0.3317, -0.4822, -0.1363,
     0.9420, -0.4502, -2.8174,  0.2889,  1.6715,
     0.2967,  1.6799, -0.8343,  0.4493,  0.0370,
     -0.5326,  1.1545, -1.6478,  0.7779, -0.9257,
     -1.4822, -0.8717, -0.0174,  2.0685, -0.7620,
     0.0105, -2.9378,  0.8887, -0.9478, -1.5725]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 6, 6],
    [1.0135, -0.2632, -0.6786, -1.0179, -2.1319, -0.0036,
     1.9585,  1.1375,  2.1210,  0.6409, -2.0503, -2.4921,
     0.5932,  1.5161, -0.7769,  0.2849,  0.2072, -0.3086,
     -0.9655,  0.9178, -0.4292, -1.5054, -0.7396,  0.8929,
     -0.1836, -1.6292,  1.0712,  0.3770,  0.1779, -1.1167,
     -0.6861,  1.2391, -0.5448, -0.3881, -0.5165,  0.0128]
)
tensor_B = helper.make_tensor(
    'B',
    TensorProto.FLOAT,
    [1, 12],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
)
tensor_seq = helper.make_tensor(
    'sequence_lens',
    TensorProto.FLOAT,
    [3],
    [3, 2, 1]
)

graph = onnx.helper.make_graph(
    [node],
    'rnn_sequence',
    [X, W, R, B, sequence_lens],
    [Y, Y_h],
    [tensor_W, tensor_R, tensor_B, tensor_seq]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'rnn_sequence.onnx')

# Test 7: rnn sequence batchwise
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 3, 1, 6])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [3, 1, 6])

node = helper.make_node(
    'RNN',
    ['X', 'W', 'R', 'B', 'sequence_lens'],
    ['Y', 'Y_h'],
    "",
    activations=['Tanh'],
    clip=0.,
    direction='forward',
    hidden_size=6,
    layout=1
)

graph = onnx.helper.make_graph(
    [node],
    'rnn_sequence_batchwise',
    [X, W, R, B, sequence_lens],
    [Y, Y_h],
    [tensor_W, tensor_R, tensor_B, tensor_seq]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'rnn_sequence_batchwise.onnx')

