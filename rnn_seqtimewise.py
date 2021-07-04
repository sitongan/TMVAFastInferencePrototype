import torch
from torch import nn
from torch.nn import RNN
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence
#import onnx
#import onnxruntime
import numpy as np
batch_size = 3
seq_length = 3
input_size = 5
hidden_size = 6

# W_B = np.random.randn(1, hidden_size).astype(np.float32)
# R_B = np.random.randn(1, hidden_size).astype(np.float32)
# B = np.concatenate((W_B, R_B), axis=1)

#class RNNSEQ(nn.Module):
#    def __init__(self):
#        super(RNNSEQ, self).__init__()
    
model = RNN(5, 6, batch_first=False)
# set the weights
model.weight_ih_l0.data = torch.FloatTensor(
[[ 0.2369,  0.1346,  0.3317, -0.4822, -0.1363],
[ 0.9420, -0.4502, -2.8174,  0.2889,  1.6715],
[ 0.2967,  1.6799, -0.8343,  0.4493,  0.0370],
[-0.5326,  1.1545, -1.6478,  0.7779, -0.9257],
[-1.4822, -0.8717, -0.0174,  2.0685, -0.7620],
[ 0.0105, -2.9378,  0.8887, -0.9478, -1.5725]])
# set recurrence
model.weight_hh_l0.data = torch.FloatTensor(
[[ 1.0135, -0.2632, -0.6786, -1.0179, -2.1319, -0.0036],
[ 1.9585,  1.1375,  2.1210,  0.6409, -2.0503, -2.4921],
[ 0.5932,  1.5161, -0.7769,  0.2849,  0.2072, -0.3086],
[-0.9655,  0.9178, -0.4292, -1.5054, -0.7396,  0.8929],
[-0.1836, -1.6292,  1.0712,  0.3770,  0.1779, -1.1167],
[-0.6861,  1.2391, -0.5448, -0.3881, -0.5165,  0.0128]])
# set the bias
model.bias_ih_l0.data = torch.tensor(
[.0 ,  .0, 0., .0 ,  .0, .0], dtype=torch.float32
)
model.bias_hh_l0.data = torch.tensor(
[.0, .0, .0, .0, .0, .0], dtype=torch.float32
)

#for name, param in model.named_parameters():
#    print(name, param)
model.eval()
# batch, seq, input = 4, 3, 5
#input = torch.arange(1, 19, dtype=torch.float32).reshape(2, 3, 3) / 100.
b1 = torch.FloatTensor([0.01, -0.01, 0.08, 0.09, 0.001,
                           0.05, -0.09, 0.013, 0.5, 0.005,
                           0.06, 0.087, 0.01, 0.3, -0.001])
b2 = torch.FloatTensor([.09, -0.7, -0.35, 0.0, 0.001,
                           .2, -0.05, .062, -0.04, -0.04])
b3 = torch.FloatTensor([0.001, -0.007, 0.03, 0.0001, 0.0003,
                            -0.008, 0.03, 0.01, -0.034, -0.005])
b4 = torch.FloatTensor([0.16, -0.19, 0.003, 0., 0.0001])


seq = torch.tensor([[[0.01, -0.01, 0.08, 0.09, 0.001],
                     [.09, -0.7, -0.35, 0.0, 0.001],
                     [0.16, -0.19, 0.003, 0., 0.0001]],

                    [[0.05, -0.09, 0.013, 0.5, 0.005],
                     [.2, -0.05, .062, -0.04, -0.04],
                     [.0, .0, .0, .0, .0]],

                    [[0.06, 0.087, 0.01, 0.3, -0.001],
                     [.0, .0, .0, .0, .0],
                     [.0, .0, .0, .0, .0]]])
input = pack_padded_sequence(seq, [3, 2, 1], batch_first=False, enforce_sorted=False)
print("input =\n")
print(input)
# torch.cat((input[1,:,:], input[0,:,:]), 0);
#print(x)
#x.fill_(.1)
y, yh = model(input)
#print(y.shape)
print("\n\n")
print(y)
for i in range(9):
    print(y[0][i])

print("\n\n")
print("\nyh\n")
#print(yh.shape)
print(yh)

#print("\n\n Output")

"""
true output unpacked
tensor([-0.0160, -0.1818, -0.0401, -0.0794,  0.1761,  0.0137, -0.4409, -0.5119,
-0.1651,  0.0995,  0.8556, -0.4281, -0.9776, -0.9818, -0.2740, -0.6920,
0.9529, -0.8501, -0.1869,  0.8827, -0.6948, -0.2732,  0.4479,  0.9408,
-0.4965, -0.9996,  0.8845,  0.9602, -0.9983,  0.9460,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0133,  0.2241, -0.2675, -0.3001,
-0.0715,  0.5097,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000])
"""


#print(output.shape)
#print(output)
#y = output.reshape(18, 4)
#print(y)
#out = y.reshape(3, 2, 3, 4)
#print("\n\nout\n")
#print(out)
#print("\nlast")
#print(y_h)

#y = model(x)
#torch.onnx.export(
#    model,
#    (input, h0),
#    'RNN_bidirectional.onnx',
#    export_params = True,
#    opset_version=10,
#    do_constant_folding=True,
#    input_names=['input', 'h0'],
#    #dynamic_axes={'input': {0:'batch_size'},
#    #              'output': {0:'batch_size'}},
#    output_names=['output']
#)

# Check model
#onnx_model = onnx.load("RNN_bidirectional.onnx")
#onnx.checker.check_model(onnx_model)

# Validate
#ort_session = onnxruntime.InferenceSession("RNN_bidirectional.onnx")
#def to_numpy(tensor):
#    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input),
#                ort_session.get_inputs()[1].name: to_numpy(h0)}
#ort_outs = ort_session.run(None, ort_inputs)

#np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)
#print("SUCCES")

#y = torch.from_numpy(ort_outs[0])
#y = y.reshape(3, 3, 2, 4)
#x = y.permute(0, 2, 1, 3)
#x = y.view(3, 2, 3, 4)
#print(x.shape)
#print(x.reshape(18, 4))
#
#print("yh")
#print(ort_outs[1])

#print(y.view(3, 2, 3, 4))
#print(y.flatten().reshape(3, 3, 2, 4).reshape(3, 2, 3, 4).reshape(18, 4))
#y = ort_outs[0].reshape(3, 3, 2, 4)
#print(y.reshape(18, 4))
