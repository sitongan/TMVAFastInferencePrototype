import onnx
from onnx import numpy_helper, helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np  # type: ignore

def gemm_reference_implementation(A, B, C=None, alpha=1., beta=1., transA=0,
                                  transB=0):  # type: (np.ndarray, np.ndarray, Optional[np.ndarray], float, float, int, int) -> np.ndarray
    A = A if transA == 0 else A.T
    B = B if transB == 0 else B.T
    C = C if C is not None else np.array(0)

    Y = alpha * np.dot(A, B) + beta * C

    return Y


node = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['y']
)
a = np.random.ranf([3, 5]).astype(np.float32)
b = np.random.ranf([5, 4]).astype(np.float32)
c = np.zeros([1, 4]).astype(np.float32)
y = gemm_reference_implementation(a, b, c)

print(y)

print(a)


i_b = numpy_helper.from_array(b, 'b')
i_c = numpy_helper.from_array(c, 'c')

t_a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [3,5])
t_b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [5,4])
t_c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [1,4])
t_y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3,4])


node = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['y']
)

graph = helper.make_graph(
    [node],
    'test-model',
    [t_a, t_b, t_c],
    [t_y],
    [i_b,i_c]
)

model = helper.make_model(graph, producer_name='python_script')
onnx.checker.check_model(model)

onnx.save(model, 'testCaseGemmModel.onnx')
