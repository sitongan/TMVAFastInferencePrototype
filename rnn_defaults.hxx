//Code generated automatically by TMVA for Inference of Model file [rnn_defaults.onnx] at [Sat Jul 31 21:26:10 2021] 
#include<vector>
namespace TMVA_SOFIE_rnn_defaults{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_B[15] = {0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978};
float tensor_R[25] = {0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978};
float tensor_W[15] = {0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978, 0.00999999978};
float tensor_Yh[5];
float tensor_Y[15];
// TMVA SOFIE - Warning Model with more than 1 output
void infer(float* tensor_X){
	float *op_0_input = tensor_X;
	float op_0_feedforward[15];
	float *op_0_hidden_state = tensor_Y;
	char op_0_transA = 'N';
	char op_0_transB = 'T';
	int op_0_m = 3;
	int op_0_n = 5;
	int op_0_k = 3;
	float op_0_alpha = 1.;
	float op_0_beta = .0;
	int op_0_bias_size = 15;
	int op_0_incx = 1;
	int op_0_incy = 1;
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_feedforward, &op_0_n);
	BLAS::saxpy_(&op_0_bias_size, &op_0_alpha, tensor_B, &op_0_incx, op_0_feedforward, &op_0_incy);
	for (size_t seq = 0; seq < 3; seq++) {
		size_t offset = seq * 5;
		size_t size = 5;
		size_t h_offset = seq * 5 + 0;
		std::copy(op_0_feedforward + offset, op_0_feedforward + offset + size, op_0_hidden_state + h_offset);
	}
	for (size_t seq = 0; seq < 3; seq++) {
		size_t index = seq;
		int m2 = 1;
		size_t offset = index * 5 + 0;
		size_t size = 5;
		if (seq == 0) {
		} else {
			size_t r_offset = 0;
			size_t previous_offset = (seq - 1) * 5 + 0;
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + r_offset, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_hidden_state + offset, &op_0_n);
		}
		for (size_t i = offset; i < offset + size; i++) {
			float ex = exp(-2 * op_0_hidden_state[i]);
				op_0_hidden_state[i] = (1. - ex) / (1. + ex);
		}
	}
	std::copy(op_0_hidden_state + 10, op_0_hidden_state + 10 + 5, tensor_Yh);
}
} //TMVA_SOFIE_rnn_defaults
