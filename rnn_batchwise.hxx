//Code generated automatically by TMVA for Inference of Model file [rnn_batchwise.onnx] at [Sun Jul  4 18:07:17 2021] 
#include<string>
#include<vector>
namespace TMVA_SOFIE_rnn_batchwise{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_R[16] = {0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007};
float tensor_W[8] = {0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007, 0.0500000007};
float tensor_Yh[12];
float tensor_Y[12];
// TMVA SOFIE - Warning Model with more than 1 output
void infer(float* tensor_X){
	float op_0_input[6];
	for(size_t seq = 0; seq < 1; seq++) {
		for(size_t batch = 0; batch < 3; batch++) {
			for(size_t i = 0; i < 2; i++) {
				op_0_input[seq * 6 + batch * 2 + i] = tensor_X[batch * 2 + seq * 2 + i];
			}
		}
	}
	float op_0_feedforward[12];
	float op_0_hidden_state[12];
	char op_0_transA = 'N';
	char op_0_transB = 'T';
	int op_0_m = 3;
	int op_0_n = 4;
	int op_0_k = 2;
	float op_0_alpha = 1.;
	float op_0_beta = .0;
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_feedforward, &op_0_n);
	for (size_t seq = 0; seq < 1; seq++) {
		size_t feedforward_offset = seq * 12;
		size_t h_offset = seq * 12 + 0;
		size_t feedforward_size = 12;
		std::copy(op_0_feedforward + feedforward_offset, op_0_feedforward + feedforward_offset + feedforward_size, op_0_hidden_state + h_offset);
	}
	for (size_t seq = 0; seq < 1; seq++) {
		size_t index = seq;
		int m2 = 3;
		size_t offset = index * 12 + 0;
		size_t size = 12;
		if (seq == 0) {
		} else {
			size_t r_offset = 0;
			size_t previous_offset = (seq - 1) * 12 + 0;
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + r_offset, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_hidden_state + offset, &op_0_n);
		}
		for (size_t i = offset; i < offset + size; i++) {
			float ex = exp(-2 * op_0_hidden_state[i]);
				op_0_hidden_state[i] = (1. - ex) / (1. + ex);
		}
	}
	for (size_t seq = 0; seq < 1; seq++) {
		for (size_t batch = 0; batch < 3; batch++) {
			size_t offset = seq * 12 + 0 + batch * 4;
			size_t y_offset = batch * 4 + seq * 4 + 0;
			std::copy(op_0_hidden_state + offset, op_0_hidden_state + offset + 4, tensor_Y + y_offset);
		}
	}
	for (size_t batch = 0; batch < 3; batch++) {
		size_t seq = 0;
		size_t offset = seq * 12 + batch * 4;
		size_t y_h_offset = batch * 4;
		std::copy(op_0_hidden_state + offset, op_0_hidden_state + offset + 4, tensor_Yh + y_h_offset);
	}
}
} //TMVA_SOFIE_rnn_batchwise
