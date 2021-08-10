//Code generated automatically by TMVA for Inference of Model file [gru_defaults.onnx] at [Sun Aug  8 19:30:11 2021] 
#include<vector>
namespace TMVA_SOFIE_gru_defaults{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_R[75] = {0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001};
float tensor_W[30] = {0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001};
float tensor_Yh[45];
float tensor_Y[45];
// TMVA SOFIE - Warning Model with more than 1 output
infer(float* tensor_X){
	float *op_0_input = tensor_X;
	float op_0_f_update_gate[15];
	float op_0_f_reset_gate[15];
	float op_0_f_hidden_gate[15];
	float op_0_update_gate[15];
	float op_0_reset_gate[15];
	float op_0_hidden_gate[15];
	float *op_0_hidden_state = tensor_Y;
	float op_0_feedback[15];
	char op_0_transA = 'N';
	char op_0_transB = 'T';
	int op_0_m = 3;
	int op_0_m2 = 3;
	int op_0_n = 5;
	int op_0_k = 2;
	float op_0_alpha = 1.;
	float op_0_beta = 0.;
	int op_0_incx = 1;
	int op_0_incy = 1;
	int op_0_feedback_size = 15;
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_f_update_gate, &op_0_n);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W + 10, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_f_reset_gate, &op_0_n);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W + 20, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_f_hidden_gate, &op_0_n);
	for (size_t seq = 0; seq < 1; seq++) {
		size_t offset = seq * 15;
		size_t gate_offset = seq * 15;
		std::copy(op_0_f_update_gate + offset, op_0_f_update_gate + offset + 15, op_0_update_gate + gate_offset);
		std::copy(op_0_f_reset_gate + offset, op_0_f_reset_gate + offset + 15, op_0_reset_gate + gate_offset);
		std::copy(op_0_f_hidden_gate + offset, op_0_f_hidden_gate + offset + 15, op_0_hidden_gate + gate_offset);
	}
	for (size_t seq = 0; seq < 1; seq++) {
		size_t index = seq;
		int m2 = 3;
		size_t offset = index * 15;
		if (seq == 0) {
		} else {
			size_t previous_offset = (seq - 1) * 15;
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_update_gate + offset, &op_0_n);
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + 25, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_reset_gate + offset, &op_0_n);
		}
		for (size_t i = offset; i < offset + 15; i++) {
				op_0_update_gate[i] = 1. / (1. + exp(-op_0_update_gate[i]));
				op_0_reset_gate[i] = 1. / (1. + exp(-op_0_reset_gate[i]));
		}
		if (seq == 0) {
		} else {
			size_t previous_offset = (seq - 1) * 15;
			for (size_t i = 0; i < 15; i++) {
				op_0_feedback[i] = op_0_reset_gate[i + offset] * op_0_hidden_state[i + previous_offset];
			}
		}
		BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m2, &op_0_n, &op_0_alpha, tensor_R + 50, &op_0_n, op_0_feedback, &op_0_n, &op_0_beta, op_0_feedback, &op_0_n);
		BLAS::saxpy_(&op_0_feedback_size, &op_0_alpha, op_0_feedback, &op_0_incx, op_0_hidden_gate + offset, &op_0_incy);
		for (size_t i = offset; i < offset + 15; i++) {
			float ex = exp(-2 * op_0_hidden_gate[i]);
				op_0_hidden_gate[i] = (1. - ex) / (1. + ex);
		}
		for (size_t i = offset; i < offset + 15; i++) {
			op_0_hidden_state[i] = ( 1. - op_0_update_gate[i]) * op_0_hidden_gate[i];
		}
		if (seq == 0) {
		} else {
			size_t previous_offset = (seq - 1) * 15;
			for (size_t i = 0; i < 15; i++) {
				op_0_hidden_state[i + offset] += op_0_update_gate[i + offset] * op_0_hidden_state[i + previous_offset];
			}
		}
	}
	std::copy(op_0_hidden_state + 0, op_0_hidden_state + 0 + 15, tensor_Yh);
}
} //TMVA_SOFIE_gru_defaults
