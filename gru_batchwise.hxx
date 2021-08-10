//Code generated automatically by TMVA for Inference of Model file [gru_batchwise.onnx] at [Sun Aug  8 19:30:11 2021] 
#include<vector>
namespace TMVA_SOFIE_gru_batchwise{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_R[108] = {0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003};
float tensor_W[36] = {0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003, 0.200000003};
float tensor_Yh[54];
float tensor_Y[54];
// TMVA SOFIE - Warning Model with more than 1 output
infer(float* tensor_X){
	float *op_0_input = tensor_X;
	float op_0_f_update_gate[18];
	float op_0_f_reset_gate[18];
	float op_0_f_hidden_gate[18];
	float op_0_update_gate[18];
	float op_0_reset_gate[18];
	float op_0_hidden_gate[18];
	float *op_0_hidden_state = tensor_Y;
	float op_0_feedback[18];
	char op_0_transA = 'N';
	char op_0_transB = 'T';
	int op_0_m = 3;
	int op_0_m2 = 3;
	int op_0_n = 6;
	int op_0_k = 2;
	float op_0_alpha = 1.;
	float op_0_beta = 0.;
	int op_0_incx = 1;
	int op_0_incy = 1;
	int op_0_feedback_size = 18;
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_f_update_gate, &op_0_n);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W + 12, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_f_reset_gate, &op_0_n);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W + 24, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_f_hidden_gate, &op_0_n);
	for (size_t seq = 0; seq < 1; seq++) {
		size_t offset = seq * 18;
		size_t gate_offset = seq * 18;
		std::copy(op_0_f_update_gate + offset, op_0_f_update_gate + offset + 18, op_0_update_gate + gate_offset);
		std::copy(op_0_f_reset_gate + offset, op_0_f_reset_gate + offset + 18, op_0_reset_gate + gate_offset);
		std::copy(op_0_f_hidden_gate + offset, op_0_f_hidden_gate + offset + 18, op_0_hidden_gate + gate_offset);
	}
	for (size_t seq = 0; seq < 1; seq++) {
		size_t index = seq;
		int m2 = 3;
		size_t offset = index * 18;
		if (seq == 0) {
		} else {
			size_t previous_offset = (seq - 1) * 18;
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_update_gate + offset, &op_0_n);
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + 36, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_reset_gate + offset, &op_0_n);
		}
		for (size_t i = offset; i < offset + 18; i++) {
				op_0_update_gate[i] = 1. / (1. + exp(-op_0_update_gate[i]));
				op_0_reset_gate[i] = 1. / (1. + exp(-op_0_reset_gate[i]));
		}
		if (seq == 0) {
		} else {
			size_t previous_offset = (seq - 1) * 18;
			for (size_t i = 0; i < 18; i++) {
				op_0_feedback[i] = op_0_reset_gate[i + offset] * op_0_hidden_state[i + previous_offset];
			}
		}
		BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m2, &op_0_n, &op_0_alpha, tensor_R + 72, &op_0_n, op_0_feedback, &op_0_n, &op_0_beta, op_0_feedback, &op_0_n);
		BLAS::saxpy_(&op_0_feedback_size, &op_0_alpha, op_0_feedback, &op_0_incx, op_0_hidden_gate + offset, &op_0_incy);
		for (size_t i = offset; i < offset + 18; i++) {
			float ex = exp(-2 * op_0_hidden_gate[i]);
				op_0_hidden_gate[i] = (1. - ex) / (1. + ex);
		}
		for (size_t i = offset; i < offset + 18; i++) {
			op_0_hidden_state[i] = ( 1. - op_0_update_gate[i]) * op_0_hidden_gate[i];
		}
		if (seq == 0) {
		} else {
			size_t previous_offset = (seq - 1) * 18;
			for (size_t i = 0; i < 18; i++) {
				op_0_hidden_state[i + offset] += op_0_update_gate[i + offset] * op_0_hidden_state[i + previous_offset];
			}
		}
	}
	std::copy(op_0_hidden_state + 0, op_0_hidden_state + 0 + 18, tensor_Yh);
}
} //TMVA_SOFIE_gru_batchwise
