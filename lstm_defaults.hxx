//Code generated automatically by TMVA for Inference of Model file [lstm_defaults.onnx] at [Thu Jul  8 13:29:49 2021] 
#include<vector>
namespace TMVA_SOFIE_lstm_defaults{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_R[36] = {0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001};
float tensor_W[24] = {0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001};
float tensor_Yh[12];
float tensor_Y[36];
// TMVA SOFIE - Warning Model with more than 1 output
void infer(float* tensor_X){
	float *op_0_input = tensor_X;
	float op_0_ff_input_gate[9];
	float op_0_ff_output_gate[9];
	float op_0_ff_cell_gate[9];
	float op_0_ff_forget_gate[9];
	float op_0_input_gate[9];
	float op_0_output_gate[9];
	float op_0_cell_gate[9];
	float op_0_forget_gate[9];
	float op_0_cell_state[9];
	float op_0_new_cell_state[9];
	float *op_0_hidden_state = tensor_Y;
	char op_0_transA = 'N';
	char op_0_transB = 'T';
	int op_0_m = 3;
	int op_0_n = 3;
	int op_0_k = 2;
	float op_0_alpha = 1.;
	float op_0_beta = 0.;
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_input_gate, &op_0_n);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W + 12, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_output_gate, &op_0_n);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W + 18, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_cell_gate, &op_0_n);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W + 6, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_forget_gate, &op_0_n);
	for (size_t seq = 0; seq < 3; seq++) {
		size_t ff_offset = seq * 3;
		size_t gate_offset = seq * 3;
		std::copy(op_0_ff_input_gate + ff_offset, op_0_ff_input_gate + ff_offset + 3, op_0_input_gate + gate_offset);
		std::copy(op_0_ff_output_gate + ff_offset, op_0_ff_output_gate + ff_offset + 3, op_0_output_gate + gate_offset);
		std::copy(op_0_ff_cell_gate + ff_offset, op_0_ff_cell_gate + ff_offset + 3, op_0_cell_gate + gate_offset);
		std::copy(op_0_ff_forget_gate + ff_offset, op_0_ff_forget_gate + ff_offset + 3, op_0_forget_gate + gate_offset);
	}
	for (size_t seq = 0; seq < 3; seq++) {
		size_t index = seq;
		int m2 = 1;
		size_t offset = index * 3;
		if (seq == 0) {
		} else {
			size_t previous_offset = (seq - 1) * 3;
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_input_gate + offset, &op_0_n);
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + 18, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_output_gate + offset, &op_0_n);
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + 27, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_cell_gate + offset, &op_0_n);
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + 9, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_forget_gate + offset, &op_0_n);
		}
		for (size_t i = offset; i < offset + 3; i++) {
			float ex = exp(-2 * op_0_cell_gate[i]);
				op_0_cell_gate[i] = (1. - ex) / (1. + ex);
		}
		for (size_t i = offset; i < offset + 3; i++) {
				op_0_input_gate[i] = 1. / (1. + exp(-op_0_input_gate[i]));
		}
		for (size_t i = offset; i < offset + 3; i++) {
				op_0_forget_gate[i] = 1. / (1. + exp(-op_0_forget_gate[i]));
		}
		for (size_t i = offset; i < offset + 3; i++) {
			op_0_cell_state[i] = op_0_input_gate[i] * op_0_cell_gate[i];
		}
		if (seq == 0) {
		} else {
			size_t previous_offset = (seq - 1) * 3;
			for (size_t i = 0; i < 3; i++) {
				op_0_cell_state[i + offset] += op_0_forget_gate[i + offset] * op_0_cell_state[i + previous_offset];
			}
		}
		for (size_t i = offset; i < offset + 3; i++) {
				op_0_output_gate[i] = 1. / (1. + exp(-op_0_output_gate[i]));
		}
		std::copy(op_0_cell_state + offset, op_0_cell_state + offset + 3, op_0_new_cell_state + offset);
		for (size_t i = offset; i < offset + 3; i++) {
			float ex = exp(-2 * op_0_new_cell_state[i]);
				op_0_new_cell_state[i] = (1. - ex) / (1. + ex);
		}
		for (size_t i = offset; i < offset + 3; i++) {
			op_0_hidden_state[i] = op_0_output_gate[i] * op_0_new_cell_state[i];
		}
	}
	std::copy(op_0_hidden_state + 6, op_0_hidden_state + 6 + 3, tensor_Yh);
}
} //TMVA_SOFIE_lstm_defaults
