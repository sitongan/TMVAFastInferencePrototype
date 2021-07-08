//Code generated automatically by TMVA for Inference of Model file [lstm_batchwise.onnx] at [Thu Jul  8 13:29:49 2021] 
#include<vector>
namespace TMVA_SOFIE_lstm_batchwise{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_R[196] = {0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012};
float tensor_W[56] = {0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012, 0.300000012};
float tensor_Yh[84];
float tensor_Y[84];
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
	float op_0_ff_input_gate[21];
	float op_0_ff_output_gate[21];
	float op_0_ff_cell_gate[21];
	float op_0_ff_forget_gate[21];
	float op_0_input_gate[21];
	float op_0_output_gate[21];
	float op_0_cell_gate[21];
	float op_0_forget_gate[21];
	float op_0_cell_state[21];
	float op_0_new_cell_state[21];
	float op_0_hidden_state[21];
	char op_0_transA = 'N';
	char op_0_transB = 'T';
	int op_0_m = 3;
	int op_0_n = 7;
	int op_0_k = 2;
	float op_0_alpha = 1.;
	float op_0_beta = 0.;
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_input_gate, &op_0_n);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W + 28, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_output_gate, &op_0_n);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W + 42, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_cell_gate, &op_0_n);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W + 14, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_forget_gate, &op_0_n);
	for (size_t seq = 0; seq < 1; seq++) {
		size_t ff_offset = seq * 21;
		size_t gate_offset = seq * 21;
		std::copy(op_0_ff_input_gate + ff_offset, op_0_ff_input_gate + ff_offset + 21, op_0_input_gate + gate_offset);
		std::copy(op_0_ff_output_gate + ff_offset, op_0_ff_output_gate + ff_offset + 21, op_0_output_gate + gate_offset);
		std::copy(op_0_ff_cell_gate + ff_offset, op_0_ff_cell_gate + ff_offset + 21, op_0_cell_gate + gate_offset);
		std::copy(op_0_ff_forget_gate + ff_offset, op_0_ff_forget_gate + ff_offset + 21, op_0_forget_gate + gate_offset);
	}
	for (size_t seq = 0; seq < 1; seq++) {
		size_t index = seq;
		int m2 = 3;
		size_t offset = index * 21;
		if (seq == 0) {
		} else {
			size_t previous_offset = (seq - 1) * 21;
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_input_gate + offset, &op_0_n);
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + 98, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_output_gate + offset, &op_0_n);
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + 147, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_cell_gate + offset, &op_0_n);
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + 49, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_forget_gate + offset, &op_0_n);
		}
		for (size_t i = offset; i < offset + 21; i++) {
			float ex = exp(-2 * op_0_cell_gate[i]);
				op_0_cell_gate[i] = (1. - ex) / (1. + ex);
		}
		for (size_t i = offset; i < offset + 21; i++) {
				op_0_input_gate[i] = 1. / (1. + exp(-op_0_input_gate[i]));
		}
		for (size_t i = offset; i < offset + 21; i++) {
				op_0_forget_gate[i] = 1. / (1. + exp(-op_0_forget_gate[i]));
		}
		for (size_t i = offset; i < offset + 21; i++) {
			op_0_cell_state[i] = op_0_input_gate[i] * op_0_cell_gate[i];
		}
		if (seq == 0) {
		} else {
			size_t previous_offset = (seq - 1) * 21;
			for (size_t i = 0; i < 21; i++) {
				op_0_cell_state[i + offset] += op_0_forget_gate[i + offset] * op_0_cell_state[i + previous_offset];
			}
		}
		for (size_t i = offset; i < offset + 21; i++) {
				op_0_output_gate[i] = 1. / (1. + exp(-op_0_output_gate[i]));
		}
		std::copy(op_0_cell_state + offset, op_0_cell_state + offset + 21, op_0_new_cell_state + offset);
		for (size_t i = offset; i < offset + 21; i++) {
			float ex = exp(-2 * op_0_new_cell_state[i]);
				op_0_new_cell_state[i] = (1. - ex) / (1. + ex);
		}
		for (size_t i = offset; i < offset + 21; i++) {
			op_0_hidden_state[i] = op_0_output_gate[i] * op_0_new_cell_state[i];
		}
	}
	for (size_t seq = 0; seq < 1; seq++) {
		for (size_t batch = 0; batch < 3; batch++) {
			size_t offset = seq * 21 + 0 + batch * 7;
			size_t y_offset = batch * 7 + seq * 7 + 0;
			std::copy(op_0_hidden_state + offset, op_0_hidden_state + offset + 7, tensor_Y + y_offset);
		}
	}
	for (size_t batch = 0; batch < 3; batch++) {
		size_t seq = 0;
		size_t offset = seq * 21 + batch * 7;
		size_t y_h_offset = batch * 7;
		std::copy(op_0_hidden_state + offset, op_0_hidden_state + offset + 7, tensor_Yh + y_h_offset);
	}
}
} //TMVA_SOFIE_lstm_batchwise
