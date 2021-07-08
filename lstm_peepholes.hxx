//Code generated automatically by TMVA for Inference of Model file [lstm_peepholes.onnx] at [Thu Jul  8 13:29:49 2021] 
#include<vector>
namespace TMVA_SOFIE_lstm_peepholes{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_P[18] = {0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001};
float tensor_initialc[6] = {0, 0, 0, 0, 0, 0};
float tensor_R[36] = {0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001};
float tensor_initialh[6] = {0, 0, 0, 0, 0, 0};
float tensor_sequencelens[2] = {1, 1};
float tensor_B[24] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
float tensor_W[48] = {0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001};
float tensor_Yh[24];
float tensor_Y[24];
// TMVA SOFIE - Warning Model with more than 1 output
void infer(float* tensor_X){
	float *op_0_input = tensor_X;
	float *op_0_initial_hidden_state =  tensor_initialh;
	float *op_0_initial_cell_state =  tensor_initialc;
	float op_0_ff_input_gate[6];
	float op_0_ff_output_gate[6];
	float op_0_ff_cell_gate[6];
	float op_0_ff_forget_gate[6];
	float op_0_input_gate[6];
	float op_0_output_gate[6];
	float op_0_cell_gate[6];
	float op_0_forget_gate[6];
	float op_0_cell_state[6];
	float op_0_new_cell_state[6];
	float *op_0_hidden_state = tensor_Y;
	char op_0_transA = 'N';
	char op_0_transB = 'T';
	int op_0_m = 2;
	int op_0_n = 3;
	int op_0_k = 4;
	float op_0_alpha = 1.;
	float op_0_beta = 0.;
	int op_0_bias_size = 6;
	int op_0_incx = 1;
	int op_0_incy = 1;
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_input_gate, &op_0_n);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W + 24, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_output_gate, &op_0_n);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W + 36, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_cell_gate, &op_0_n);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W + 12, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_forget_gate, &op_0_n);
	BLAS::saxpy_(&op_0_bias_size, &op_0_alpha, tensor_B, &op_0_incx, op_0_ff_input_gate, &op_0_incy);
	BLAS::saxpy_(&op_0_bias_size, &op_0_alpha, tensor_B + 12, &op_0_incx, op_0_ff_output_gate, &op_0_incy);
	BLAS::saxpy_(&op_0_bias_size, &op_0_alpha, tensor_B + 18, &op_0_incx, op_0_ff_cell_gate, &op_0_incy);
	BLAS::saxpy_(&op_0_bias_size, &op_0_alpha, tensor_B + 6, &op_0_incx, op_0_ff_forget_gate, &op_0_incy);
	for (size_t seq = 0; seq < 1; seq++) {
		size_t ff_offset = seq * 6;
		size_t gate_offset = seq * 6;
		std::copy(op_0_ff_input_gate + ff_offset, op_0_ff_input_gate + ff_offset + 6, op_0_input_gate + gate_offset);
		std::copy(op_0_ff_output_gate + ff_offset, op_0_ff_output_gate + ff_offset + 6, op_0_output_gate + gate_offset);
		std::copy(op_0_ff_cell_gate + ff_offset, op_0_ff_cell_gate + ff_offset + 6, op_0_cell_gate + gate_offset);
		std::copy(op_0_ff_forget_gate + ff_offset, op_0_ff_forget_gate + ff_offset + 6, op_0_forget_gate + gate_offset);
	}
	for (size_t seq = 0; seq < 1; seq++) {
		size_t index = seq;
		int m2 = 2;
		size_t offset = index * 6;
		if (seq == 0) {
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R, &op_0_n, op_0_initial_hidden_state, &op_0_n, &op_0_alpha, op_0_input_gate + offset, &op_0_n);
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + 18, &op_0_n, op_0_initial_hidden_state, &op_0_n, &op_0_alpha, op_0_output_gate + offset, &op_0_n);
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + 27, &op_0_n, op_0_initial_hidden_state, &op_0_n, &op_0_alpha, op_0_cell_gate + offset, &op_0_n);
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + 9, &op_0_n, op_0_initial_hidden_state, &op_0_n, &op_0_alpha, op_0_forget_gate + offset, &op_0_n);
		} else {
			size_t previous_offset = (seq - 1) * 6;
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_input_gate + offset, &op_0_n);
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + 18, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_output_gate + offset, &op_0_n);
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + 27, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_cell_gate + offset, &op_0_n);
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + 9, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_forget_gate + offset, &op_0_n);
		}
		for (size_t i = offset; i < offset + 6; i++) {
			float ex = exp(-2 * op_0_cell_gate[i]);
				op_0_cell_gate[i] = (1. - ex) / (1. + ex);
		}
		if (seq == 0) {
			for (size_t i = 0; i < 6; i++) {
				op_0_input_gate[i + offset] += tensor_P[i] * op_0_initial_cell_state[i];
			}
			for (size_t i = 0; i < 6; i++) {
				op_0_forget_gate[i + offset] += tensor_P[i + 6] * op_0_initial_cell_state[i];
			}
		} else {
			size_t c_offset = (seq - 1) * 6;
			for (size_t i = 0; i < 6; i++) {
				op_0_input_gate[i + offset] += tensor_P[i] * op_0_cell_state[i + c_offset];
			}
			for (size_t i = 0; i < 6; i++) {
				op_0_forget_gate[i + offset] += tensor_P[i + 6] * op_0_cell_state[i + c_offset];
			}
		}
		for (size_t i = offset; i < offset + 6; i++) {
				op_0_input_gate[i] = 1. / (1. + exp(-op_0_input_gate[i]));
		}
		for (size_t i = offset; i < offset + 6; i++) {
				op_0_forget_gate[i] = 1. / (1. + exp(-op_0_forget_gate[i]));
		}
		for (size_t i = offset; i < offset + 6; i++) {
			op_0_cell_state[i] = op_0_input_gate[i] * op_0_cell_gate[i];
		}
		if (seq == 0) {
			for (size_t i = 0; i < 6; i++) {
				op_0_cell_state[i + offset] += op_0_forget_gate[i + offset] * op_0_initial_cell_state[i];
			}
		} else {
			size_t previous_offset = (seq - 1) * 6;
			for (size_t i = 0; i < 6; i++) {
				op_0_cell_state[i + offset] += op_0_forget_gate[i + offset] * op_0_cell_state[i + previous_offset];
			}
		}
			for (size_t i = 0; i < 6; i++) {
				op_0_output_gate[i + offset] += tensor_P[i + 12] * op_0_cell_state[i + offset];
			}
		for (size_t i = offset; i < offset + 6; i++) {
				op_0_output_gate[i] = 1. / (1. + exp(-op_0_output_gate[i]));
		}
		std::copy(op_0_cell_state + offset, op_0_cell_state + offset + 6, op_0_new_cell_state + offset);
		for (size_t i = offset; i < offset + 6; i++) {
			float ex = exp(-2 * op_0_new_cell_state[i]);
				op_0_new_cell_state[i] = (1. - ex) / (1. + ex);
		}
		for (size_t i = offset; i < offset + 6; i++) {
			op_0_hidden_state[i] = op_0_output_gate[i] * op_0_new_cell_state[i];
		}
	}
	for (size_t seq = 0; seq < 1; seq++) {
		for (size_t batch = 0; batch < 2; batch++) {
			if (seq >= tensor_sequencelens[batch]) {
					for (size_t h = 0; h < 3; h++) {
						size_t idx = seq * 6 + batch * 3 + h;
						op_0_cell_state[idx] = 0.;
						op_0_hidden_state[idx] = 0.;
					}
			}
		}
	}
	for (size_t batch = 0; batch < 2; batch++) {
		size_t seq = tensor_sequencelens[batch] - 1;
		size_t offset = seq * 6 + batch * 3;
		size_t y_h_offset = batch * 3;
		std::copy(op_0_hidden_state + offset, op_0_hidden_state + offset + 3, tensor_Yh + y_h_offset);
	}
}
} //TMVA_SOFIE_lstm_peepholes
