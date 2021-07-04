//Code generated automatically by TMVA for Inference of Model file [rnn_sequence_batchwise.onnx] at [Sun Jul  4 18:07:17 2021] 
#include<string>
#include<vector>
namespace TMVA_SOFIE_rnn_sequence_batchwise{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_sequencelens[3] = {3, 2, 1};
float tensor_B[54] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
float tensor_R[36] = {1.01349998, -0.263200015, -0.678600013, -1.01789999, -2.13190007, -0.00359999994, 1.95850003, 1.13750005, 2.12100005, 0.640900016, -2.05029988, -2.4921, 0.593200028, 1.51610005, -0.776899993, 0.28490001, 0.207200006, -0.308600008, -0.965499997, 0.917800009, -0.429199994, -1.50539994, -0.739600003, 0.89289999, -0.183599994, -1.62919998, 1.07120001, 0.377000004, 0.177900001, -1.11670005, -0.686100006, 1.23909998, -0.544799984, -0.388099998, -0.516499996, 0.0127999997};
float tensor_W[30] = {0.236900002, 0.134599999, 0.331699997, -0.482199997, -0.136299998, 0.941999972, -0.450199991, -2.81739998, 0.288899988, 1.67149997, 0.296700001, 1.67990005, -0.834299982, 0.449299991, 0.0370000005, -0.532599986, 1.15450001, -1.64779997, 0.777899981, -0.925700009, -1.48220003, -0.871699989, -0.0174000002, 2.06850004, -0.762000024, 0.0104999999, -2.93779993, 0.888700008, -0.947799981, -1.57249999};
float tensor_Yh[18];
float tensor_Y[54];
// TMVA SOFIE - Warning Model with more than 1 output
void infer(float* tensor_X){
	float op_0_input[45];
	for(size_t seq = 0; seq < 3; seq++) {
		for(size_t batch = 0; batch < 3; batch++) {
			for(size_t i = 0; i < 5; i++) {
				op_0_input[seq * 15 + batch * 5 + i] = tensor_X[batch * 15 + seq * 5 + i];
			}
		}
	}
	float op_0_feedforward[54];
	float op_0_hidden_state[54];
	char op_0_transA = 'N';
	char op_0_transB = 'T';
	int op_0_m = 9;
	int op_0_n = 6;
	int op_0_k = 5;
	float op_0_alpha = 1.;
	float op_0_beta = .0;
	int op_0_bias_size = 54;
	int op_0_incx = 1;
	int op_0_incy = 1;
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_feedforward, &op_0_n);
	BLAS::saxpy_(&op_0_bias_size, &op_0_alpha, tensor_B, &op_0_incx, op_0_feedforward, &op_0_incy);
	for (size_t seq = 0; seq < 3; seq++) {
		size_t feedforward_offset = seq * 18;
		size_t h_offset = seq * 18 + 0;
		size_t feedforward_size = 18;
		std::copy(op_0_feedforward + feedforward_offset, op_0_feedforward + feedforward_offset + feedforward_size, op_0_hidden_state + h_offset);
	}
	for (size_t seq = 0; seq < 3; seq++) {
		size_t index = seq;
		int m2 = 3;
		size_t offset = index * 18 + 0;
		size_t size = 18;
		if (seq == 0) {
		} else {
			size_t r_offset = 0;
			size_t previous_offset = (seq - 1) * 18 + 0;
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + r_offset, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_hidden_state + offset, &op_0_n);
		}
		for (size_t i = offset; i < offset + size; i++) {
			float ex = exp(-2 * op_0_hidden_state[i]);
				op_0_hidden_state[i] = (1. - ex) / (1. + ex);
		}
	}
	for (size_t seq = 0; seq < 3; seq++) {
		for (size_t batch = 0; batch < 3; batch++) {
			if (seq >= tensor_sequencelens[batch]) {
					for (size_t h = 0; h < 6; h++) {
						op_0_hidden_state[seq * 18 + batch * 6 + h] = 0.;
					}
			}
		}
	}
	for (size_t seq = 0; seq < 3; seq++) {
		for (size_t batch = 0; batch < 3; batch++) {
			size_t offset = seq * 18 + 0 + batch * 6;
			size_t y_offset = batch * 18 + seq * 6 + 0;
			std::copy(op_0_hidden_state + offset, op_0_hidden_state + offset + 6, tensor_Y + y_offset);
		}
	}
	for (size_t batch = 0; batch < 3; batch++) {
		size_t seq = tensor_sequencelens[batch] - 1;
		size_t offset = seq * 18 + batch * 6;
		size_t y_h_offset = batch * 6;
		std::copy(op_0_hidden_state + offset, op_0_hidden_state + offset + 6, tensor_Yh + y_h_offset);
	}
}
} //TMVA_SOFIE_rnn_sequence_batchwise
