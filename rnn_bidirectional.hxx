//Code generated automatically by TMVA for Inference of Model file [rnn_bidirectional.onnx] at [Sat Jul 31 21:26:10 2021] 
#include<vector>
namespace TMVA_SOFIE_rnn_bidirectional{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_initialh[24] = {-0.371075004, 0.252532989, -1.42194998, 0.393029988, -0.463111997, -1.02437997, -0.538398981, -2.21508002, -1.42209995, -0.149364993, 1.25870001, 1.38294005, -0.0841611996, 1.45696998, 0.0679387003, 2.11547995, -1.51050997, 1.50948, 0.206350997, -0.981445014, -0.221477002, -0.230483994, 0.453312993, 0.795476019};
float tensor_sequencelens[3] = {3, 3, 3};
float tensor_B[72] = {-0.111975998, -4.4878602, 0.0126000047, -0.476267993, -0.111975998, -4.4878602, 0.0126000047, -0.476267993, -0.111975998, -4.4878602, 0.0126000047, -0.476267993, -0.111975998, -4.4878602, 0.0126000047, -0.476267993, -0.111975998, -4.4878602, 0.0126000047, -0.476267993, -0.111975998, -4.4878602, 0.0126000047, -0.476267993, -0.111975998, -4.4878602, 0.0126000047, -0.476267993, -0.111975998, -4.4878602, 0.0126000047, -0.476267993, -0.111975998, -4.4878602, 0.0126000047, -0.476267993, 1.81043601, 0.377670944, -1.78106797, -0.725206077, 1.81043601, 0.377670944, -1.78106797, -0.725206077, 1.81043601, 0.377670944, -1.78106797, -0.725206077, 1.81043601, 0.377670944, -1.78106797, -0.725206077, 1.81043601, 0.377670944, -1.78106797, -0.725206077, 1.81043601, 0.377670944, -1.78106797, -0.725206077, 1.81043601, 0.377670944, -1.78106797, -0.725206077, 1.81043601, 0.377670944, -1.78106797, -0.725206077, 1.81043601, 0.377670944, -1.78106797, -0.725206077};
float tensor_R[32] = {-0.264847994, -1.30311, 0.0712087005, 0.641979992, -2.76537991, -0.652073979, -0.784274995, -1.76749003, -0.450673014, -0.917928994, -0.966654003, 0.650856018, 0.285537988, -0.909847975, -1.90459001, -0.140926003, -1.37131, 0.780644, 0.441009015, 1.15856004, 0.313297987, 1.96765995, -1.11991, -0.0044095898, 0.40762201, 2.60569, -0.840986013, 0.585658014, 0.823292017, -0.696817994, 1.15114999, 0.150269002};
float tensor_W[16] = {1.16307998, 2.21220994, 0.483805001, 0.774003983, 0.299562991, 1.04343998, 0.153025001, 1.18393004, -1.16881001, 1.89171004, 1.55806994, -1.23474002, -0.545944989, -1.77102995, -2.35562992, -0.451384008};
float tensor_Yh[24];
float tensor_Y[72];
// TMVA SOFIE - Warning Model with more than 1 output
void infer(float* tensor_X){
	float *op_0_input = tensor_X;
	float *op_0_initial_hidden_state =  tensor_initialh;
	float op_0_feedforward[36];
	float *op_0_hidden_state = tensor_Y;
	char op_0_transA = 'N';
	char op_0_transB = 'T';
	int op_0_m = 9;
	int op_0_n = 4;
	int op_0_k = 2;
	float op_0_alpha = 1.;
	float op_0_beta = .0;
	int op_0_bias_size = 36;
	int op_0_incx = 1;
	int op_0_incy = 1;
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_feedforward, &op_0_n);
	BLAS::saxpy_(&op_0_bias_size, &op_0_alpha, tensor_B, &op_0_incx, op_0_feedforward, &op_0_incy);
	for (size_t seq = 0; seq < 3; seq++) {
		size_t offset = seq * 12;
		size_t size = 12;
		size_t h_offset = seq * 24 + 0;
		std::copy(op_0_feedforward + offset, op_0_feedforward + offset + size, op_0_hidden_state + h_offset);
	}
	for (size_t seq = 0; seq < 3; seq++) {
		size_t index = seq;
		int m2 = 3;
		size_t offset = index * 24 + 0;
		size_t size = 12;
		if (seq == 0) {
			size_t r_offset = 0;
			size_t initial_hidden_state_offset = 0;
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + r_offset, &op_0_n, op_0_initial_hidden_state + initial_hidden_state_offset, &op_0_n, &op_0_alpha, op_0_hidden_state + offset, &op_0_n);
		} else {
			size_t r_offset = 0;
			size_t previous_offset = (seq - 1) * 24 + 0;
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + r_offset, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_hidden_state + offset, &op_0_n);
		}
		for (size_t i = offset; i < offset + size; i++) {
			float ex = exp(-2 * op_0_hidden_state[i]);
				op_0_hidden_state[i] = (1. - ex) / (1. + ex);
		}
	}
	size_t op_0_w_offset = 8;
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W + op_0_w_offset, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_feedforward, &op_0_n);
	size_t op_0_bias_offset = 36;
	BLAS::saxpy_(&op_0_bias_size, &op_0_alpha, tensor_B + op_0_bias_offset, &op_0_incx, op_0_feedforward, &op_0_incy);
	for (size_t seq = 0; seq < 3; seq++) {
		size_t offset = seq * 12;
		size_t size = 12;
		size_t h_offset = seq * 24 + 12;
		std::copy(op_0_feedforward + offset, op_0_feedforward + offset + size, op_0_hidden_state + h_offset);
	}
	for (size_t seq = 0; seq < 3; seq++) {
		size_t index = 2 - seq;
		int m2 = 3;
		size_t offset = index * 24 + 12;
		size_t size = 12;
		if (seq == 0) {
			size_t r_offset = 16;
			size_t initial_hidden_state_offset = 12;
			BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + r_offset, &op_0_n, op_0_initial_hidden_state + initial_hidden_state_offset, &op_0_n, &op_0_alpha, op_0_hidden_state + offset, &op_0_n);
		} else {
			size_t r_offset = 16;
			size_t previous_offset = (index + 1) * 24 + 12;
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
					for (size_t h = 0; h < 4; h++) {
						op_0_hidden_state[seq * 24 + batch * 4 + h] = 0.;
					}
					for (size_t h = 0; h < 4; h++) {
						op_0_hidden_state[seq * 36 + batch * 4 + h] = 0.;
					}
			}
		}
	}
	for (size_t batch = 0; batch < 3; batch++) {
		size_t seq = tensor_sequencelens[batch] - 1;
		size_t offset = seq * 24 + batch * 4;
		size_t yh_offset = batch * 4;
		std::copy(op_0_hidden_state + offset, op_0_hidden_state + offset + 4, tensor_Yh + yh_offset);
	}
	for (size_t batch = 0; batch < 3; batch++) {
		size_t offset = 12 + batch * 4;
		size_t yh_offset = 12 + batch * 4;
		std::copy(op_0_hidden_state + offset, op_0_hidden_state + offset + 4, tensor_Yh + yh_offset);
	}
}
} //TMVA_SOFIE_rnn_bidirectional
