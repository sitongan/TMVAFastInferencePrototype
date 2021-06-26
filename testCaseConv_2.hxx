//Code generated automatically by TMVA for Inference of Model file [testCaseConv_2.onnx] at [Sat Jun 26 20:12:57 2021] 
#include<vector>
namespace TMVA_SOFIE_testCaseConv_2{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_W[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
float tensor_y[8];
std::vector<float> infer(float* tensor_x){
	float op_0_xpad[45] = {0};
	for (size_t n = 0; n < 1; n++) {
		for (size_t c = 0; c < 1; c++) {
			for (size_t h = 0; h < 7; h++) {
				for (size_t w = 0; w < 5; w++) {
					op_0_xpad[n * 45 + c * 45 + (h + 1) * 5 + w + 0] = tensor_x[n * 35 + c * 35 + h * 5 + w];
				}
			}
		}
	}
	float op_0_xcol[72] = {0};
	for (size_t g = 0; g < 1; g++) {
		size_t idx = g * 9;
		for (size_t n = 0; n < 1; n++) {
			for (size_t c = g * 1; c < (g + 1) * 1; c++) {
				for (size_t h = 0; h < 7; h += 2) {
					for (size_t w = 0; w < 3;w += 2) {
						for (size_t x = 0; x < 3; x++) {
							for (size_t y = 0; y < 3; y++) {
								op_0_xcol[idx] = op_0_xpad[n * 45 + c * 45 + (h + x) * 5 + w + y];
								idx++;
							}
						}
					}
				}
			}
		}
	}
	float op_0_f[9] = {0};
	for (std::size_t k = 0; k < 1; k++) {
		for (std::size_t d = 0; d < 1; d++) {
			for (std::size_t h = 0; h < 3; h++) {
				for (std::size_t w = 0; w < 3; w++) {
					op_0_f[k + (d * 9 + h * 3 + w * 1) * 1] = tensor_W[k * 9 + d * 9 + h * 3 + w ];
				}
			}
		}
	}
	char op_0_transF = 'N';
	char op_0_transXcol = 'N';
	int op_0_m = 1;
	int op_0_n = 8;
	int op_0_k = 9;
	float op_0_alpha = 1.0;
	float op_0_beta = 0.0;
	BLAS::sgemm_(&op_0_transF, &op_0_transXcol, &op_0_m, &op_0_n, &op_0_k, &op_0_alpha, op_0_f, &op_0_m,
		op_0_xcol, &op_0_k, &op_0_beta, tensor_y, &op_0_m);
	std::vector<float> ret (tensor_y, tensor_y + sizeof(tensor_y) / sizeof(tensor_y[0]));
	return ret;
}
} //TMVA_SOFIE_testCaseConv_2
