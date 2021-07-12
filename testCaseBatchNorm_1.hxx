//Code generated automatically by TMVA for Inference of Model file [testCaseBatchNorm_1.onnx] at [Mon Jul 12 20:08:52 2021] 
#include<cmath>
#include<vector>
namespace TMVA_SOFIE_testCaseBatchNorm_1{
namespace BLAS{
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
}//BLAS
float tensor_var[3] = {0.478634357, 0.322660893, 0.727022946};
float tensor_mean[3] = {-0.307919502, -0.691410065, 0.627395153};
float tensor_bias[3] = {-0.0293571278, 0.105456114, 0.146158263};
float tensor_s[3] = {0.709476411, 0.877527714, 0.309638649};
float tensor_y[120];
std::vector<float> infer(float* tensor_x){
	for (size_t n = 0; n < 2; n++) {
		for (size_t c = 0; c < 3; c++) {
			for (size_t h = 0; h < 4; h++) {
				for (size_t w = 0; w < 5; w++) {
					tensor_y[n * 60 + c * 20 + h * 5 + w] = ((tensor_x[n * 60 + c * 20 + h * 5 + w] - tensor_mean[c])/ std::sqrt(tensor_var[c]) + 1e-05 ) * tensor_s[c] + tensor_bias[c];
				}
			}
		}
	}
	std::vector<float> ret (tensor_y, tensor_y + sizeof(tensor_y) / sizeof(tensor_y[0]));
	return ret;
}
} //TMVA_SOFIE_testCaseBatchNorm_1
