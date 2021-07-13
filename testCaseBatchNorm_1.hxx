//Code generated automatically by TMVA for Inference of Model file [testCaseBatchNorm_1.onnx] at [Tue Jul 13 13:13:48 2021] 
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
float tensor_var[3] = {0.801017821, 0.650941014, 0.502549887};
float tensor_mean[3] = {0.610773981, 0.373008311, 1.85429156};
float tensor_bias[3] = {-0.844192743, -1.79882622, -0.302479535};
float tensor_s[3] = {2.34357285, 0.356457889, 1.21757233};
float tensor_y[120];
std::vector<float> infer(float* tensor_x){
	for (size_t n = 0; n < 2; n++) {
		for (size_t c = 0; c < 3; c++) {
			for (size_t h = 0; h < 4; h++) {
				for (size_t w = 0; w < 5; w++) {
					tensor_y[n * 60 + c * 20 + h * 5 + w] = ((tensor_x[n * 60 + c * 20 + h * 5 + w] - tensor_mean[c])/ std::sqrt(tensor_var[c] + 1e-05 ))* tensor_s[c] + tensor_bias[c];
				}
			}
		}
	}
	std::vector<float> ret (tensor_y, tensor_y + sizeof(tensor_y) / sizeof(tensor_y[0]));
	return ret;
}
} //TMVA_SOFIE_testCaseBatchNorm_1
