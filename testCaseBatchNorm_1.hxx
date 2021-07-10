//Code generated automatically by TMVA for Inference of Model file [testCaseBatchNorm_1.onnx] at [Sat Jul 10 06:53:59 2021] 
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
float tensor_var[3] = {0.221740454, 0.116114967, 0.341743618};
float tensor_mean[3] = {1.05077136, -0.772704065, 0.282667071};
float tensor_bias[3] = {-0.978066981, 0.852475882, -1.56837988};
float tensor_s[3] = {1.57780123, -0.12694189, 0.912706256};
float tensor_y[120];
std::vector<float> infer(float* tensor_x){
	for (size_t n = 0; n < 2; n++) {
		for (size_t c = 0; c < 3; c++) {
			for (size_t h = 0; h < 4; h++) {
				for (size_t w = 0; w < 5; w++) {
					tensor_y[n * 60 + c * 20 + h * 5 + w] = ((tensor_x[n * 60 + c * 20 + h * 5 + w] - tensor_mean[c * 20])/ std::sqrt(tensor_var[c * 20]) + 1e-05 ) * tensor_bias[c * 20] + tensor_s[c * 20];
				}
			}
		}
	}
	std::vector<float> ret (tensor_y, tensor_y + sizeof(tensor_y) / sizeof(tensor_y[0]));
	return ret;
}
} //TMVA_SOFIE_testCaseBatchNorm_1
