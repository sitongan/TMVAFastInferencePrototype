//Code generated automatically by TMVA for Inference of Model file [testCaseBatchNorm_1.onnx] at [Mon Jul  5 17:11:51 2021] 
#include<vector>
namespace TMVA_SOFIE_testCaseBatchNorm_1{
namespace BLAS{
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
}//BLAS
float tensor_y[120];
std::vector<float> infer(float* tensor_varfloat* tensor_meanfloat* tensor_biasfloat* tensor_sfloat* tensor_x){
 BatchNormalization Op 
	for (size_t n = 0; n < 2; n++) {
		for (size_t c = 0; c < 3; c++) {
			for (size_t h = 0; h < 4; h++) {
				for (size_t w = 0; w < 5; w++) {
					tensor_y[n * 60 + c * 20 + h * 5 + w] = ((tensor_x[n * 60 + c * 20 + h * 5 + w] - mean[c * 20])/ op_0_sqrt(var[c * 20]) + 1e-05 ) * bias[c * 20] + s[c * 20];
				}
			}
		}
	}
	std::vector<float> ret (tensor_y, tensor_y + sizeof(tensor_y) / sizeof(tensor_y[0]));
	return ret;
}
} //TMVA_SOFIE_testCaseBatchNorm_1
