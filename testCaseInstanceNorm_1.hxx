//Code generated automatically by TMVA for Inference of Model file [testCaseInstanceNorm_1.onnx] at [Sat Aug 14 14:06:38 2021] 
#include<cmath>
#include<vector>
namespace TMVA_SOFIE_testCaseInstanceNorm_1{
namespace BLAS{
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
	extern "C" void saxpy_(const int *n, const float* alpha, const float* x, const int *incx, float* y, const int* incy);
	extern "C" void scopy_(const int *n, const float* x, const int *incx, float* y, const int* incy);
	extern "C" void ssbmv_(const char *uplo, const int *n, const int *k, const float *alpha, const float *a, const int *lda,
	                       const float *x, const int *incx, const float *beta, float *y, const int *incy);
}//BLAS
float tensor_bias[3] = {-0.116739199, -0.574047804, -0.765320599};
float tensor_s[3] = {-0.459328562, -0.6573385, -1.10990715};
float tensor_y[120];
std::vector<float> infer(float* tensor_x){
	for (size_t n = 0; n < 2; n++) {
		for (size_t c = 0; c < 3; c++) {
			float op_0_mean = 0;
			for (size_t h = 0; h < 4; h++) {
				for (size_t w = 0; w < 5; w++) {
					op_0_mean += tensor_x[n * 60 + c * 20 + h * 5 + w];
				}
			}
			op_0_mean = op_0_mean/20;
			float op_0_var = 0;
			for (size_t h = 0; h < 4; h++) {
				for (size_t w = 0; w < 5; w++) {
					op_0_var += (tensor_x[n * 60 + c * 20 + h * 5 + w] - op_0_mean) * (tensor_x[n * 60 + c * 20 + h * 5 + w] - op_0_mean);
				}
			}
			op_0_var = op_0_var/20;
			for (size_t h = 0; h < 4; h++) {
				for (size_t w = 0; w < 5; w++) {
					tensor_y[n * 60 + c * 20 + h * 5 + w] = ((tensor_x[n * 60 + c * 20 + h * 5 + w] - op_0_mean)/ std::sqrt(op_0_var + 1e-05 ))* tensor_s[c] + tensor_bias[c];
				}
			}
		}
	}
	std::vector<float> ret (tensor_y, tensor_y + sizeof(tensor_y) / sizeof(tensor_y[0]));
	return ret;
}
} //TMVA_SOFIE_testCaseInstanceNorm_1
