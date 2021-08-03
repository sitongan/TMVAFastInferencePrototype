//Code generated automatically by TMVA for Inference of Model file [testCaseBatchNorm_1.onnx] at [Tue Aug  3 16:38:04 2021] 
#include<cmath>
#include<vector>
namespace TMVA_SOFIE_testCaseBatchNorm_1{
namespace BLAS{
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
	extern "C" void saxpy_(const int *n, const float* alpha, const float* x, const int *incx, float* y, const int* incy);
	extern "C" void ssbmv_(const char *uplo, const int *n, const int *k, const float *alpha, const float *a, const int *lda,
	                       const float *x, const int *incx, const float *beta, float *y, const int *incy);
}//BLAS
float tensor_var[3] = {0.801017821, 0.650941014, 0.502549887};
float tensor_mean[3] = {0.610773981, 0.373008311, 1.85429156};
float tensor_bias[3] = {-0.844192743, -1.79882622, -0.302479535};
float tensor_s[3] = {2.34357285, 0.356457889, 1.21757233};
float tensor_y[120];
std::vector<float> infer(float* tensor_x){
	float op_0_A[3];
	for (size_t c = 0; c < 3; c++) {
		op_0_A[c] = (tensor_s[c] / std::sqrt(tensor_var[c] + 1e-05 )); 
	}
	float op_0_Ba[120];
	float op_0_Bmean[120];
	size_t bs = 0;
	for (size_t c = 0; c < 3; c++) {
		for (size_t h = 0; h < 4; h++) {
			for (size_t w = 0; w < 5; w++) {
				op_0_Ba[ bs* 60 + c * 20 + h * 5 + w] = op_0_A[c];
				op_0_Bmean[ bs* 60 + c * 20 + h * 5 + w] = tensor_mean[c];
				tensor_y[ bs* 60 + c * 20 + h * 5 + w] = tensor_bias[c];
			}
		}
	}
	size_t op_0_batchOffset = 60;
	for (bs = 0; bs < 2; bs++) {
		std::copy(op_0_Ba, op_0_Ba+ op_0_batchOffset, op_0_Ba+ (bs* op_0_batchOffset));
		std::copy(op_0_Bmean, op_0_Bmean+op_0_batchOffset, op_0_Bmean+ (bs*op_0_batchOffset));
		std::copy(tensor_y, tensor_y +op_0_batchOffset, tensor_y + (bs*op_0_batchOffset));
	}
	float op_0_C[120];
	std::copy(tensor_x, tensor_x+120, op_0_C);
	const int N =120;
	const int op_0_incx = 1;
	const int op_0_incy = 1;
	float op_0_alpha = -1;
	BLAS::saxpy_(&N, &op_0_alpha, op_0_Bmean, &op_0_incx,op_0_C, &op_0_incy);

	char op_0_uplo = 'L';
	const int op_0_k = 0;
	const int op_0_lda = 1;
	float op_0_beta = -1;
	op_0_alpha = 1;
	BLAS::ssbmv_(&op_0_uplo, &N, &op_0_k, &op_0_alpha, op_0_C, &op_0_lda, op_0_Ba, &op_0_incx, &op_0_beta, tensor_y, &op_0_incy);

	std::vector<float> ret (tensor_y, tensor_y + sizeof(tensor_y) / sizeof(tensor_y[0]));
	return ret;
}
} //TMVA_SOFIE_testCaseBatchNorm_1
