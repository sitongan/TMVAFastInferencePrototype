
namespace BLAS{
extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
                       const float * beta, float * C, const int * ldc);
extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);


}//BLAS




#include<iostream>


int main(){


   float a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};    //2x3
   float b[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};    //2x3
   float c[6] = {0};

   int tA = 0;
   int tB = 1;

   char transA = tA? 't' : 'n';
   char transB = tB? 't' : 'n';
   int m = 2;
   int n = 2;
   int k = 3;
   float attr_alpha =1.0;
   float attr_beta =1.0;

   int lda = tA? m: k;
   //int lda = m;

   int ldb = tB? k: n;
   //int ldb = n;
   //int ldb = k;


   BLAS::sgemm_(&transB, &transA, &n, &m, &k, &attr_alpha, b, &ldb, a,  &lda, &attr_beta, c, &n);
   //BLAS::sgemm_(&transA, &transB, &m, &n, &k, &attr_alpha, a, &lda, b,  &ldb, &attr_beta, c, &n);



   std::cout << "[";
   for (int i = 0; i < 6; i++){
      std::cout << c[i] << "\t";
   }
   std::cout <<"]\n";

   //array([[22, 28],
   //    [49, 64]])


   const int M = 2, N = 2;
   const int one = 1;
   const float alpha = -1.0, beta = 1.0;
   const char trans = 'N';
   const char noTrans = 'N';

   float Yc[4] = { 0x1.42c7bd3b6266cp+4, 0x1.6c6ff393729dp+4, 0x1.acee1f3938c0bp-2, 0x1.b0cd5ba440d93p+0 };
   float Yr[4] = { 0x1.42c7bd3b6266cp+4, 0x1.6c6ff393729dp+4, 0x1.acee1f3938c0bp-2,  0x1.b0cd5ba440d93p+0 };

   float A[2] = { 0x1.11acee560242ap-2, 0x1p+0 };

   float Bc[2] = { 0x1.8p+2, 0x1.cp+2 };
   float Br[2] = { 0x1.8p+2, 0x1.cp+2 };

   BLAS::sgemv_( &noTrans, &M, &N, &alpha, Yc, &M, A, &one, &beta, Bc, &one );

   printf("Result Column Major\n");
   printf("%a %a\n", Bc[0], Bc[1]);

   BLAS::sgemv_( &trans, &N, &M, &alpha, Yr, &N, A, &one, &beta, Br, &one );

   printf("Result Row Major\n");
   printf("%a %a\n", Br[0], Br[1]);

}
