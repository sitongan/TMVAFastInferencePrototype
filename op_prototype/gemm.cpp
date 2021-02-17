
namespace BLAS{
extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
                       const float * beta, float * C, const int * ldc);
}//BLAS




#include<iostream>


int main(){


   float a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};    //2x3
   float b[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};    //3x2
   float c[6] = {0};

   int tA = 1;
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

}
