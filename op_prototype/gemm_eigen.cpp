

// g++ -I ./eigen/ gemm_eigen.cpp -o gemm_eigen


#include <iostream>
#include <Eigen/Dense>


int main(){


   float a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};    //2x3
   float b[6] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};    //2x3
   //float c[6] = {0};

   float c[6] = {0};
   float d[4] = {10.0};

   using namespace Eigen;
   Map< Matrix<float, 2,3> > ma(a);
   Map< Matrix<float, 2,3> > mb(b);
   //Matrix<float, 2, 2> mc;
   Map< Matrix<float, 2,2> > mc(c);
   Map< Matrix<float, 2,2> > md(d);
   mc = ma * mb.transpose() + md;
   std::cout << mc << std::endl;


/*


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


   //BLAS::sgemm_(&transB, &transA, &n, &m, &k, &attr_alpha, b, &ldb, a,  &lda, &attr_beta, c, &n);


*/
   std::cout << "[";
   for (int i = 0; i < 6; i++){
      //std::cout << mc.data()[i] << "\t";
      std::cout << c[i] << "\t";
   }
   std::cout <<"]\n";

   //array([[22, 28],
   //    [49, 64]])

}
