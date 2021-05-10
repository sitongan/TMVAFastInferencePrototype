

// g++ -I ./eigen/ gemm_eigen.cpp -o gemm_eigen -std=c++14



#include <iostream>
#include <Eigen/Dense>


int main(){


   float a[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};    //2x3
   //float b[6] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};    //2x3
   //float c[6] = {0};
   float b[42] = {1.0};
   float c[6] = {0};
   float d[6] = {10.0};


   //Map< Matrix<float, 2,3> > ma(a);
   //Map< Matrix<float, 2,3> > mb(b);
   //Matrix<float, 2, 2> mc;
   //Map< Matrix<float, 2,2> > mc(c);
   //Map< Matrix<float, 2,2> > md(d);
   //mc = ma * mb.transpose() + md;
   //std::cout << mc << std::endl;

   Eigen::Map< Eigen::Vector<float,7> > va(a);
   Eigen::Map< Eigen::Matrix<float,6,7> > vb(b);
   Eigen::Map< Eigen::Vector<float,6> > vd(d);
   Eigen::Map< Eigen::Vector<float,6> > vc(c);
   vc = (va.transpose() * vb.transpose()).transpose() + vd;
   std::cout << vc << std::endl;
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
