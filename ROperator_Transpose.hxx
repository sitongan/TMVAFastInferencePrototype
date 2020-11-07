#ifndef TMVA_SOFIE_ROPERATOR_TRANSPOSE
#define TMVA_SOFIE_ROPERATOR_TRANSPOSE


#include "SOFIE_common.hxx"
#include "ROperator.hxx"
#include "onnx.pb.h"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

namespace BLAS{
extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
                       const float * beta, float * C, const int * ldc);
}//BLAS


template <typename V, typename C = RTensor<V>>
class ROperator_Transpose final : public ROperator
{

private:
   std::vector<int_t> attr_perm;
   std::unique_ptr<C> data;
   std::unique_ptr<C> transposed;

public:

   ROperator_Transpose() = delete;

   ROperator_Transpose<float>::ROperator_Transpose(std::vector<int_t> attr_perm, C&& data, C&& transposed):
      attr_perm(attr_perm), data(data), transposed(transposed) {}



   /*
   ROperator_Gemm(const onnx::NodeProto& nodeproto, RGraph& this_graph);
   ROperator_Gemm(const std::string& name_A , const std::string& name_B, const std::string& name_C,
      const std::string& name_Y, float attribute_alpha, float attribute_beta, int attribute_transA, int attribute_transB,
      RGraph& this_graph);
   void Forward_reference() final;
   void Forward_blas() final;
   */
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_TRANSPOSE
