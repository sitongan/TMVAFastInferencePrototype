#ifndef TMVA_SOFIE_ROPERATOR_TRANSPOSE
#define TMVA_SOFIE_ROPERATOR_TRANSPOSE


#include "SOFIE_common.hxx"
#include "ROperator.hxx"
#include "RModel.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{




template <typename T>
class ROperator_Transpose final : public ROperator
{

private:
   std::vector<int_t> fAttrPerm;
   std::string fNData;
   std::string fNOutput;

   std::string fType;

public:

   ROperator_Transpose() = delete;
   ROperator_Transpose(std::vector<int_t> attr_perm, std::string nameData, std::string nameOutput):
      fAttrPerm(attr_perm), fNData(nameData), fNOutput(nameOutput) {

      if (std::is_same<T, float>::value) {
         fType = "float";
      }else{
         throw std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a transpose operator");
      }
   }



   void Initialize(RModel& model){
      if (model.CheckIfTensorAlreadyExist(fNData) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Tranpose Op Input Tensor is not found in model");
      }
      auto input_shape {model.GetTensorShape(fNData)};
      std::vector<size_t> output_shape;
      for (int i = 0; i < fAttrPerm.size(); i++){
         output_shape[fAttrPerm[i]] = input_shape[i];
      }
      model.AddIntermediateTensor(fNOutput, model.GetTensorType(fNData), output_shape);

   }




   //ROperator_Transpose<float>::ROperator_Transpose(std::vector<int_t> attr_perm, C&& data, C&& transposed):
   //   attr_perm(attr_perm), data(data), transposed(transposed) {}



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
