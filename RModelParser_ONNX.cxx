#include "RModelParser_ONNX.hxx"

#include <string> 
namespace TMVA{
namespace Experimental{
namespace SOFIE{



namespace INTERNAL{

ROperator* make_ROperator_Transpose(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, const std::unordered_map<std::string, size_t>& tensorname2idx){
   const auto& example_input_name = nodeproto.input(0);
   auto it = tensorname2idx.find(example_input_name);
   ETensorType operator_type = static_cast<ETensorType>(graphproto.input(it->second).type().tensor_type().elem_type());

   switch(operator_type){
   case ETensorType::FLOAT:
      //return make_ROperator_Transpose<float>(nodeproto, graphproto);
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Transpose does not yet support input type " + std::to_string(static_cast<int>(operator_type)));
   }
}

} //INTERNAL







}//SOFIE
}//Experimental
}//TMVA
