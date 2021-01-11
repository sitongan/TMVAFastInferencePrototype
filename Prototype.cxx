#include "RModel.hxx"
#include "RModelParser_ONNX.hxx"

int main(){

   TMVA::Experimental::SOFIE::RModelParser_ONNX parser;
   TMVA::Experimental::SOFIE::RModel model = parser.Parse("LinearNN.onnx");
   TMVA::Experimental::SOFIE::RModel model2 = std::move(model);
   model2.PrintRequiredInputTensors();
   model2.PrintInitializedTensors();
   model2.HeadInitializedTensors("18.bias");
   model2.HeadInitializedTensors("0.weight");
}
