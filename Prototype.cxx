#include "RModel.hxx"
#include "RModelParser_ONNX.hxx"

int main(){

   TMVA::Experimental::SOFIE::RModelParser_ONNX parser;
   TMVA::Experimental::SOFIE::RModel model = parser.Parse("LinearNN.onnx");
   model.PrintRequiredInputTensors();
   model.PrintInitializedTensors();

}
