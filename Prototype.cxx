#include <memory>

#include "RModel.hxx"
#include "RModelParser_ONNX.hxx"

#include <cctype>
#include <algorithm>


using namespace TMVA::Experimental::SOFIE;

int main(){


   RModelParser_ONNX parser;

   RModel model = parser.Parse("./lstm_defaults.onnx");
   RModel model1 = std::move(model);
   model1.Generate();
   model1.OutputGenerated("lstm_defaults.hxx");

   RModel model2 = parser.Parse("./lstm_initial_bias.onnx");
   model2.Generate();
   model2.OutputGenerated("lstm_initial_bias.hxx");

   RModel model3 = parser.Parse("./lstm_peepholes.onnx");
   model3.Generate();
   model3.OutputGenerated("lstm_peepholes.hxx");

   RModel model4 = parser.Parse("./lstm_batchwise.onnx");
   model4.Generate();
   model4.OutputGenerated("lstm_batchwise.hxx");

   RModel model5 = parser.Parse("./lstm_bidirectional.onnx");
   model5.Generate();
   model5.OutputGenerated("lstm_bidirectional.hxx");

}
