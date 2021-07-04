#include <memory>

#include "RModel.hxx"
#include "RModelParser_ONNX.hxx"

#include <cctype>
#include <algorithm>


using namespace TMVA::Experimental::SOFIE;

int main(){


   RModelParser_ONNX parser;

   RModel model = parser.Parse("./rnn_defaults.onnx");
   RModel model1 = std::move(model);
   model1.Generate();
   model1.OutputGenerated("rnn_defaults.hxx");

   RModel model2 = parser.Parse("./rnn_seq_length.onnx");
   model2.Generate();
   model2.OutputGenerated("rnn_seq_length.hxx");

   RModel model3 = parser.Parse("./rnn_batchwise.onnx");
   model3.Generate();
   model3.OutputGenerated("rnn_batchwise.hxx");

   RModel model4 = parser.Parse("rnn_bidirectional.onnx");
   model4.Generate();
   model4.OutputGenerated("rnn_bidirectional.hxx");

   RModel model5 = parser.Parse("rnn_bidirectional_batchwise.onnx");
   model5.Generate();
   model5.OutputGenerated("rnn_bidirectional_batchwise.hxx");

   RModel model6 = parser.Parse("rnn_sequence.onnx");
   model6.Generate();
   model6.OutputGenerated("rnn_sequence.hxx");

   RModel model7 = parser.Parse("rnn_sequence_batchwise.onnx");
   model7.Generate();
   model7.OutputGenerated("rnn_sequence_batchwise.hxx");

}
