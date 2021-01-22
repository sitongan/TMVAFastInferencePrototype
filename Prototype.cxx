#include <memory>

#include "RModel.hxx"
#include "RModelParser_ONNX.hxx"


using namespace TMVA::Experimental::SOFIE;

int main(){


   RModelParser_ONNX parser;
   RModel model = parser.Parse("./LinearNN.onnx");
   RModel model2 = std::move(model);
   //model2.printRequiredInputTensors();
   //model2.printInitializedTensors();
   //model2.headInitializedTensors("18.bias");
   //model2.headInitializedTensors("0.weight");
	model2.Generate();
	model2.PrintGenerated();

	std::cout << "===" << std::endl;

	RModel model3;
	model3.AddInputTensorInfo("1", ETensorType::FLOAT, {1,2,3,4});
	//auto op = std::make_unique<ROperator_Transpose<float>>({3,2,1,0}, "1", "2");
	std::unique_ptr<ROperator_Transpose<float>>op ( new ROperator_Transpose<float>({3,2,1,0}, "1", "2")) ;
	auto  a = model3.GetTensorShape("1");

	op->Initialize(model3);
	std::cout << (op->Generate());

}
