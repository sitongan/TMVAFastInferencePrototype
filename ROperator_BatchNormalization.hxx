#ifndef TMVA_SOFIE_ROPERATOR_BatchNormalization
#define TMVA_SOFIE_ROPERATOR_BatchNormalization

#include "SOFIE_common.hxx"
#include "ROperator.hxx"
#include "RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_BatchNormalization final : public ROperator
{

private:

	/* Attributes */
	float fepsilon = 1e-05;
	float fmomentum = 0.9;
	std::size_t ftraining_mode = 0;

	std::string fNX;
	std::string fNY;
	std::string fNB;
	std::string fNScale;
	std::string fNMean;
	std::string fNVar;

	std::vector<size_t> fShapeX;
	std::vector<size_t> fShapeY;
	std::vector<size_t> fShapeB;
	std::vector<size_t> fShapeScale;
	std::vector<size_t> fShapeMean;
	std::vector<size_t> fShapeVar;
	
	std::string fType;

public:
	ROperator_BatchNormalization() = delete;
	
	/* Constructor */
	ROperator_BatchNormalization( float epsilon, float momentum, std::size_t training_mode,
	std::string nameX, std::string nameY, std::string nameB, std::string nameScale,
	std::string nameMean, std::string nameVar):
	fepsilon(epsilon), fmomentum(momentum), ftraining_mode(training_mode),
	fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)),
	fNB(UTILITY::Clean_name(nameB)), fNScale(UTILITY::Clean_name(nameScale)),
	fNMean(UTILITY::Clean_name(nameMean)), fNVar(UTILITY::Clean_name(nameVar))
	{
		if(std::is_same<T, float>::value){
			fType = "float";
		}
		else{
			throw
				std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a BatchNormalization operator");
		}
	}
	

	std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) {
		ETensorType out = input[0];
		return {out};
	}

	std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) {
		if (input.size() > 3 ) {
			throw
				std::runtime_error("TMVA SOFIE BatchNormalization Op Shape inference need 2 or 3 input tensors");
		}
		for(size_t i = 0; i < input.size(); i++) {
			if (input[i].size() != 4) {
				throw
				std::runtime_error("TMVA SOFIE BatchNormalization Op Shape inference only accept tensor with 4 dimensions");
			}
		}

		auto ret = input; //suggest copy to compiler
		return ret;
	}

	void Initialize(RModel& model){
		if (!model.CheckIfTensorAlreadyExist(fNX)) {
			throw
				std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNX +  " is not found in model");
		}

		if (fNB != "") {
			if (!model.CheckIfTensorAlreadyExist(fNB)) {
				throw
				std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNB + " is not found in model");
			}
		}
		if (fNScale != "") {
			if (!model.CheckIfTensorAlreadyExist(fNScale)) {
				throw
				std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNScale + " is not found in model");
			}
		}
		if (fNMean != "") {
			if (!model.CheckIfTensorAlreadyExist(fNMean)) {
				throw
				std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNMean + " is not found in model");
			}
		}
		if (fNVar != "") {
			if (!model.CheckIfTensorAlreadyExist(fNVar)) {
				throw
				std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNVar + " is not found in model");
			}
		}

		fShapeX = model.GetTensorShape(fNX);
		if (fShapeX.size() != 4) {
			throw
				std::runtime_error("TMVA SOFIE BatchNormalization Op input tensor" + fNX + " is not of 4 dimensions");
		}

		fShapeY = fShapeX;
		if (fNB != "") { /* ? */
			fShapeB = model.GetTensorShape(fNB);
		}

		model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
	}


	std::string Generate(std::string OpName){
		OpName = "op_" + OpName;
		if (fShapeX.empty()){
			throw std::runtime_error("TMVA SOFIE Batch Normalization called to Generate without being initialized first");
		}
		std::stringstream out;
		int length = 1;
		for(auto& i: fShapeX){
			length *= i;
		}

		// Batch Norm op
		out << "\t" << "for (size_t n = 0; n < " << fShapeX[0] << "; n++) {\n";
		out << "\t" << "\t" << "for (size_t c = 0; c < " << fShapeX[1] << "; c++) {\n";
		out << "\t" << "\t" << "\t" << "for (size_t h = 0; h < " << fShapeX[2] << "; h++) {\n";
		out << "\t" << "\t" << "\t" << "\t" << "for (size_t w = 0; w < " << fShapeX[3] << "; w++) {\n";
		out << "\t" << "\t" << "\t" << "\t" << "\t" << "tensor_" << fNY << "[n * " << fShapeX[1] * fShapeX[2] * fShapeX[3] << " + c * "<< fShapeX[2] * fShapeX[3] << " + h * " << fShapeX[3] << " + w] = ((tensor_" << fNX << "[n * " << fShapeX[1] * fShapeX[2] * fShapeX[3] << " + c * "<< fShapeX[2] * fShapeX[3] << " + h * " << fShapeX[3] << " + w] - "<< fNMean <<"[c * "<< fShapeX[2] * fShapeX[3] << "])/ " << OpName << "_sqrt("<< fNVar<< "[c * "<< fShapeX[2] * fShapeX[3] << "]) + "<<fepsilon<<" )) * "<< fNScale <<"[c * "<< fShapeX[2] * fShapeX[3] << "] + "<<fNB<<"[c * "<< fShapeX[2] * fShapeX[3] << "];\n";
		out << "\t" << "\t" << "\t" << "\t" << "}\n";
		out << "\t" << "\t" << "\t" << "}\n";
		out << "\t" << "\t" << "}\n";	
		out << "\t" << "}\n";

		return out.str();
	}

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_BatchNormalization
