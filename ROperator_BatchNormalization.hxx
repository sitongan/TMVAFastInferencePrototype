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
	std::string fNScale;
	std::string fNB;
	std::string fNMean;
	std::string fNVar;
	std::string fNY;

	std::vector<size_t> fShapeX;
	std::vector<size_t> fShapeScale;
	std::vector<size_t> fShapeB;
	std::vector<size_t> fShapeMean;
	std::vector<size_t> fShapeVar;
	std::vector<size_t> fShapeY;
	
	std::string fType;

public:
	ROperator_BatchNormalization() = delete;
	
	/* Constructor */
	ROperator_BatchNormalization( float epsilon, float momentum, std::size_t training_mode,
	std::string nameX, std::string nameScale, std::string nameB, 
	std::string nameMean, std::string nameVar, std::string nameY):
	fepsilon(epsilon), fmomentum(momentum), ftraining_mode(training_mode),
	fNX(UTILITY::Clean_name(nameX)), fNScale(UTILITY::Clean_name(nameScale)), 
	fNB(UTILITY::Clean_name(nameB)), fNMean(UTILITY::Clean_name(nameMean)), 
	fNVar(UTILITY::Clean_name(nameVar)), fNY(UTILITY::Clean_name(nameY))
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
		if (input.size() != 5 ) {
			throw
				std::runtime_error("TMVA SOFIE BatchNormalization Op Shape inference need 5 input tensors");
		}
		for(size_t i = 0; i < input.size(); i++) {
			if (input[i].size() != 4) {
				throw
				std::runtime_error("TMVA SOFIE BatchNormalization Op Shape inference only accept tensor with 4 dimensions");
			}
		}

		auto ret = input; 
		return ret;
	}

	void Initialize(RModel& model){
		if (!model.CheckIfTensorAlreadyExist(fNX)) {
			throw
				std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNX + " fnx is not found in model");
		}
		if (!model.CheckIfTensorAlreadyExist(fNScale)) {
			throw
				std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNScale + " fns is not found in model");
		}
		if (!model.CheckIfTensorAlreadyExist(fNB)) {
			throw
				std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNB + " fnb is not found in model");
		}
		if (!model.CheckIfTensorAlreadyExist(fNMean)) {
			throw
				std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNMean + " fnm is not found in model");
		}
		if (!model.CheckIfTensorAlreadyExist(fNVar)) {
			throw
				std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNVar + " fnv is not found in model");
		}

		fShapeX = model.GetTensorShape(fNX);
		if (fShapeX.size() != 4) {
			throw
				std::runtime_error("TMVA SOFIE BatchNormalization Op input tensor " + fNX + " fnx is not of 4 dimensions");
		}
		
		fShapeScale = model.GetTensorShape(fNScale);
		fShapeB = model.GetTensorShape(fNB);
		fShapeMean = model.GetTensorShape(fNMean);
		fShapeVar = model.GetTensorShape(fNVar);
		
		fShapeY = fShapeX;
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
		// ///initialize A
		size_t batchSize = fShapeX[0], channels = fShapeX[1], height = fShapeX[2], width = fShapeX[3];
		size_t n = batchSize * channels * height * width;
		if (fType == "float") {
        	out << "\t" << "float " << OpName << "_A[" << channels << "];\n";
        }
		out << "\t"<< "for (size_t c = 0; c < " << channels << "; c++) {\n";
		out << "\t"<< "\t" << OpName << "_A[c] = (tensor_" << fNScale <<"[c] / std::sqrt(" << "tensor_" << fNVar<< "[c] + "<<fepsilon<<" )); \n";
		out << "\t"<< "}\n";

		/// Broadcast A, bias and input_mean to shape_X
		if (fType == "float") {
        	out << "\t" << "float " << OpName << "_Ba[" << n << "];\n";
        }
		if (fType == "float") {
        	out << "\t" << "float " << OpName << "_Bmean[" << n << "];\n";
        }
		out << "\t" << "size_t bs = 0;\n";
		out << "\t" << "for (size_t c = 0; c < " << channels << "; c++) {\n";
		out << "\t" << "\t" << "for (size_t h = 0; h < " << height << "; h++) {\n";
		out << "\t" << "\t" << "\t" << "for (size_t w = 0; w < " << width << "; w++) {\n";
		out << "\t" << "\t" << "\t" << "\t" << OpName << "_Ba[ bs* " << channels*height*width << " + c * "<< height*width << " + h * " << width << " + w] = "<<OpName << "_A[c];\n";
		out << "\t" << "\t" << "\t" << "\t" << OpName << "_Bmean[ bs* " << channels*height*width << " + c * "<< height*width << " + h * " << width << " + w] = tensor_" << fNMean <<"[c];\n";
		out << "\t" << "\t" << "\t" << "\t" << "tensor_" << fNY << "[ bs* " << channels*height*width << " + c * "<< height*width << " + h * " << width << " + w] = tensor_" << fNB <<"[c];\n";
		out << "\t" << "\t" << "\t" << "}\n";
		out << "\t" << "\t" << "}\n";
		out << "\t" << "}\n";

		out << "\t" << "size_t "<<OpName<< "_batchOffset = "<< channels*height*width<<";\n";		
		out << "\t" << "for (bs = 0; bs < " << batchSize << "; bs++) {\n";
		out << "\t"<< "\t" << "std::copy("<< OpName << "_Ba, "<< OpName << "_Ba+ "<<OpName<< "_batchOffset, "<< OpName << "_Ba+ (bs* "<<OpName<< "_batchOffset));\n";
		out << "\t"<< "\t" << "std::copy("<< OpName << "_Bmean, "<< OpName << "_Bmean+"<<OpName<< "_batchOffset, "<< OpName << "_Bmean+ (bs*"<<OpName<< "_batchOffset));\n";
		out << "\t"<< "\t" << "std::copy("<< "tensor_" << fNY << ", "<< "tensor_" << fNY << " +"<<OpName<< "_batchOffset, "<< "tensor_" << fNY << " + (bs*"<<OpName<< "_batchOffset));\n";
		out << "\t" << "}\n";

		/// initailize C
		if (fType == "float") {
        	out << "\t" << "float " << OpName << "_C[" << n << "];\n";
        }
		out << "\t"<< "std::copy("<< "tensor_" << fNX <<", "<< "tensor_" << fNX <<"+"<< n<<", "<< OpName << "_C);\n"; 
		
		/// blas saxpy (C = X - Bmean)
		out << "\t" << "const int N ="<<batchSize * channels * height * width<<";\n";
		out << "\t" << "const int "<<OpName<< "_incx = 1;\n";
		out << "\t" << "const int "<<OpName<< "_incy = 1;\n";
		out << "\t" << "float "<<OpName<< "_alpha = -1;\n";
		out << "\t" << "BLAS::saxpy_(&N, &" << OpName << "_alpha, " << OpName << "_Bmean, &" << OpName << "_incx," << OpName << "_C, &" << OpName << "_incy);\n\n";
        

		// blas smbv (Y = CxBa + Bbias)
		out << "\t" << "char " << OpName << "_uplo = 'L';\n";
		out << "\t" << "const int "<<OpName<< "_k = 0;\n";
		out << "\t" << "const int "<<OpName<< "_lda = 1;\n";
		out << "\t" << "float "<<OpName<< "_beta = -1;\n";
		out << "\t" <<OpName<< "_alpha = 1;\n";
		out << "\t" << "BLAS::ssbmv_(&" << OpName << "_uplo, &N, &" << OpName << "_k, &" << OpName << "_alpha, " << OpName << "_C, &" << OpName << "_lda, " << OpName << "_Ba, &" << OpName << "_incx, &" << OpName << "_beta, " << "tensor_" << fNY << ", &" << OpName << "_incy);\n\n";
		// std::cout<<out;
		return out.str();
	}

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_BatchNormalization
