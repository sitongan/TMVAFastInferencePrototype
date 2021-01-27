#include "RModel.hxx"




namespace TMVA{
namespace Experimental{
namespace SOFIE{

   RModel::RModel(RModel&& other){
      fInputTensorInfos = other.fInputTensorInfos;
      fInputTensorInfos = std::move(other.fInputTensorInfos);
      fOperators = std::move(other.fOperators);
      fInitializedTensors = std::move(other.fInitializedTensors);
      fAllTensorInitialized = other.fAllTensorInitialized;
      fName = other.fName;
      fFileName = other.fFileName;
      fParseTime = other.fParseTime;
   }

   RModel& RModel::operator=(RModel&& other){
      fInputTensorInfos = std::move(other.fInputTensorInfos);
      fOperators = std::move(other.fOperators);
      fInitializedTensors = std::move(other.fInitializedTensors);
      fAllTensorInitialized = other.fAllTensorInitialized;
      fName = other.fName;
      fFileName = other.fFileName;
      fParseTime = other.fParseTime;
      return *this;
   }

   RModel::RModel(std::string name, std::string parsedtime): fFileName (name), fParseTime(parsedtime) {
      fName = fFileName.substr(0, fFileName.rfind("."));
   }

   const std::vector<size_t>& RModel::GetTensorShape(std::string name){
      auto f = fReadyInputTensorInfos.find(name);
      if (f != fReadyInputTensorInfos.end()){
         return f->second.shape;
      }
      auto f2 = fInitializedTensors.find(name);
      if (f2 != fInitializedTensors.end()){
         return f2->second.shape;
      }
      auto f3 = fInputTensorInfos.find(name);
      if (f3 != fInputTensorInfos.end()){
         throw std::runtime_error("TMVA SOFIE tensor [" + name + "] is an input tensor with unspecified dimension parameter");
      }

      throw std::runtime_error("TMVA SOFIE tensor [" + name + "] for which the shape is requested is not found");
   }

   const ETensorType& RModel::GetTensorType(std::string name){
      auto f = fReadyInputTensorInfos.find(name);
      if (f != fReadyInputTensorInfos.end()){
         return f->second.type;
      }
      auto f2 = fInitializedTensors.find(name);
      if (f2 != fInitializedTensors.end()){
         return f2->second.type;
      }
      auto f3 = fInputTensorInfos.find(name);
      if (f3 != fInputTensorInfos.end()){
         return f3->second.type;
      }

      throw std::runtime_error("TMVA SOFIE tensor [" + name + "] for which the shape is requested is not found");
   }

   bool RModel::CheckIfTensorAlreadyExist(std::string tensor_name){
      if (fReadyInputTensorInfos.find(tensor_name) != fReadyInputTensorInfos.end())  return true;
      if (fInitializedTensors.find(tensor_name) != fInitializedTensors.end()) return true;
      if (fIntermediateTensorInfos.find(tensor_name) != fIntermediateTensorInfos.end()) return true;
      return false;
   }

   void RModel::AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<Dim> shape){
      if (CheckIfTensorAlreadyExist(input_name)){
         throw std::runtime_error("TMVA-SOFIE: input tensor with name " + input_name + " already exists \n");
      }

      InputTensorInfo inputInfo { type, shape };
      fInputTensorInfos[input_name] = inputInfo;
   }

   void RModel::AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<size_t> shape){
      if (CheckIfTensorAlreadyExist(input_name)){
         throw std::runtime_error("TMVA-SOFIE: input tensor with name " + input_name + " already exists \n");
      }
      TensorInfo inputInfo { type, shape };
      fReadyInputTensorInfos[input_name] = inputInfo;
   }

   void RModel::AddOperator(std::unique_ptr<ROperator> op, int order_execution){
      if (order_execution >= 0) {
         fOperators.insert(fOperators.begin() + order_execution, std::move(op));
      }else{
         fOperators.push_back(std::move(op));
      }
   }

   void RModel::AddInitializedTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape, std::shared_ptr<void> data){
      //NB: own data
      if (CheckIfTensorAlreadyExist(tensor_name)){
         throw std::runtime_error("TMVA-SOFIE: initialized tensor with name " + tensor_name + " already exists \n");
      }
      InitializedTensor new_tensor {type, shape, data};
      fInitializedTensors[tensor_name] = new_tensor;
   }

   void RModel::AddIntermediateTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape){
      if (CheckIfTensorAlreadyExist(tensor_name)){
         throw std::runtime_error("TMVA-SOFIE: intermediate tensor with name " + tensor_name + " already exists \n");
      }
      TensorInfo new_tensor {type, shape};
      fIntermediateTensorInfos[tensor_name] = new_tensor;
   }

   void RModel::Initialize(){
      for (auto& i : fOperators){
         i->Initialize(*this);
      }
   }

   void RModel::Generate(){
      Initialize();
      fGC += ("//Code generated automatically by TMVA for Inference of Model file [" + fFileName + "] at [" + fParseTime.substr(0, fParseTime.length()-1) +"] \n");
      for (auto& i: fNeededStdLib){
         fGC += "#include<" + i + ">\n";
      }
      fGC += ("namespace TMVA_SOFIE_" + fName + "{\n");
      if (fNeedGemm){
         fGC += ("namespace BLAS{\n"
         "\textern \"C\" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,\n"
         "\t                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,\n"
         "\t                       const float * beta, float * C, const int * ldc);\n"
         "}//BLAS\n");

      fGC += "void infer(){\n";
      for (int id = 0; id < fOperators.size() ; id++){
         fGC+= (fOperators[id]->Generate(std::to_string(id)));
      }
      fGC += "}\n";
      }






      fGC += ("} //TMVA_SOFIE_" + fName + "\n");
   }



   void RModel::PrintRequiredInputTensors(){
      std::cout << "Model requires following inputs:\n";
      for (auto& inputInfo: fInputTensorInfos){
         std::cout << "Tensor name: " << inputInfo.first << "\t";
         std::cout << "type: " << ConvertTypeToString(inputInfo.second.type) << "\t";
         std::cout << "shape: [";
         for (int i = 0; i < inputInfo.second.shape.size(); i++){
            if (inputInfo.second.shape[i].isParam){
               std::cout << inputInfo.second.shape[i].param;
            }else{
               std::cout << inputInfo.second.shape[i].dim ;
            }
            if (i < inputInfo.second.shape.size() - 1) std::cout << ",";
         }
         std::cout << "]" << std::endl;
      }
   }

   void RModel::PrintInitializedTensors(){
      std::cout << "Model initialized the following tensors:\n";
      for (auto& it: fInitializedTensors){
         std::cout << "Tensor name: \"" << it.first << "\"\t";
         std::cout << "type: " << ConvertTypeToString(it.second.type) << "\t";
         std::cout << "shape: [";
         for (int i = 0; i < it.second.shape.size(); i++){
            std::cout << it.second.shape[i];
            if (i < it.second.shape.size() - 1) std::cout << ",";
         }
         std::cout << "]" << std::endl;
      }
   }

   void RModel::HeadInitializedTensors(std::string name, int n_print){
      auto it = fInitializedTensors.find(name);
      if (it == fInitializedTensors.end()){
         std::cout << "Tensor " << name << " not found in model's intiialized tensor list" << std::endl;
         return;
      }

      std::cout << "Tensor name: " << it->first << "\t";
      std::cout << "type: " << ConvertTypeToString(it->second.type) << "\t";
      std::size_t length =1;
      std::cout << "shape: [";
      for (int i = 0; i < it->second.shape.size(); i++){
         std::cout << it->second.shape[i];
         length *= it->second.shape[i];
         if (i < it->second.shape.size() - 1) std::cout << ",";
      }
      std::cout << "]" << std::endl;
      bool ellipsis = true;
      if (n_print > length){
         n_print = length;
         ellipsis = false;
      }

      std::cout << "data: [" << std::endl;
      switch(it->second.type){
         case ETensorType::FLOAT : {
            auto converted_data = std::static_pointer_cast<float>(it->second.data).get();
            for (int i =0; i < n_print; i++){
               std::cout << converted_data[i];
               if (i < n_print - 1) std::cout << " ,";
            }
            break;
         }
      }
      if (ellipsis) std::cout << ", ...";
      std::cout << "]" << std::endl;

   }

}//SOFIE
}//Experimental
}//TMVA
