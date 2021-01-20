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

   const std::vector<Dim>& RModel::GetTensorShape(std::string name){
      auto f = fInputTensorInfos.find(name);
      if (f != fInputTensorInfos.end()){
         return f->second.shape;
      }else{
         auto f2 = fInitializedTensors.find(name);
         if (f2 != fInitializedTensors.end()){
            return f->second.shape;
         }else{
            throw std::runtime_error("TMVA SOFIE tensor [" + name + "] for which the shape is requested is not found");
         }
      }
   }

   void RModel::AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<Dim> shape){
      if (fInputTensorInfos.find(input_name) != fInputTensorInfos.end()){
         throw std::runtime_error("TMVA-SOFIE: input tensor with name " + input_name + " already exists \n");
      }
      TensorInfo inputInfo { type, shape };
      fInputTensorInfos[input_name] = inputInfo;
   }

   void RModel::AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<size_t> shape){
      if (fInputTensorInfos.find(input_name) != fInputTensorInfos.end()){
         throw std::runtime_error("TMVA-SOFIE: input tensor with name " + input_name + " already exists \n");
      }
      TensorInfo inputInfo { type, ConvertShapeToDim(shape) };
      fInputTensorInfos[input_name] = inputInfo;
   }

   void RModel::AddOperator(std::unique_ptr<ROperator> op, size_t order_execution){
      if (order_execution >= 0) {
         fOperators.insert(fOperators.begin() + order_execution, std::move(op));
      }else{
         fOperators.push_back(std::move(op));
      }
   }

   void RModel::AddInitializedTensors(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape, std::shared_ptr<void> data){
      //NB: own data
      if (fInitializedTensors.find(tensor_name) != fInitializedTensors.end()){
         throw std::runtime_error("TMVA-SOFIE: initialized tensor with name " + tensor_name + " already exists \n");
      }
      InitializedTensor new_tensor {type, shape, data};
      fInitializedTensors[tensor_name] = new_tensor;
   }

   void RModel::Generate(){
      fGC += ("//Code generated automatically by TMVA for Inference of Model file [" + fFileName + "] at [" + fParseTime.substr(0, fParseTime.length()-1) +"] \n");
      fGC += ("namespace TMVA_SOFIE_" + fName + "{\n");






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
