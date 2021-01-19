#ifndef TMVA_SOFIE_RMODEL
#define TMVA_SOFIE_RMODEL

#include <vector>
#include <unordered_map>
#include <iostream>
#include <memory>
#include <ctime>

#include "SOFIE_common.hxx"
#include "ROperator.hxx"


namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RModel{

private:

   std::unordered_map<std::string, TensorInfo> fInputTensorInfos;
   std::vector<std::unique_ptr<ROperator>> fOperators;

   std::unordered_map<std::string, InitializedTensor> fInitializedTensors;
   bool fAllTensorInitialized = false;

   std::string fName="UninitializedModel";
   std::string fFileName; //file name of original model file for identification
   std::string fParseTime; //UTC date and time string at parsing


   std::string fGC; //generated code

public:

   //explicit move ctor/assn
   RModel(RModel&& other){
      fInputTensorInfos = other.fInputTensorInfos;
      fInputTensorInfos = std::move(other.fInputTensorInfos);
      fOperators = std::move(other.fOperators);
      fInitializedTensors = std::move(other.fInitializedTensors);
      fAllTensorInitialized = other.fAllTensorInitialized;
      fName = other.fName;
      fFileName = other.fFileName;
      fParseTime = other.fParseTime;
   }

   RModel& operator=(RModel&& other){
      fInputTensorInfos = std::move(other.fInputTensorInfos);
      fOperators = std::move(other.fOperators);
      fInitializedTensors = std::move(other.fInitializedTensors);
      fAllTensorInitialized = other.fAllTensorInitialized;
      fName = other.fName;
      fFileName = other.fFileName;
      fParseTime = other.fParseTime;
      return *this;
   }

   //disallow copy
   RModel(const RModel& other) = delete;
   RModel& operator=(const RModel& other) = delete;

   RModel(){}
   RModel(std::string name, std::string parsedtime): fFileName (name), fParseTime(parsedtime) {
      fName = fFileName.substr(0, fFileName.rfind("."));
   }



   void AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<Dim> shape){
      if (fInputTensorInfos.find(input_name) != fInputTensorInfos.end()){
         throw std::runtime_error("TMVA-SOFIE: input tensor with name " + input_name + " already exists \n");
      }
      TensorInfo inputInfo { type, shape };
      fInputTensorInfos[input_name] = inputInfo;
   }

   void AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<size_t> shape){
      if (fInputTensorInfos.find(input_name) != fInputTensorInfos.end()){
         throw std::runtime_error("TMVA-SOFIE: input tensor with name " + input_name + " already exists \n");
      }
      TensorInfo inputInfo { type, ConvertShapeToDim(shape) };
      fInputTensorInfos[input_name] = inputInfo;
   }


   void AddOperator(std::unique_ptr<ROperator> op, size_t order_execution = -1){

      if (order_execution >= 0) {
         fOperators.insert(fOperators.begin() + order_execution, std::move(op));
      }else{
         fOperators.push_back(std::move(op));
      }

   }


   void AddInitializedTensors(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape, std::shared_ptr<void> data){
      //NB: own data
      if (fInitializedTensors.find(tensor_name) != fInitializedTensors.end()){
         throw std::runtime_error("TMVA-SOFIE: initialized tensor with name " + tensor_name + " already exists \n");
      }
      InitializedTensor new_tensor {type, shape, data};
      fInitializedTensors[tensor_name] = new_tensor;
   }

   void Generate(){
      fGC += ("//Code generated automatically by TMVA for Inference of Model file [" + fFileName + "] at [" + fParseTime.substr(0, fParseTime.length()-1) +"] \n");
      fGC += ("namespace TMVA_SOFIE_" + fName + "{\n");






      fGC += ("} //TMVA_SOFIE_" + fName + "\n");
   }

   void PrintGenerated(){
      std::cout << fGC;
   }


/*
   template <typename T>
   void addInitializedTensors(std::string tensor_name, RTensor<T> new_tensor){
      //a view only
      T obj;
      if (fInitializedTensors.find(tensor_name) != fInitializedTensors.end()){
         throw std::runtime_error("TMVA-SOFIE: initialized tensor with name " + tensor_name + " already exists \n");
      }
      InitializedTensor new_tensor_ {GetTemplatedType(obj), new_tensor.GetShape() , static_cast<void>(new_tensor.GetData())};
      fInitializedTensors[tensor_name] = new_tensor_;
   }
*/

   void PrintRequiredInputTensors(){
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

   void PrintInitializedTensors(){
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

   void HeadInitializedTensors(std::string name, int n_print = 50){
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

   ~RModel(){
      /*
      for (auto& i: fInitializedTensors){
         free(i.second.data);
      }
      */
   }

};

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RMODEL
