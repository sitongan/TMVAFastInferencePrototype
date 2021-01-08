#ifndef TMVA_SOFIE_RMODEL
#define TMVA_SOFIE_RMODEL

#include <vector>
#include <unordered_map>
#include <iostream>
#include <memory>

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

public:

   RModel(RModel&& other){
      fInputTensorInfos = other.fInputTensorInfos;
      fInputTensorInfos = std::move(other.fInputTensorInfos);
      fOperators = std::move(other.fOperators);
      fInitializedTensors = std::move(other.fInitializedTensors);
      fAllTensorInitialized = other.fAllTensorInitialized;
   }

   RModel& operator=(RModel&& other){
      fInputTensorInfos = std::move(other.fInputTensorInfos);
      fOperators = std::move(other.fOperators);
      fInitializedTensors = std::move(other.fInitializedTensors);
      fAllTensorInitialized = other.fAllTensorInitialized;
      return *this;
   }

   //disallow copy
   RModel(const RModel& other) = delete;
   RModel& operator=(const RModel& other) = delete;

   RModel(){};

   void addInputTensorInfo(std::string input_name, ETensorType type, std::vector<Dim> shape){
      if (fInputTensorInfos.find(input_name) != fInputTensorInfos.end()){
         throw std::runtime_error("TMVA-SOFIE: input tensor with name " + input_name + " already exists \n");
      }
      TensorInfo inputInfo { type, shape };
      fInputTensorInfos[input_name] = inputInfo;
   }

   void addInputTensorInfo(std::string input_name, ETensorType type, std::vector<size_t> shape){
      if (fInputTensorInfos.find(input_name) != fInputTensorInfos.end()){
         throw std::runtime_error("TMVA-SOFIE: input tensor with name " + input_name + " already exists \n");
      }
      TensorInfo inputInfo { type, ConvertShapeToDim(shape) };
      fInputTensorInfos[input_name] = inputInfo;
   }


   void addOperator(std::unique_ptr<ROperator> op, size_t order_execution = -1){

      if (order_execution >= 0) {
         fOperators.insert(fOperators.begin() + order_execution, std::move(op));
      }else{
         fOperators.push_back(std::move(op));
      }

   }


   void addInitializedTensors(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape, void* data){
      //NB: own data
      if (fInitializedTensors.find(tensor_name) != fInitializedTensors.end()){
         throw std::runtime_error("TMVA-SOFIE: initialized tensor with name " + tensor_name + " already exists \n");
      }
      InitializedTensor new_tensor {type, shape, data};
      fInitializedTensors[tensor_name] = new_tensor;
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
         std::cout << "Tensor name: " << it.first << "\t";
         std::cout << "type: " << ConvertTypeToString(it.second.type) << "\t";
         std::cout << "shape: [";
         for (int i = 0; i < it.second.shape.size(); i++){
            std::cout << it.second.shape[i];
            if (i < it.second.shape.size() - 1) std::cout << ",";
         }
         std::cout << "]" << std::endl;
      }
   }

   ~RModel(){
      for (auto& i: fInitializedTensors){
         free(i.second.data);
      }
   }

};

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RMODEL
