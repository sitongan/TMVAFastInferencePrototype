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

   std::unordered_map<std::string, TensorInfo> fInputTensorInfos; //graph input only; not including operator input (intermediate tensors)
   std::unordered_map<std::string, InitializedTensor> fInitializedTensors;

   std::vector<std::unique_ptr<ROperator>> fOperators;
   bool fAllTensorInitialized = false;

   std::string fName="UninitializedModel";
   std::string fFileName; //file name of original model file for identification
   std::string fParseTime; //UTC date and time string at parsing


   std::string fGC; //generated code

public:

   //explicit move ctor/assn
   RModel(RModel&& other);

   RModel& operator=(RModel&& other);

   //disallow copy
   RModel(const RModel& other) = delete;
   RModel& operator=(const RModel& other) = delete;

   RModel(){}
   RModel(std::string name, std::string parsedtime);

   const std::vector<Dim>& GetTensorShape(std::string name);

   bool CheckIfTensorAlreadyExist(std::string tensor_name);
   void AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<Dim> shape);
   void AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<size_t> shape);
   void AddOperator(std::unique_ptr<ROperator> op, size_t order_execution = -1);
   void AddInitializedTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape, std::shared_ptr<void> data);
   void AddIntermediateTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape);
   void Generate();

   void PrintGenerated(){
      std::cout << fGC;
   }


/*
   template <typename T>
   void AddInitializedTensor(std::string tensor_name, RTensor<T> new_tensor){
      //a view only
      T obj;
      if (fInitializedTensors.find(tensor_name) != fInitializedTensors.end()){
         throw std::runtime_error("TMVA-SOFIE: initialized tensor with name " + tensor_name + " already exists \n");
      }
      InitializedTensor new_tensor_ {GetTemplatedType(obj), new_tensor.GetShape() , static_cast<void>(new_tensor.GetData())};
      fInitializedTensors[tensor_name] = new_tensor_;
   }
*/

   void PrintRequiredInputTensors();
   void PrintInitializedTensors();
   void HeadInitializedTensors(std::string name, int n_print = 50);

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
