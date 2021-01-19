#ifndef TMVA_SOFIE_RMODELPARSER_ONNX
#define TMVA_SOFIE_RMODELPARSER_ONNX

#include "onnx.proto3.pb.h"

#include "SOFIE_common.hxx"
#include "RModel.hxx"


#include <string>
#include <fstream>
#include <memory>
#include <ctime>


namespace TMVA{
namespace Experimental{
namespace SOFIE{


namespace INTERNAL{
//unique_ptr<ROperator> make_ROperator_Gemm(const onnx::NodeProto& nodeproto, RModel& this_graph);
//ROperator* make_ROperator_Transpose(const onnx::NodeProto& nodeproto, const onnx::GraphProto& graphproto, const std::unordered_map<std::string, size_t>& tensorname2idx);
//unique_ptr<ROperator> make_ROperator_Relu(const onnx::NodeProto& nodeproto, RModel& this_graph);

using factoryMethodMap = std::unordered_map<std::string, ROperator* (*)(const onnx::NodeProto&, const onnx::GraphProto&, const std::unordered_map<std::string, size_t>&)>;
const factoryMethodMap mapOptypeOperator = {
      //{"Gemm", &make_ROperator_Gemm}//,
      //{"Transpose", &make_ROperator_Transpose}//,
      //{"Relu", &make_ROperator_Relu}
   };


ROperator* make_ROperator(size_t idx, const onnx::GraphProto& graphproto, const std::unordered_map<std::string, size_t>& tensorname2idx);
}//INTERNAL



class RModelParser_ONNX{
public:
   RModel Parse(std::string filename){
      char sep = '/';
      #ifdef _WIN32
         sep = '\\';
      #endif
      size_t i = filename.rfind(sep, filename.length());
      std::string modelname;
      if (i != std::string::npos){
         filename = (filename.substr(i+1, filename.length() - i));
      }



      std::time_t ttime = std::time(0);
      std::tm* gmt_time = std::gmtime(&ttime);
      std::string parsetime (std::asctime(gmt_time));




      GOOGLE_PROTOBUF_VERIFY_VERSION;
      //model I/O
      onnx::ModelProto model;
      RModel rmodel(filename, parsetime);

      std::fstream input(filename, std::ios::in | std::ios::binary);
      if (!model.ParseFromIstream(&input)){
         throw std::runtime_error("TMVA::SOFIE - Failed to parse onnx file");
      }

      const onnx::GraphProto& graph = model.graph(); //not a memory leak. model freed automatically at the end.
      google::protobuf::ShutdownProtobufLibrary();

      std::unordered_set<std::string> initializer_names;
      for (int i=0; i < graph.initializer_size(); i++){
         initializer_names.insert(graph.initializer(i).name());
      }

      std::unordered_map<std::string, size_t> tensorname2idx;
      for (int i=0; i < graph.input_size(); i++){
         tensorname2idx.emplace(graph.input(i).name(), i);
         if (initializer_names.find(graph.input(i).name()) != initializer_names.end())  continue;

         //input datanode is not a weight node (has no initializer)
         const onnx::ValueInfoProto& valueinfoproto = graph.input(i);
         std::string input_name = valueinfoproto.name();
         ETensorType type = static_cast<ETensorType>(valueinfoproto.type().tensor_type().elem_type());
         if (type != ETensorType::FLOAT){
            throw std::runtime_error("TMVA::SOFIE Data type in input tensor " + input_name + " not supported!\n");
         }

         std::vector<Dim> fShape;
         if (!valueinfoproto.type().tensor_type().has_shape()) throw std::runtime_error("TMVA::SOFIE datanode with no shape restrictions is not supported yet");
         for (int i = 0; i < valueinfoproto.type().tensor_type().shape().dim_size(); i++){
            Dim dim;
            if (valueinfoproto.type().tensor_type().shape().dim(i).value_case() == onnx::TensorShapeProto_Dimension::ValueCase::kDimValue){
               dim.dim = valueinfoproto.type().tensor_type().shape().dim(i).dim_value();
            }else if (valueinfoproto.type().tensor_type().shape().dim(i).value_case() == onnx::TensorShapeProto_Dimension::ValueCase::kDimParam){
               dim.isParam = true;
               dim.param = valueinfoproto.type().tensor_type().shape().dim(i).dim_param();
            }else{
               throw std::runtime_error("TMVA::SOFIE ONNX file error: Valueinfoproto " + input_name + " has neither dim_value nor dim_param! \n");
            }
            fShape.push_back(dim);
         }
         if (valueinfoproto.type().tensor_type().shape().dim_size() == 0){
            Dim dim;
            dim.dim = 1;
            fShape.push_back(dim);
         } //in case this TensorShapeProto has no dimension message: ONNX IR defines this to be a scalar

         rmodel.AddInputTensorInfo(input_name, type, fShape);

      }

      for (int i=0; i < graph.initializer_size(); i++){
         onnx::TensorProto* tensorproto = const_cast<onnx::TensorProto*>(&graph.initializer(i));
         std::vector<std::size_t> fShape;
         std::size_t fLength = 1;
         for (int i = 0; i < tensorproto->dims_size(); i++){
            fShape.push_back(tensorproto->dims(i));
            fLength *= tensorproto->dims(i);
         }

         std::string input_name = graph.initializer(i).name();

         switch(static_cast<ETensorType>(graph.initializer(i).data_type())){
            case ETensorType::FLOAT : {
               //void* data = malloc (fLength * sizeof(float));
               std::shared_ptr<void> data(malloc(fLength * sizeof(float)), free);

               if (tensorproto->raw_data().empty() == false){
                  auto raw_data_ptr = reinterpret_cast<float*>(const_cast<char*>(tensorproto->raw_data().c_str()));
                  std::memcpy(data.get(), raw_data_ptr, fLength * sizeof(float));
               }else{
                  tensorproto->mutable_float_data()->ExtractSubrange(0, tensorproto->float_data_size(), static_cast<float*>(data.get()));
               }

               rmodel.AddInitializedTensors(input_name, ETensorType::FLOAT, fShape, data);
               break;
            }
            default: throw std::runtime_error("Data type in weight tensor " + graph.initializer(i).name() + " not supported!\n");
         }
      }


/*
      for (int i=0; i < graph.node_size(); i++){
         rmodel.addOperator(std::move(std::unique_ptr<ROperator>(INTERNAL::make_ROperator(i, graph, tensorname2idx))));
      }
*/


      return rmodel;

   }

};



}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RMODELPARSER_ONNX
