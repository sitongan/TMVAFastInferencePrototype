#include "SOFIE_common.hxx"


namespace TMVA{
namespace Experimental{
namespace SOFIE{

std::vector<Dim> ConvertShapeToDim(std::vector<size_t> shape){
   std::vector<Dim> fshape(shape.size());
   for (int i =0; i < shape.size(); i++){
      fshape[i].dim = shape[i];
   }
   return fshape;
}

std::size_t ConvertShapeToLength(std::vector<size_t> shape){
   std::size_t fLength = 1;
   for (auto& dim: shape) fLength *= dim;
   return fLength;
}


}//SOFIE
}//Experimental
}//TMVA
