#ifndef TMVA_SOFIE_ROPERATOR
#define TMVA_SOFIE_ROPERATOR

#include <vector>

#include "SOFIE_common.hxx"



namespace TMVA{
namespace Experimental{
namespace SOFIE{

class ROperator{

public:
   virtual const std::vector<std::vector<size_t>> shapeInference() = 0;
   virtual void Forward_reference() = 0;
   virtual void Forward_blas() = 0;
   virtual ~ROperator(){}
};



}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_OPERATOR
