#ifndef TMVA_SOFIE_ROPERATOR
#define TMVA_SOFIE_ROPERATOR

#include <vector>
#include <memory>

#include "SOFIE_common.hxx"
//#include "RModel.hxx"


namespace TMVA{
namespace Experimental{
namespace SOFIE{

class ROperator{

public:
   virtual const std::vector<std::vector<size_t>> shapeInference() = 0;
   //virtual void Initialize(const RModel&); //

   //virtual void Forward_reference() = 0;
   //irtual void Forward_blas() = 0;
   virtual ~ROperator(){}

   typedef std::int64_t int_t;


};



}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_OPERATOR
