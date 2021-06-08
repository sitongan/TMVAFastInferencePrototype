#ifndef TMVA_SOFIE_ROPERATOR_RNN
#define TMVA_SOFIE_ROPERATOR_RNN

#include "SOFIE_common.hxx"
#include "ROperator.hxx"
#include "RModel.hxx"
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>


namespace TMVA {
namespace Experimental {
namespace SOFIE {

template<typename T>
class ROperator_RNN final : public ROperator
{
private:
   std::vector<float> fAttrActivationAlpha;
   std::vector<float> fAttrActivationBeta;
   std::vector<std::string> fAttrActivations;
   float fAttrClip;
   std::string fAttrDirection;
   size_t fAttrHiddenSize;
   size_t fAttrLayout;

   std::string fNX;
   std::string fNW;
   std::string fNR;
   std::string fNB;
   std::string fNSequence_lens;
   std::string fNInitial_h;
   std::string fNY;
   std::string fNY_h;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeW;
   std::vector<size_t> fShapeR;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeSequence_lens;
   std::vector<size_t> fShapeInitial_h;
   std::vector<size_t> fShapeY;
   std::vector<size_t> fShapeY_h;

   std::string fType;

public:

   ROperator_RNN() = delete;

   ROperator_RNN(std::vector<float> activation_alpha, std::vector<float> activation_beta, std::vector<std::string> activations, float clip, std::string direction, size_t hidden_size, size_t layout, std::string nameX, std::string nameW, std::string nameR, std::string nameB, std::string nameSequence_lens, std::string nameInitial_h, std::string nameY, std::string nameY_h):
   fAttrActivationAlpha(activation_alpha), fAttrActivationBeta(activation_beta), fAttrActivations(activations),
   fAttrClip(clip), fAttrDirection(direction), fAttrHiddenSize(hidden_size), fAttrLayout(layout),
   fNX(UTILITY::Clean_name(nameX)), fNW(UTILITY::Clean_name(nameW)), fNR(UTILITY::Clean_name(nameR)),
   fNB(UTILITY::Clean_name(nameB)), fNSequence_lens(UTILITY::Clean_name(nameSequence_lens)), fNInitial_h(UTILITY::Clean_name(nameInitial_h)),
   fNY(UTILITY::Clean_name(nameY)), fNY_h(UTILITY::Clean_name(nameY_h))
   {
      if (std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw
            std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a RNN operator");
      }
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) {
      ETensorType out = input[0];
      return {out, out};
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) {
      // TODO Infer the shape of Y and Y_h
   }

   void Initialize(RModel& model) {
      // TODO Initialize the model
   }

   std::string Generate(std::string OpName) {
      OpName = "op_" + OpName;
      std::stringstream out;
      // TODO Implement the forward pass of RNN
      return out.str();
   }
};

}
}
}

#endif
