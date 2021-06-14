#ifndef TMVA_SOFIE_ROPERATOR_RNN
#define TMVA_SOFIE_ROPERATOR_RNN

#include "RModel.hxx"
#include "ROperator.hxx"
#include "SOFIE_common.hxx"
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_RNN final : public ROperator {
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

   ROperator_RNN(std::vector<float> activation_alpha,
                 std::vector<float> activation_beta,
                 std::vector<std::string> activations, float clip,
                 std::string direction, size_t hidden_size, size_t layout,
                 std::string nameX, std::string nameW, std::string nameR,
                 std::string nameB, std::string nameSequence_lens,
                 std::string nameInitial_h, std::string nameY,
                 std::string nameY_h)
       : fAttrActivationAlpha(activation_alpha),
         fAttrActivationBeta(activation_beta), fAttrActivations(activations),
         fAttrClip(clip), fAttrDirection(direction),
         fAttrHiddenSize(hidden_size), fAttrLayout(layout),
         fNX(UTILITY::Clean_name(nameX)), fNW(UTILITY::Clean_name(nameW)),
         fNR(UTILITY::Clean_name(nameR)), fNB(UTILITY::Clean_name(nameB)),
         fNSequence_lens(UTILITY::Clean_name(nameSequence_lens)),
         fNInitial_h(UTILITY::Clean_name(nameInitial_h)),
         fNY(UTILITY::Clean_name(nameY)), fNY_h(UTILITY::Clean_name(nameY_h)) {
      if (std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw std::runtime_error(
             "TMVA SOFIE Encountered unsupported type parsing a RNN operator");
      }
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) {
      ETensorType out = input[0];
      return {out, out};
   }

   std::vector<std::vector<size_t>>
   ShapeInference(std::vector<std::vector<size_t>> input) {
      size_t num_directions = input[1][0];
      size_t hidden_size = input[1][1];
      if (fAttrLayout == 0) {
         size_t seq_length = input[0][0];
         size_t batch_size = input[0][1];
         std::vector<std::vector<size_t>> ret(
             {{seq_length, num_directions, batch_size, hidden_size},
              {num_directions, batch_size, hidden_size}});
         return ret;
      } else {
         size_t batch_size = input[0][0];
         size_t seq_length = input[0][1];
         std::vector<std::vector<size_t>> ret(
             {{batch_size, seq_length, num_directions, hidden_size},
              {batch_size, num_directions, hidden_size}});
         return ret;
      }
   }

   void Initialize(RModel &model) {
      // Check the input and output tensors
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA SOFIE RNN Op input tensor " + fNX +
                                  "  is not found in model.");
      }
      fShapeX = model.GetTensorShape(fNX);
      if (fShapeX.size() != 3) {
         throw std::runtime_error("TMVA SOFIE RNN Op input tensor " + fNX +
                                  " is not of 3 dimensions.");
      }
      if (!model.CheckIfTensorAlreadyExist(fNW)) {
         throw std::runtime_error("TMVA SOFIE RNN Op input tensor " + fNW +
                                  "  is not found in model.");
      }
      fShapeW = model.GetTensorShape(fNW);
      if (fShapeW.size() != 3) {
         throw std::runtime_error("TMVA SOFIE RNN Op input tensor " + fNR +
                                  " is not of 3 dimensions.");
      }
      if (!model.CheckIfTensorAlreadyExist(fNR)) {
         throw std::runtime_error("TMVA SOFIE RNN Op input tensor " + fNX +
                                  "  is not found in model.");
      }
      fShapeR = model.GetTensorShape(fNR);
      if (fShapeR.size() != 3) {
         throw std::runtime_error("TMVA SOFIE RNN Op input tensor " + fNR +
                                  " is not of 3 dimensions.");
      }
      if (fNB != "") {
         if (!model.CheckIfTensorAlreadyExist(fNB)) {
            throw std::runtime_error("TMVA SOFIE RNN op input tensor " + fNB +
                                     " is not  found in model.");
         }
         fShapeB = model.GetTensorShape(fNB);
         if (fShapeB.size() != 2) {
            throw std::runtime_error("TMVA SOFIE RNN op input tensor " + fNB +
                                     " is not of 2 dimensions.");
         }
         // TODO Broadcasting the bias
      }
      if (fNSequence_lens != "") {
         if (!model.CheckIfTensorAlreadyExist(fNSequence_lens)) {
            throw std::runtime_error("TMVA SOFIE RNN Op input tensor " +
                                     fNSequence_lens +
                                     "is not found in model.");
         }
         fShapeSequence_lens = model.GetTensorShape(fNSequence_lens);
         if (fShapeSequence_lens.size() != 1) {
            throw std::runtime_error("TMVA SOFIE RNN Op input tensor " +
                                     fNSequence_lens +
                                     " is not of 1 dimension.");
         }
      }
      if (fNInitial_h != "") {
         if (!model.CheckIfTensorAlreadyExist(fNInitial_h)) {
            throw std::runtime_error("TMVA SOFIE RNN Op input tensor " +
                                     fNInitial_h + " is not found in model.");
         }
         fShapeInitial_h = model.GetTensorShape(fNInitial_h);
         if (fShapeInitial_h.size() != 3) {
            throw std::runtime_error("TMVA SOFIE RNN Op input tensor " +
                                     fNInitial_h + " is not of 3 dimensions.");
         }
      }
      if (fNY != "") {
         fShapeY = ShapeInference({fShapeX, fShapeW})[0];
      }
      if (fNY_h != "") {
         fShapeY_h = ShapeInference({fShapeX, fShapeW})[1];
      }
      // Check the attributes
      for (auto &activation : fAttrActivations) {
         if (activation != "Relu" && activation != "Tanh" &&
             activation != "Sigmoid" && activation != "Affine" &&
             activation != "LeakyRelu" && activation != "ThresholdRelu" &&
             activation != "ScaledTanh" && activation != "HardSigmoid" &&
             activation != "Elu" && activation != "Softsign" &&
             activation != "Softplus") {
            throw std::runtime_error("TMVA SOFIE - Activation function " +
                                     activation + " not implemented");
         }
      }
      if (fAttrDirection != "forward" && fAttrDirection != "backward" &&
          fAttrDirection != "bidirectional") {
         throw std::runtime_error(
             "TMVA SOFIE - Invalid RNN direction fAttrDirection = " +
             fAttrDirection);
      }
      if (fAttrHiddenSize != fShapeW[1]) {
         throw std::runtime_error(
             "TMVA SOFIE - fAttrHiddenSize must be equal to " +
             std::to_string(fShapeW[1]));
      }
      if (fAttrLayout > 1) {
         throw std::runtime_error("TMVA SOFIE - Layout fAttrLayout = " +
                                  std::to_string(fAttrLayout) +
                                  " must be 0 (timewise) or 1 (batchwise)");
      }
      if (fAttrActivations.empty()) {
         if (fAttrDirection == "bidirectional") {
            fAttrActivations = {"Tanh", "Tanh"};
         } else {
            fAttrActivations = {"Tanh"};
         }
      }
   }

   std::string Generate(std::string OpName) {
      OpName = "op_" + OpName;
      std::stringstream out;
      // TODO Implement the forward pass of RNN
      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
