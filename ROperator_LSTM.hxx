#ifndef TMVA_SOFIE_ROPERATOR_LSTM
#define TMVA_SOFIE_ROPERATOR_LSTM

#include "RModel.hxx"
#include "ROperator.hxx"
#include "SOFIE_common.hxx"
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T> class ROperator_LSTM final : public ROperator {
 private:
   std::vector<float> fAttrActivationAlpha;
   std::vector<float> fAttrActivationBeta;
   std::vector<std::string> fAttrActivations;
   float fAttrClip;
   std::string fAttrDirection;
   size_t fAttrHiddenSize;
   size_t fAttrInputForget;
   size_t fAttrLayout;

   std::string fNX;
   std::string fNW;
   std::string fNR;
   std::string fNB;
   std::string fNSequence_lens;
   std::string fNInitial_h;
   std::string fNInitial_c;
   std::string fNY;
   std::string fNY_h;
   std::string fNY_c;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeW;
   std::vector<size_t> fShapeR;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeSequence_lens;
   std::vector<size_t> fShapeInitial_h;
   std::vector<size_t> fShapeInitial_c;
   std::vector<size_t> fShapeY;
   std::vector<size_t> fShapeY_h;
   std::vector<size_t> fShapeY_c;

   std::string fType;

 public:
   ROperator_LSTM() = delete;

   ROperator_LSTM(std::vector<float> activation_alpha,
                 std::vector<float> activation_beta,
                 std::vector<std::string> activations, float clip,
                 std::string direction, size_t hidden_size,
                 size_t input_forget, size_t layout,
                 std::string nameX, std::string nameW, std::string nameR,
                 std::string nameB, std::string nameSequence_lens,
                 std::string nameInitial_h, std::string nameInitial_c,
                 std::string nameY, std::string nameY_h, std::string nameY_c)
       : fAttrActivationAlpha(activation_alpha),
         fAttrActivationBeta(activation_beta), fAttrActivations(activations),
         fAttrClip(clip), fAttrDirection(direction), fAttrHiddenSize(hidden_size),
         fAttrInputForget(input_forget), fAttrLayout(layout),
         fNX(UTILITY::Clean_name(nameX)), fNW(UTILITY::Clean_name(nameW)),
         fNR(UTILITY::Clean_name(nameR)), fNB(UTILITY::Clean_name(nameB)),
         fNSequence_lens(UTILITY::Clean_name(nameSequence_lens)),
         fNInitial_h(UTILITY::Clean_name(nameInitial_h)),
         fNInitial_c(UTILITY::Clean_name(nameInitial_c)),
         fNY(UTILITY::Clean_name(nameY)), fNY_h(UTILITY::Clean_name(nameY_h)),
         fNY_c(UTILITY::Clean_name(nameY_c)) {
      if (std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw std::runtime_error(
             "TMVA SOFIE Encountered unsupported type parsing a LSTM operator");
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
              {num_directions, batch_size, hidden_size},
              {num_directions, batch_size, hidden_size}});
         return ret;
      } else {
         size_t batch_size = input[0][0];
         size_t seq_length = input[0][1];
         std::vector<std::vector<size_t>> ret(
             {{batch_size, seq_length, num_directions, hidden_size},
              {batch_size, num_directions, hidden_size},
              {batch_size, num_directions, hidden_size}});
         return ret;
      }
   }

   void Initialize(RModel &model) {
      // Check the input and output tensors
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA SOFIE LSTM Op input tensor " + fNX + "  is not found in model.");
      }
      fShapeX = model.GetTensorShape(fNX);
      if (fShapeX.size() != 3) {
         throw std::runtime_error("TMVA SOFIE LSTM Op input tensor " + fNX + " is not of 3 dimensions.");
      }
      if (!model.CheckIfTensorAlreadyExist(fNW)) {
         throw std::runtime_error("TMVA SOFIE LSTM Op input tensor " + fNW + "  is not found in model.");
      }
      fShapeW = model.GetTensorShape(fNW);
      if (fShapeW.size() != 3) {
         throw std::runtime_error("TMVA SOFIE LSTM Op input tensor " + fNW + " is not of 3 dimensions.");
      }
      if (!model.CheckIfTensorAlreadyExist(fNR)) {
         throw std::runtime_error("TMVA SOFIE LSTM Op input tensor " + fNR + "  is not found in model.");
      }
      fShapeR = model.GetTensorShape(fNR);
      if (fShapeR.size() != 3) {
         throw std::runtime_error("TMVA SOFIE LSTM Op input tensor " + fNR + " is not of 3 dimensions.");
      }
      if (!fNB.empty()) {
         if (!model.CheckIfTensorAlreadyExist(fNB)) {
            throw std::runtime_error("TMVA SOFIE LSTM op input tensor " + fNB + " is not  found in model.");
         }
         fShapeB = model.GetTensorShape(fNB);
         if (fShapeB.size() != 2 && fShapeB.size() != 5) {
            throw std::runtime_error("TMVA SOFIE LSTM op input tensor " + fNB + " is not of 2 or 4 dimensions.");
         }
         if (fShapeB.size() == 2) {
            // Broadcasting the bias
            auto original_data = model.GetInitializedTensorData(fNB);
            size_t num_directions = fShapeW[0];
            size_t seq_length = (fAttrLayout == 0)? fShapeX[0] : fShapeX[1];
            size_t batch_size = (fAttrLayout == 0)? fShapeX[1] : fShapeX[0];
            if (fType == "float") {
               float *original_bias = static_cast<float*>(original_data.get());
               float *new_bias = new float[4 * num_directions * seq_length * batch_size * fAttrHiddenSize];
               for (size_t gate = 0; gate < 4; gate++) {
                  float sum[fAttrHiddenSize];
                  for (size_t direction = 0; direction < num_directions; direction++) {
                     size_t offset = direction * 8 * fAttrHiddenSize + gate * fAttrHiddenSize;
                     for (size_t h = 0; h < fAttrHiddenSize; h++) {
                        sum[h] = original_bias[offset + h] + original_bias[offset + h + 4 * fAttrHiddenSize];
                     }
                     for (size_t seq = 0; seq < seq_length; seq++) {
                        for (size_t batch = 0; batch < batch_size; batch++) {
                           size_t bias_offset = gate * num_directions * seq_length * batch_size * fAttrHiddenSize + direction * seq_length * batch_size * fAttrHiddenSize + seq * batch_size * fAttrHiddenSize + batch * fAttrHiddenSize;
                           std::copy(sum, sum + fAttrHiddenSize, new_bias + bias_offset);
                        }
                     }
                  }
               }
               std::vector<size_t> new_bias_shape = {4, num_directions, seq_length, batch_size, fAttrHiddenSize};
               std::shared_ptr<void> new_bias_ptr(new_bias, std::default_delete<float[]>());
               model.UpdateInitializedTensor(fNB, model.GetTensorType(fNB), new_bias_shape, new_bias_ptr);
               fShapeB = model.GetTensorShape(fNB);
            }
         }
      }
      if (!fNSequence_lens.empty()) {
         if (!model.CheckIfTensorAlreadyExist(fNSequence_lens)) {
            throw std::runtime_error("TMVA SOFIE LSTM Op input tensor " +
                                     fNSequence_lens +
                                     "is not found in model.");
         }
         fShapeSequence_lens = model.GetTensorShape(fNSequence_lens);
         if (fShapeSequence_lens.size() != 1) {
            throw std::runtime_error("TMVA SOFIE LSTM Op input tensor " +
                                     fNSequence_lens +
                                     " is not of 1 dimension.");
         }
      }
      if (!fNInitial_h.empty()) {
         if (!model.CheckIfTensorAlreadyExist(fNInitial_h)) {
            throw std::runtime_error("TMVA SOFIE LSTM Op input tensor " +
                                     fNInitial_h + " is not found in model.");
         }
         fShapeInitial_h = model.GetTensorShape(fNInitial_h);
         if (fShapeInitial_h.size() != 3) {
            throw std::runtime_error("TMVA SOFIE LSTM Op input tensor " +
                                     fNInitial_h + " is not of 3 dimensions.");
         }
      }
      if (!fNInitial_c.empty()) {
         if (!model.CheckIfTensorAlreadyExist(fNInitial_c)) {
            throw std::runtime_error("TMVA SOFIE LSTM Op input tensor " +
                                     fNInitial_c + " is not found in model.");
         }
         fShapeInitial_c = model.GetTensorShape(fNInitial_c);
         if (fShapeInitial_c.size() != 3) {
            throw std::runtime_error("TMVA SOFIE LSTM Op input tensor " +
                                     fNInitial_c + " is not of 3 dimensions.");
         }
      }
      if (!fNY.empty()) {
         fShapeY = ShapeInference({fShapeX, fShapeW})[0];
         if (!model.CheckIfTensorAlreadyExist(fNY)) {
            model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
         }
      }
      if (!fNY_h.empty()) {
         fShapeY_h = ShapeInference({fShapeX, fShapeW})[1];
         if (!model.CheckIfTensorAlreadyExist(fNY_h)) {
            model.AddIntermediateTensor(fNY_h, model.GetTensorType(fNX), fShapeY_h);
         }
      }
      if (!fNY_c.empty()) {
         fShapeY_c = ShapeInference({fShapeX, fShapeW})[2];
         if (!model.CheckIfTensorAlreadyExist(fNY_c)) {
            model.AddIntermediateTensor(fNY_c, model.GetTensorType(fNX), fShapeY_c);
         }
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
             "TMVA SOFIE - Invalid LSTM direction fAttrDirection = " +
             fAttrDirection);
      }
      if (4 * fAttrHiddenSize != fShapeW[1]) {
         throw std::runtime_error(
             "TMVA SOFIE - fAttrHiddenSize must be equal to " +
             std::to_string(fShapeW[1] / 4));
      }
      if (fAttrInputForget > 1) {
         throw std::runtime_error(
            "TMVA SOFIE - fAttrInputForget = " + std::to_string(fAttrInputForget)
            + " must be 0 or 1.");
      }
      if (fAttrLayout > 1) {
         throw std::runtime_error("TMVA SOFIE - Layout fAttrLayout = " +
                                  std::to_string(fAttrLayout) +
                                  " must be 0 (timewise) or 1 (batchwise)");
      }
      if (fAttrActivations.empty()) {
         if (fAttrDirection == "bidirectional") {
            fAttrActivations = {"Sigmoid", "Tanh", "Tanh", "Sigmoid", "Tanh", "Tanh"};
         } else {
            fAttrActivations = {"Sigmoid", "Tanh", "Tanh"};
         }
      }
   }

   std::string Generate(std::string OpName) {
      OpName = "op_" + OpName;
      std::stringstream out;

      size_t seq_length = (fAttrLayout == 0) ? fShapeX[0] : fShapeX[1];
      size_t batch_size = (fAttrLayout == 0) ? fShapeX[1] : fShapeX[0];
      size_t input_size = fShapeX[2];
      size_t num_directions = fShapeW[0];

      // TODO Implement the forward pass of LSTM

      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
