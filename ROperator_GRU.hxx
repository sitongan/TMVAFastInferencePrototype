#ifndef TMVA_SOFIE_ROPERATOR_GRU
#define TMVA_SOFIE_ROPERATOR_GRU

#include "RModel.hxx"
#include "ROperator.hxx"
#include "SOFIE_common.hxx"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T> class ROperator_GRU final : public ROperator {
 private:
   std::vector<float> fAttrActivationAlpha;
   std::vector<float> fAttrActivationBeta;
   std::vector<std::string> fAttrActivations;
   float fAttrClip;
   std::string fAttrDirection;
   size_t fAttrHiddenSize;
   size_t fAttrLayout;
   size_t fAttrLinearBeforeReset;

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
   ROperator_GRU() = delete;

   ROperator_GRU(std::vector<float> activation_alpha,
                 std::vector<float> activation_beta,
                 std::vector<std::string> activations, float clip,
                 std::string direction, size_t hidden_size,
                 size_t layout, size_t linear_before_reset,
                 std::string nameX, std::string nameW, std::string nameR,
                 std::string nameB, std::string nameSequence_lens,
                 std::string nameInitial_h, std::string nameY, std::string nameY_h)
       : fAttrActivationAlpha(activation_alpha),
         fAttrActivationBeta(activation_beta), fAttrActivations(activations),
         fAttrClip(clip), fAttrDirection(direction), fAttrHiddenSize(hidden_size),
         fAttrLayout(layout), fAttrLinearBeforeReset(linear_before_reset),
         fNX(UTILITY::Clean_name(nameX)), fNW(UTILITY::Clean_name(nameW)),
         fNR(UTILITY::Clean_name(nameR)), fNB(UTILITY::Clean_name(nameB)),
         fNSequence_lens(UTILITY::Clean_name(nameSequence_lens)),
         fNInitial_h(UTILITY::Clean_name(nameInitial_h)),
         fNY(UTILITY::Clean_name(nameY)), fNY_h(UTILITY::Clean_name(nameY_h)) {
      if (std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw std::runtime_error(
             "TMVA SOFIE Encountered unsupported type parsing a GRU operator");
      }
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) {
      ETensorType out = input[0];
      return {out, out};
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) {
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
              {batch_size, num_directions, hidden_size}});
         return ret;
      }
   }

   void Initialize(RModel &model) {
      // Check the input and output tensors
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA SOFIE GRU Op input tensor " + fNX + "  is not found in model.");
      }
      fShapeX = model.GetTensorShape(fNX);
      if (fShapeX.size() != 3) {
         throw std::runtime_error("TMVA SOFIE GRU Op input tensor " + fNX + " is not of 3 dimensions.");
      }
      if (!model.CheckIfTensorAlreadyExist(fNW)) {
         throw std::runtime_error("TMVA SOFIE GRU Op input tensor " + fNW + "  is not found in model.");
      }
      fShapeW = model.GetTensorShape(fNW);
      if (fShapeW.size() != 3) {
         throw std::runtime_error("TMVA SOFIE GRU Op input tensor " + fNW + " is not of 3 dimensions.");
      }
      if (!model.CheckIfTensorAlreadyExist(fNR)) {
         throw std::runtime_error("TMVA SOFIE GRU Op input tensor " + fNR + "  is not found in model.");
      }
      fShapeR = model.GetTensorShape(fNR);
      if (fShapeR.size() != 3) {
         throw std::runtime_error("TMVA SOFIE GRU Op input tensor " + fNR + " is not of 3 dimensions.");
      }
      if (!fNB.empty()) {
         if (!model.CheckIfTensorAlreadyExist(fNB)) {
            throw std::runtime_error("TMVA SOFIE GRU op input tensor " + fNB + " is not  found in model.");
         }
         fShapeB = model.GetTensorShape(fNB);
         if (fShapeB.size() != 2 && fShapeB.size() != 4) {
            throw std::runtime_error("TMVA SOFIE GRU op input tensor " + fNB + " is not of 2 or 4 dimensions.");
         }
         if (fShapeB.size() == 2) {
            // Broadcasting the bias
            auto original_data = model.GetInitializedTensorData(fNB);
            size_t num_directions = fShapeW[0];
            size_t batch_size = (fAttrLayout == 0)? fShapeX[1] : fShapeX[0];
            if (fType == "float") {
               float *original_bias = static_cast<float*>(original_data.get());
               float *new_bias = new float[num_directions * 6 * batch_size * fAttrHiddenSize];
               for (size_t direction = 0; direction < num_directions; direction++) {
                  for (size_t i = 0; i < 6; i++) {
                     for (size_t batch = 0; batch < batch_size; batch++) {
                        size_t bias_offset = direction * 6 * fAttrHiddenSize + i * fAttrHiddenSize;
                        size_t offset = direction * 6 * batch_size * fAttrHiddenSize
                           + i * batch_size * fAttrHiddenSize + batch * fAttrHiddenSize;
                        std::copy(original_bias + bias_offset, original_bias + bias_offset + fAttrHiddenSize,
                           new_bias + offset);
                     }
                  }
               }

               std::vector<size_t> new_bias_shape = {num_directions, 6, batch_size, fAttrHiddenSize};
               std::shared_ptr<void> new_bias_ptr(new_bias, std::default_delete<float[]>());
               model.UpdateInitializedTensor(fNB, model.GetTensorType(fNB), new_bias_shape, new_bias_ptr);
               fShapeB = model.GetTensorShape(fNB);
            }
         }
      }
      if (!fNSequence_lens.empty()) {
         if (!model.CheckIfTensorAlreadyExist(fNSequence_lens)) {
            throw std::runtime_error("TMVA SOFIE GRU Op input tensor " +
                                     fNSequence_lens +
                                     "is not found in model.");
         }
         fShapeSequence_lens = model.GetTensorShape(fNSequence_lens);
         if (fShapeSequence_lens.size() != 1) {
            throw std::runtime_error("TMVA SOFIE GRU Op input tensor " +
                                     fNSequence_lens +
                                     " is not of 1 dimension.");
         }
      }
      if (!fNInitial_h.empty()) {
         if (!model.CheckIfTensorAlreadyExist(fNInitial_h)) {
            throw std::runtime_error("TMVA SOFIE GRU Op input tensor " +
                                     fNInitial_h + " is not found in model.");
         }
         fShapeInitial_h = model.GetTensorShape(fNInitial_h);
         if (fShapeInitial_h.size() != 3) {
            throw std::runtime_error("TMVA SOFIE GRU Op input tensor " +
                                     fNInitial_h + " is not of 3 dimensions.");
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
             "TMVA SOFIE - Invalid GRU direction fAttrDirection = " +
             fAttrDirection);
      }
      if (3 * fAttrHiddenSize != fShapeW[1]) {
         throw std::runtime_error(
             "TMVA SOFIE - fAttrHiddenSize must be equal to " +
             std::to_string(fShapeW[1] / 3));
      }
      if (fAttrLayout > 1) {
         throw std::runtime_error("TMVA SOFIE - Layout fAttrLayout = " +
                                  std::to_string(fAttrLayout) +
                                  " must be 0 (timewise) or 1 (batchwise)");
      }
      if (fAttrLinearBeforeReset > 1) {
         throw std::runtime_error(
            "TMVA SOFIE - fAttrInputForget = " + std::to_string(fAttrLinearBeforeReset)
            + " must be 0 or 1.");
      }
      if (fAttrActivations.empty()) {
         if (fAttrDirection == "bidirectional") {
            fAttrActivations = {"Sigmoid", "Tanh", "Sigmoid", "Tanh"};
         } else {
            fAttrActivations = {"Sigmoid", "Tanh"};
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

      // set the input
      if (fAttrLayout == 0) {
         if (fType == "float") {
            out << "\t" << "float *" << OpName << "_input = tensor_" << fNX << ";\n";
         }
      } else {
         if (fType == "float") {
            out << "\t" << "float " << OpName << "_input[" << seq_length * batch_size * input_size << "];\n";
         }
         out << "\t" << "for(size_t seq = 0; seq < " << seq_length << "; seq++) {\n";
         out << "\t" << "\t" << "for(size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
         out << "\t" << "\t" << "\t" << "for(size_t i = 0; i < " << input_size << "; i++) {\n";
         out << "\t" << "\t" << "\t" << "\t" << OpName << "_input[seq * " << batch_size * input_size
             << " + batch * " << input_size << " + i] = " << "tensor_" << fNX << "[batch * "
             << seq_length * input_size << " + seq * " << input_size << " + i];\n";
         out << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "}\n";
         out << "\t" << "}\n";
      }

      // Set the initial hidden state
      if (!fNInitial_h.empty()) {
         if (fAttrLayout == 0) {
            if (fType == "float") {
               out << "\t" << "float *" << OpName << "_initial_hidden_state = " << " tensor_"
                   << fNInitial_h << ";\n";
            }
         } else {
            if (fType == "float") {
               out << "\t" << "float " << OpName << "_initial_hidden_state[" << num_directions * batch_size *
                   fAttrHiddenSize << "];\n";
            }
            for (size_t direction = 0; direction < num_directions; direction++) {
               out << "\t" << "for(size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
               out << "\t" << "\t" << "for(size_t h = 0; h < " << fAttrHiddenSize << "; h++) {\n";
               out << "\t" << "\t" << "\t" << OpName << "_initial_hidden_state["
                   << direction * batch_size * fAttrHiddenSize << " + batch * " << fAttrHiddenSize
                   << " + h] = tensor_" << fNInitial_h << "[batch * " << num_directions * fAttrHiddenSize
                   << " + " << direction * fAttrHiddenSize << " + h];\n";
               out << "\t" << "\t" << "}\n";
               out << "\t" << "}\n";
            }
         }
      }

      // Set the feedforward
      size_t feedforward_size = seq_length * batch_size * fAttrHiddenSize;
      if (fType == "float") {
         out << "\t" << "float " << OpName << "_f_update_gate[" << feedforward_size << "];\n";
         out << "\t" << "float " << OpName << "_f_reset_gate[" << feedforward_size << "];\n";
         out << "\t" << "float " << OpName << "_f_hidden_gate[" << feedforward_size << "];\n";
      }
      // Set the gates
      size_t hidden_state_size = seq_length * num_directions * batch_size * fAttrHiddenSize;
      if (fType == "float") {
         out << "\t" << "float " << OpName << "_update_gate[" << hidden_state_size << "];\n";
         out << "\t" << "float " << OpName << "_reset_gate[" << hidden_state_size << "];\n";
         out << "\t" << "float " << OpName << "_hidden_gate[" << hidden_state_size << "];\n";
      }
      // Set the hidden state
      if (fAttrLayout == 0 && !fNY.empty()) {
         if (fType == "float") {
            out << "\t" << "float *" << OpName << "_hidden_state = tensor_" << fNY << ";\n";
         }
      } else {
         if (fType == "float") {
            out << "\t" << "float " << OpName << "_hidden_state[" << hidden_state_size << "];\n";
         }
      }

      if (fType == "float") {
         out << "\t" << "float " << OpName << "_feedback[" << batch_size * fAttrHiddenSize << "];\n";
      }

      out << "\t" << "char " << OpName << "_transA = 'N';\n";
      out << "\t" << "char " << OpName << "_transB = 'T';\n";
      out << "\t" << "int " << OpName << "_m = " << seq_length * batch_size << ";\n";
      out << "\t" << "int " << OpName << "_m2 = " << batch_size << ";\n";
      out << "\t" << "int " << OpName << "_n = " << fAttrHiddenSize << ";\n";
      out << "\t" << "int " << OpName << "_k = " << input_size << ";\n";
      if (fType == "float") {
         out << "\t" << "float " << OpName << "_alpha = 1.;\n";
         out << "\t" << "float " << OpName << "_beta = 0.;\n";
      }
      if (!fNB.empty()) {
         out << "\t" << "int " << OpName << "_bias_size = " << seq_length * batch_size * fAttrHiddenSize << ";\n";
      }
      out << "\t" << "int " << OpName << "_incx = 1;\n";
      out << "\t" << "int " << OpName << "_incy = 1;\n";
      out << "\t" << "int " << OpName << "_feedback_size = " << batch_size * fAttrHiddenSize << ";\n";

      for (size_t direction = 0; direction < num_directions; direction++) {
         if (direction == 0) {
            if (fType == "float") {
               // f_update_gate = input * weight_z^T
               out << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                   << OpName <<"_n, &" << OpName << "_m, &" << OpName << "_k, &" << OpName << "_alpha, tensor_"
                   << fNW << ", &" << OpName << "_k, " << OpName << "_input, &" << OpName << "_k, &"
                  << OpName << "_beta, " << OpName << "_f_update_gate, &" << OpName << "_n);\n";
               // f_reset_gate = input * weight_r^T
               size_t wr_offset = fAttrHiddenSize * input_size;
               out << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                   << OpName <<"_n, &" << OpName << "_m, &" << OpName << "_k, &" << OpName << "_alpha, tensor_"
                  << fNW << " + " << wr_offset << ", &" << OpName << "_k, " << OpName << "_input, &"
                  << OpName << "_k, &" << OpName << "_beta, " << OpName << "_f_reset_gate, &" << OpName << "_n);\n";
               // f_hidden_gate = input * weight_h^T
               size_t wh_offset = 2 * fAttrHiddenSize * input_size;
               out << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                   << OpName <<"_n, &" << OpName << "_m, &" << OpName << "_k, &" << OpName << "_alpha, tensor_"
                  << fNW << " + " << wh_offset << ", &" << OpName << "_k, " << OpName << "_input, &"
                  << OpName << "_k, &" << OpName << "_beta, " << OpName << "_f_hidden_gate, &" << OpName << "_n);\n";
            }
         } else {
            if (fType == "float") {
               // f_update_gate = input * weight_z^T
               size_t wz_offset = 3 * fAttrHiddenSize * input_size;
               out << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                   << OpName <<"_n, &" << OpName << "_m, &" << OpName << "_k, &" << OpName << "_alpha, tensor_"
                  << fNW << " + " << wz_offset << ", &" << OpName << "_k, " << OpName << "_input, &"
                  << OpName << "_k, &" << OpName << "_beta, " << OpName << "_f_update_gate, &" << OpName << "_n);\n";
               // f_reset_gate = input * weight_r^T
               size_t wr_offset = 3 * fAttrHiddenSize * input_size + fAttrHiddenSize * input_size;
               out << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                   << OpName <<"_n, &" << OpName << "_m, &" << OpName << "_k, &" << OpName << "_alpha, tensor_"
                  << fNW << " + " << wr_offset << ", &" << OpName << "_k, " << OpName << "_input, &"
                  << OpName << "_k, &" << OpName << "_beta, " << OpName << "_f_reset_gate, &" << OpName << "_n);\n";
               // f_hidden_gate = input * weight_h^T
               size_t wh_offset = 3 * fAttrHiddenSize * input_size + 2 * fAttrHiddenSize * input_size;
               out << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                   << OpName <<"_n, &" << OpName << "_m, &" << OpName << "_k, &" << OpName << "_alpha, tensor_"
                  << fNW << " + " << wh_offset << ", &" << OpName << "_k, " << OpName << "_input, &"
                  << OpName << "_k, &" << OpName << "_beta, " << OpName << "_f_hidden_gate, &" << OpName << "_n);\n";
            }
         }

         if (!fNB.empty()) {
            if (direction == 0) {
               if (fType == "float") {
                  // Add the bias of the weight to f_update_gate
                  out << "\t" << "BLAS::saxpy_(&" << OpName << "_bias_size, &" << OpName << "_alpha, tensor_"
                      << fNB << ", &" << OpName << "_incx, " << OpName << "_f_update_gate, &" << OpName << "_incy);\n";
                  // Add the bias of the recurrence to f_update_gate
                  size_t rbz_offset = 3 * batch_size * fAttrHiddenSize;
                  out << "\t" << "BLAS::saxpy_(&" << OpName << "_bias_size, &" << OpName << "_alpha, tensor_"
                      << fNB << " + " << rbz_offset << ", &" << OpName << "_incx, " << OpName << "_f_update_gate, &"
                      << OpName << "_incy);\n";
                  // Add the bias of the weight to f_reset_gate
                  size_t wbr_offset = batch_size * fAttrHiddenSize;
                  out << "\t" << "BLAS::saxpy_(&" << OpName << "_bias_size, &" << OpName << "_alpha, tensor_"
                      << fNB << " + " << wbr_offset << ", &" << OpName << "_incx, " << OpName << "_f_reset_gate, &"
                      << OpName << "_incy);\n";
                  // Add the bias of the recurrence to f_reset_gate
                  size_t rbr_offset = fAttrHiddenSize * fAttrHiddenSize + 3 * batch_size * fAttrHiddenSize;
                  out << "\t" << "BLAS::saxpy_(&" << OpName << "_bias_size, &" << OpName << "_alpha, tensor_"
                      << fNB << " + " << rbr_offset << ", &" << OpName << "_incx, " << OpName << "_f_reset_gate, &"
                      << OpName << "_incy);\n";
                  // Add the bias of the weight to f_hidden_gate
                  size_t wbh_offset = 2 * batch_size * fAttrHiddenSize;
                  out << "\t" << "BLAS::saxpy_(&" << OpName << "_bias_size, &" << OpName << "_alpha, tensor_"
                      << fNB << " + " << wbh_offset << ", &" << OpName << "_incx, " << OpName << "_f_hidden_gate, &"
                      << OpName << "_incy);\n";
                  if (fAttrLinearBeforeReset == 0) {
                     // Add the bias of the recurrence to f_hidden_gate
                     size_t rbh_offset = 5 * batch_size * fAttrHiddenSize;
                     out << "\t" << "BLAS::saxpy_(&" << OpName << "_bias_size, &" << OpName << "_alpha, tensor_"
                         << fNB << " + " << rbh_offset << ", &" << OpName << "_incx, " << OpName
                         << "_f_hidden_gate, &" << OpName << "_incy);\n";
                  }
               }
            } else {
               if (fType == "float") {
                  // Add the bias of the weight to f_update_gate
                  size_t wbz_offset = 6 * batch_size * fAttrHiddenSize;
                  out << "\t" << "BLAS::saxpy_(&" << OpName << "_bias_size, &" << OpName << "_alpha, tensor_"
                      << fNB << " + " << wbz_offset << ", &" << OpName << "_incx, " << OpName << "_f_update_gate, &"
                      << OpName << "_incy);\n";
                  // Add the bias of the recurrence to f_update_gate
                  size_t rbz_offset = 3 * fAttrHiddenSize * fAttrHiddenSize + 3 * batch_size * fAttrHiddenSize;
                  out << "\t" << "BLAS::saxpy_(&" << OpName << "_bias_size, &" << OpName << "_alpha, tensor_"
                      << fNB << " + " << rbz_offset << ", &" << OpName << "_incx, " << OpName << "_f_update_gate, &"
                      << OpName << "_incy);\n";
                  // Add the bias of the weight to f_reset_gate
                  size_t wbr_offset = 6 * batch_size * fAttrHiddenSize + batch_size * fAttrHiddenSize;
                  out << "\t" << "BLAS::saxpy_(&" << OpName << "_bias_size, &" << OpName << "_alpha, tensor_"
                      << fNB << " + " << wbr_offset << ", &" << OpName << "_incx, " << OpName << "_f_reset_gate, &"
                      << OpName << "_incy);\n";
                  // Add the bias of the recurrence to f_reset_gate
                  size_t rbr_offset = 3 * fAttrHiddenSize * fAttrHiddenSize + fAttrHiddenSize * fAttrHiddenSize
                     + 3 * batch_size * fAttrHiddenSize;
                  out << "\t" << "BLAS::saxpy_(&" << OpName << "_bias_size, &" << OpName << "_alpha, tensor_"
                      << fNB << " + " << rbr_offset << ", &" << OpName << "_incx, " << OpName << "_f_reset_gate, &"
                      << OpName << "_incy);\n";
                  // Add the bias of the weight to f_hidden_gate
                  size_t wbh_offset = 6 * batch_size * fAttrHiddenSize + 2 * batch_size * fAttrHiddenSize;
                  out << "\t" << "BLAS::saxpy_(&" << OpName << "_bias_size, &" << OpName << "_alpha, tensor_"
                      << fNB << " + " << wbh_offset << ", &" << OpName << "_incx, " << OpName << "_f_hidden_gate, &"
                      << OpName << "_incy);\n";
                  if (fAttrLinearBeforeReset == 0) {
                     // Add the bias of the recurrence to f_hidden_gate
                     size_t rbh_offset = 6 * batch_size * fAttrHiddenSize + 5 * batch_size * fAttrHiddenSize;
                     out << "\t" << "BLAS::saxpy_(&" << OpName << "_bias_size, &" << OpName << "_alpha, tensor_"
                         << fNB << " + " << rbh_offset << ", &" << OpName << "_incx, " << OpName
                         << "_f_hidden_gate, &" << OpName << "_incy);\n";
                  }
               }
            }
         }

         // Copy the feedforward into the gates
         out << "\t" << "for (size_t seq = 0; seq < " << seq_length << "; seq++) {\n";
         out << "\t" << "\t" << "size_t offset = seq * " << batch_size * fAttrHiddenSize << ";\n";
         if (direction == 0) {
            out << "\t" << "\t" << "size_t gate_offset = seq * " << num_directions * batch_size * fAttrHiddenSize
               << ";\n";
         } else {
            out << "\t" << "\t" << "size_t gate_offset = seq * " << num_directions * batch_size * fAttrHiddenSize
                << " + " << batch_size * fAttrHiddenSize << ";\n";
         }
         size_t f_seq_size = batch_size * fAttrHiddenSize;
         out << "\t" << "\t" << "std::copy(" << OpName << "_f_update_gate + offset, " << OpName
             << "_f_update_gate + offset + " << f_seq_size << ", " << OpName << "_update_gate + gate_offset);\n";
         out << "\t" << "\t" << "std::copy(" << OpName << "_f_reset_gate + offset, " << OpName
             << "_f_reset_gate + offset + " << f_seq_size << ", " << OpName << "_reset_gate + gate_offset);\n";
         out << "\t" << "\t" << "std::copy(" << OpName << "_f_hidden_gate + offset, " << OpName
             << "_f_hidden_gate + offset + " << f_seq_size << ", " << OpName << "_hidden_gate + gate_offset);\n";
         out << "\t" << "}\n";

         out << "\t" << "for (size_t seq = 0; seq < " << seq_length << "; seq++) {\n";
         if (fAttrDirection == "backward" || direction == 1) {
            out << "\t" << "\t" << "size_t index = " << seq_length - 1 << " - seq;\n";
         } else {
            out << "\t" << "\t" << "size_t index = seq;\n";
         }
         out << "\t" << "\t" << "int m2 = " << batch_size << ";\n";
         if (direction == 0) {
            out << "\t" << "\t" << "size_t offset = index * " << num_directions * batch_size * fAttrHiddenSize
                 << ";\n";
         } else {
            out << "\t" << "\t" << "size_t offset = index * " << num_directions * batch_size * fAttrHiddenSize
                << " + " << batch_size * fAttrHiddenSize << ";\n";
         }
         size_t size = batch_size * fAttrHiddenSize;
         // gate = gate + initial_hidden_state * Recurrence^T
         out << "\t" << "\t" << "if (seq == 0) {\n";
         if (!fNInitial_h.empty()) {
            if (direction == 0) {
               if (fType == "float") {
                  out << "\t" << "\t" << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                      << OpName << "_n, &m2, &" << OpName << "_n, &" << OpName << "_alpha, tensor_" << fNR << ", &"
                      << OpName << "_n, " << OpName << "_initial_hidden_state, &" << OpName << "_n, &" << OpName
                      << "_alpha, " << OpName << "_update_gate + offset, &" << OpName << "_n);\n";
                  size_t rr_offset = fAttrHiddenSize * fAttrHiddenSize;
                  out << "\t" << "\t" << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                      << OpName << "_n, &m2, &" << OpName << "_n, &" << OpName << "_alpha, tensor_" << fNR << " + "
                      << rr_offset << ", &" << OpName << "_n, " << OpName << "_initial_hidden_state, &" << OpName
                      << "_n, &" << OpName << "_alpha, " << OpName << "_reset_gate + offset, &" << OpName << "_n);\n";
               }
            } else { // direction=1
               if (fType == "float") {
                  size_t rz_offset = 3 * fAttrHiddenSize * fAttrHiddenSize;
                  out << "\t" << "\t" << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                      << OpName << "_n, &m2, &" << OpName << "_n, &" << OpName << "_alpha, tensor_" << fNR << " + "
                      << rz_offset << ", &" << OpName << "_n, " << OpName << "_initial_hidden_state, &" << OpName
                      << "_n, &" << OpName << "_alpha, " << OpName << "_update_gate + offset, &" << OpName << "_n);\n";
                  size_t rr_offset = 4 * fAttrHiddenSize * fAttrHiddenSize;
                  out << "\t" << "\t" << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                      << OpName << "_n, &m2, &" << OpName << "_n, &" << OpName << "_alpha, tensor_" << fNR << " + "
                      << rr_offset << ", &" << OpName << "_n, " << OpName << "_initial_hidden_state, &" << OpName
                      << "_n, &" << OpName << "_alpha, " << OpName << "_reset_gate + offset, &" << OpName << "_n);\n";
               }
            }
         }
         out << "\t" << "\t" << "} else {\n";
         // gate = gate + previous_hidden_state * Recurrence^T
         if (direction == 0) {
            if (fAttrDirection == "backward") {
               out << "\t" << "\t" << "\t" << "size_t previous_offset = (index + 1) * "
                   << num_directions * batch_size * fAttrHiddenSize << ";\n";
            } else {
               out << "\t" << "\t" << "\t" << "size_t previous_offset = (seq - 1) * "
                   << num_directions * batch_size * fAttrHiddenSize << ";\n";
            }
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                << OpName << "_n, &m2, &" << OpName << "_n, &" << OpName << "_alpha, tensor_" << fNR << ", &"
                << OpName << "_n, " << OpName << "_hidden_state + previous_offset, &" << OpName << "_n, &"
                << OpName << "_alpha, " << OpName << "_update_gate + offset, &" << OpName << "_n);\n";
               size_t rr_offset = fAttrHiddenSize * fAttrHiddenSize;
               out << "\t" << "\t" << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                << OpName << "_n, &m2, &" << OpName << "_n, &" << OpName << "_alpha, tensor_" << fNR << " + "
                << rr_offset << ", &" << OpName << "_n, " << OpName << "_hidden_state + previous_offset, &"
                << OpName << "_n, &" << OpName << "_alpha, " << OpName << "_reset_gate + offset, &"
                << OpName << "_n);\n";
            }
         } else {
            out << "\t" << "\t" << "\t" << "size_t previous_offset = (index + 1) * "
                << num_directions * batch_size * fAttrHiddenSize << " + " << batch_size * fAttrHiddenSize << ";\n";
            if (fType == "float") {
               size_t rz_offset = 3 * fAttrHiddenSize * fAttrHiddenSize;
               out << "\t" << "\t" << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                << OpName << "_n, &m2, &" << OpName << "_n, &" << OpName << "_alpha, tensor_" << fNR << " + "
                << rz_offset << ", &" << OpName << "_n, " << OpName << "_hidden_state + previous_offset, &"
                << OpName << "_n, &" << OpName << "_alpha, " << OpName << "_update_gate + offset, &"
                << OpName << "_n);\n";
               size_t rr_offset = 4 * fAttrHiddenSize * fAttrHiddenSize;
               out << "\t" << "\t" << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                << OpName << "_n, &m2, &" << OpName << "_n, &" << OpName << "_alpha, tensor_" << fNR << " + "
                << rr_offset << ", &" << OpName << "_n, " << OpName << "_hidden_state + previous_offset, &"
                << OpName << "_n, &" << OpName << "_alpha, " << OpName << "_reset_gate + offset, &"
                << OpName << "_n);\n";
            }
         }
         out << "\t" << "\t" << "}\n";

         // Clip the elements of the update gate and the reset gate into the range [-fClip, fClip]
         if (fAttrClip > .0) {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float z = (" << OpName << "_update_gate[i] > " << -fAttrClip
                   << ") ? " << OpName << "_update_gate[i] : " << -fAttrClip << ";\n";
            }
            out << "\t" << "\t" << "\t" << OpName << "_update_gate[i] = (z < " << fAttrClip
                << ") ? z : " << fAttrClip << ";\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float r = (" << OpName << "_reset_gate[i] > " << -fAttrClip
                   << ") ? " << OpName << "_reset_gate[i] : " << -fAttrClip << ";\n";
            }
            out << "\t" << "\t" << "\t" << OpName << "_reset_gate[i] = (r < " << fAttrClip
                << ") ? r : " << fAttrClip << ";\n";
            out << "\t" << "\t" << "}\n";
         }

         // Apply the activation function to the update gate and the reset gate
         if (fAttrActivations[direction * 2] == "Relu") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_update_gate[i] < 0.)\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_update_gate[i] = 0.;\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_reset_gate[i] < 0.)\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_reset_gate[i] = 0.;\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2] == "Tanh") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float z = exp(-2 * " << OpName << "_update_gate[i]);\n";
            }
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_update_gate[i] = (1. - z) / (1. + z);\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float r = exp(-2 * " << OpName << "_reset_gate[i]);\n";
            }
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_reset_gate[i] = (1. - r) / (1. + r);\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2] == "Sigmoid") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_update_gate[i] = 1. / (1. + exp(-"
                << OpName << "_update_gate[i]));\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_reset_gate[i] = 1. / (1. + exp(-"
                << OpName << "_reset_gate[i]));\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2] == "Affine") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_update_gate[i] = "
                << fAttrActivationAlpha[direction * 2] << " * " << OpName << "_update_gate[i] + "
                << fAttrActivationBeta[direction * 2] << ";\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_reset_gate[i] = "
                << fAttrActivationAlpha[direction * 2] << " * " << OpName << "_reset_gate[i] + "
                << fAttrActivationBeta[direction * 2] << ";\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2] == "ScaledTanh") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float z = exp(-2 * " << fAttrActivationBeta[direction * 2]
                   << " * "<< OpName << "_update_gate[i]);\n";
               }
               out << "\t" << "\t" << "\t" << "\t" << OpName << "_update_gate[i] = "
                   << fAttrActivationAlpha[direction * 2] << " * (1. - z) / (1. + z);\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float r = exp(-2 * " << fAttrActivationBeta[direction * 2]
                   << " * "<< OpName << "_reset_gate[i]);\n";
               }
               out << "\t" << "\t" << "\t" << "\t" << OpName << "_reset_gate[i] = "
                   << fAttrActivationAlpha[direction * 2] << " * (1. - r) / (1. + r);\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2] == "HardSigmoid") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float za = " << fAttrActivationAlpha[direction * 2] << " * "
                   << OpName << "_update_gate[i] + " << fAttrActivationBeta[direction * 2] << ";\n";
               out << "\t" << "\t" << "\t" << "float zb = (za > 0.) ? za : 0.;\n";
            }
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_update_gate[i] = (zb < 1.) ? zb : 1.;\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float ra = " << fAttrActivationAlpha[direction * 2] << " * "
                   << OpName << "_reset_gate[i] + " << fAttrActivationBeta[direction * 2] << ";\n";
               out << "\t" << "\t" << "\t" << "float rb = (ra > 0.) ? ra : 0.;\n";
            }
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_reset_gate[i] = (rb < 1.) ? rb : 1.;\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2] == "LeakyRelu") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_update_gate[i] < 0.)\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_update_gate[i] = "
                << fAttrActivationAlpha[direction * 2] << " * " << OpName << "_update_gate[i];\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_reset_gate[i] < 0.)\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_reset_gate[i] = "
                << fAttrActivationAlpha[direction * 2] << " * " << OpName << "_reset_gate[i];\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2] == "ThresholdRelu") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_update_gate[i] < "
                << fAttrActivationAlpha[direction * 2] << ")\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_update_gate[i] = 0.;\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_reset_gate[i] < "
                << fAttrActivationAlpha[direction * 2] << ")\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_reset_gate[i] = 0.;\n";
            out << "\t" << "\t" << "}";
         } else if (fAttrActivations[direction * 2] == "Elu") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_update_gate[i] < 0.)\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_update_gate[i] = "
                << fAttrActivationAlpha[direction * 2] << " * exp(" << OpName << "_update_gate[i] - 1.);\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_reset_gate[i] < 0.)\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_reset_gate[i] = "
                << fAttrActivationAlpha[direction * 2] << " * exp(" << OpName << "_reset_gate[i] - 1.);\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2] == "Softsign") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_update_gate[i] = " << OpName
                << "_update_gate[i] / (1. + abs(" << OpName << "_update_gate[i]));\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_reset_gate[i] = " << OpName
                << "_reset_gate[i] / (1. + abs(" << OpName << "_reset_gate[i]));\n";
            out << "\t" << "\t" << "}\n";
         } else { // fAttrActivations[direction * 2] = Softplus
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_update_gate[i] = log(1. + exp("
                << OpName << "_update_gate[i]));\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_reset_gate[i] = log(1. + exp("
                << OpName << "_reset_gate[i]));\n";
            out << "\t" << "\t" << "}\n";
         }

         if (fAttrLinearBeforeReset == 0) {
            out << "\t" << "\t" << "if (seq == 0) {\n";
            if (!fNInitial_h.empty()) {
               // feedback = reset_gate o initial_hidden_state
               out << "\t" << "\t" << "\t" << "for (size_t i = 0; i < " << size << "; i++) {\n";
               out << "\t" << "\t" << "\t" << "\t" << OpName << "_feedback[i] = " << OpName
                   << "_reset_gate[i + offset] * " << OpName << "_initial_hidden_state[i];\n";
               out << "\t" << "\t" << "\t" << "}\n";
            }
            out << "\t" << "\t" << "} else {\n";
            // feedback = reset_gate o previous_hidden_state
            if (direction == 0) {
               if (fAttrDirection == "backward") {
                  out << "\t" << "\t" << "\t" << "size_t previous_offset = (index + 1) * "
                      << num_directions * batch_size * fAttrHiddenSize << ";\n";
               } else {
                  out << "\t" << "\t" << "\t" << "size_t previous_offset = (seq - 1) * "
                      << num_directions * batch_size * fAttrHiddenSize << ";\n";
               }
            } else {
               out << "\t" << "\t" << "\t" << "size_t previous_offset = (index + 1) * " << num_directions
                   * batch_size * fAttrHiddenSize << " + " << batch_size * fAttrHiddenSize << ";\n";
            }
            out << "\t" << "\t" << "\t" << "for (size_t i = 0; i < " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_feedback[i] = " << OpName
                << "_reset_gate[i + offset] * " << OpName << "_hidden_state[i + previous_offset];\n";
            out << "\t" << "\t" << "\t" << "}\n";
            out << "\t" << "\t" << "}\n";
            // feedback = feedback * R_h^T
            size_t rh_offset = (direction == 0) ?
               2 * fAttrHiddenSize * fAttrHiddenSize : 3 * fAttrHiddenSize * fAttrHiddenSize
               + 2 * fAttrHiddenSize * fAttrHiddenSize;
            out << "\t" << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                << OpName << "_n, &" << OpName << "_m2, &" << OpName << "_n, &" << OpName << "_alpha, tensor_"
                << fNR << " + " << rh_offset << ", &" << OpName << "_n, " << OpName << "_feedback, &" << OpName
                << "_n, &" << OpName << "_beta, " << OpName << "_feedback, &" << OpName << "_n);\n";
         } else { // fAttrLinearBeforeReset=1
            // feedback = previous_hidden_state * R_h^T
            if (direction == 0) {
               if (fAttrDirection == "backward") {
                  out << "\t" << "\t" << "size_t previous_offset = (index + 1) * "
                      << num_directions * batch_size * fAttrHiddenSize << ";\n";
               } else {
                  out << "\t" << "\t" << "size_t previous_offset = (seq - 1) * "
                      << num_directions * batch_size * fAttrHiddenSize << ";\n";
               }
            } else {
               out << "\t" << "\t" << "size_t previous_offset = (index + 1) * " << num_directions
                   * batch_size * fAttrHiddenSize << " + " << batch_size * fAttrHiddenSize << ";\n";
            }
            size_t rh_offset = (direction == 0) ?
               2 * fAttrHiddenSize * fAttrHiddenSize : 3 * fAttrHiddenSize * fAttrHiddenSize
               + 2 * fAttrHiddenSize * fAttrHiddenSize;
            out << "\t" << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                << OpName << "_n, &" << OpName << "_m2, &" << OpName << "_n, &" << OpName << "_alpha, tensor_" << fNR
                << " + " << rh_offset << ", &" << OpName << "_n, " << OpName << "_hidden_state + previous_offset, "
                << OpName << "_n, &" << OpName << "_beta, " << OpName << "_feedback, &" << OpName << "_n);\n";
            // Add the bias of the recurrence to feedback
            size_t rbh_offset = (direction == 0) ?
               5 * batch_size * fAttrHiddenSize : 6 * batch_size * fAttrHiddenSize
               + 5 * batch_size * fAttrHiddenSize;
            out << "\t" << "\t" << "BLAS::saxpy_(&" << OpName << "_feedback_size, &" << OpName
                << "_alpha, tensor_" << fNB << " + " << rbh_offset << ", &" << OpName << "_incx, "
                << OpName << "_feedback, &" << OpName << "_incy);\n";
            // feedback = reset_gate o feedback
            out << "\t" << "\t" << "for (size_t i = 0; i < " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << OpName << "_feedback[i] *= " << OpName << "_reset_gate[i + offset]\n";
            out << "\t" << "\t" << "}\n";
         }

         // hidden_gate = hidden_gate + feedback
         out << "\t" << "\t" << "BLAS::saxpy_(&" << OpName << "_feedback_size, &" << OpName << "_alpha, "
             << OpName << "_feedback, &" << OpName << "_incx, " << OpName << "_hidden_gate + offset, &"
             << OpName << "_incy);\n";

         // Clip the elements of the hidden gate into the range [-fClip, fClip]
         if (fAttrClip > .0) {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float x = (" << OpName << "_hidden_gate[i] > " << -fAttrClip
                   << ") ? " << OpName << "_hidden_gate[i] : " << -fAttrClip << ";\n";
            }
            out << "\t" << "\t" << "\t" << OpName << "_hidden_gate[i] = (x < " << fAttrClip << ") ? x : "
                << fAttrClip << ";\n";
            out << "\t" << "\t" << "}\n";
         }

         // Apply the activation function to the hidden gate
         if (fAttrActivations[direction * 2 + 1] == "Relu") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_hidden_gate[i] < 0.)\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_gate[i] = 0.;\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2 + 1] == "Tanh") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float ex = exp(-2 * " << OpName << "_hidden_gate[i]);\n";
            }
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_gate[i] = (1. - ex) / (1. + ex);\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2 + 1] == "Sigmoid") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_gate[i] = 1. / (1. + exp(-" << OpName
                << "_hidden_gate[i]));\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2 + 1] == "Affine") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_gate[i] = "
                << fAttrActivationAlpha[direction * 2 + 1] << " * " << OpName << "_hidden_gate[i] + "
                << fAttrActivationBeta[direction * 2 + 1] << ";\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2 + 1] == "ScaledTanh") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float ex = exp(-2 * " << fAttrActivationBeta[direction * 2 + 1]
                   << " * "<< OpName << "_hidden_gate[i]);\n";
               }
               out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_gate[i] = "
                   << fAttrActivationAlpha[direction * 2 + 1] << " * (1. - ex) / (1. + ex);\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2 + 1] == "HardSigmoid") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float a = " << fAttrActivationAlpha[direction * 2 + 1] << " * "
                   << OpName << "_hidden_gate[i] + " << fAttrActivationBeta[direction * 2 + 1] << ";\n";
               out << "\t" << "\t" << "\t" << "float b = (a > 0.) ? a : 0.;\n";
            }
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_gate[i] = (b < 1.) ? b : 1.;\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2 + 1] == "LeakyRelu") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_hidden_gate[i] < 0.)\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_gate[i] = "
                << fAttrActivationAlpha[direction * 2 + 1] << " * " << OpName << "_hidden_gate[i];\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2 + 1] == "ThresholdRelu") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_hidden_gate[i] < "
                << fAttrActivationAlpha[direction * 2 + 1] << ")\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_gate[i] = 0.;\n";
            out << "\t" << "\t" << "}";
         } else if (fAttrActivations[direction * 2 + 1] == "Elu") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_hidden_gate[i] < 0.)\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_gate[i] = "
                << fAttrActivationAlpha[direction * 2 + 1] << " * exp(" << OpName << "_hidden_gate[i] - 1.);\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction * 2 + 1] == "Softsign") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_gate[i] = " << OpName
                << "_hidden_gate[i] / (1. + abs(" << OpName << "_hidden_gate[i]));\n";
            out << "\t" << "\t" << "}\n";
         } else { // fAttrActivations[direction * 2 + 1] = Softplus
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_gate[i] = log(1. + exp("
                << OpName << "_hidden_gate[i]));\n";
            out << "\t" << "\t" << "}\n";
         }

         // hidden_state = (1 - update_gate) o hidden_gate
         out << "\t" << "\t" << "for (size_t i = offset; i < offset + " << size << "; i++) {\n";
         out << "\t" << "\t" << "\t" << OpName << "_hidden_state[i] = ( 1. - " << OpName
             << "_update_gate[i]) * " << OpName << "_hidden_gate[i];\n";
         out << "\t" << "\t" << "}\n";

         out << "\t" << "\t" << "if (seq == 0) {\n";
         if (!fNInitial_h.empty()) {
            // hidden_state += update_gate o initial_hidden_state
            out << "\t" << "\t" << "\t" << "for (size_t i = 0; i < " << size << "; i++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[i + offset] += " << OpName
                << "_update_gate[i + offset] * " << OpName << "_initial_hidden_state[i];\n";
            out << "\t" << "\t" << "\t" << "}\n";
         }
         out << "\t" << "\t" << "} else {\n";
         // hidden_state += update_gate o previous_hidden_state
         if (direction == 0) {
            if (fAttrDirection == "backward") {
               out << "\t" << "\t" << "\t" << "size_t previous_offset = (index + 1) * "
                   << num_directions * batch_size * fAttrHiddenSize << ";\n";
            } else {
               out << "\t" << "\t" << "\t" << "size_t previous_offset = (seq - 1) * "
                   << num_directions * batch_size * fAttrHiddenSize << ";\n";
            }
         } else {
            out << "\t" << "\t" << "\t" << "size_t previous_offset = (index + 1) * "
                << num_directions * batch_size * fAttrHiddenSize << " + " << batch_size * fAttrHiddenSize << ";\n";
         }
         out << "\t" << "\t" << "\t" << "for (size_t i = 0; i < " << size << "; i++) {\n";
         out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[i + offset] += " << OpName
             << "_update_gate[i + offset] * " << OpName << "_hidden_state[i + previous_offset];\n";
         out << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "}\n";

         out << "\t" << "}\n";
      }

      // Padding the hidden state for GRU with different sequence lengths
      if (!fNSequence_lens.empty()) {
         out << "\t" << "for (size_t seq = 0; seq < " << seq_length << "; seq++) {\n";
         out << "\t" << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
         out << "\t" << "\t" << "\t" << "if (seq >= tensor_" << fNSequence_lens << "[batch]) {\n";
         for (size_t direction = 0; direction < num_directions; direction++) {
            out << "\t" << "\t" << "\t" << "\t" << "\t" << "for (size_t h = 0; h < " << fAttrHiddenSize << "; h++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[seq * "
                << num_directions * batch_size * fAttrHiddenSize + direction * batch_size * fAttrHiddenSize
                << " + batch * " << fAttrHiddenSize << " + h] = 0.;\n";
            out << "\t" << "\t" << "\t" << "\t" << "\t" << "}\n";
         }
         out << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "}\n";
         out << "\t" << "}\n";
      }

      // Copy the hidden state into y and y_h
      if (fAttrLayout == 0) {
         if (!fNY_h.empty()) {
            // Copy hidden_state into Y_h
            if (fNSequence_lens.empty()) {
               size_t yh_size = batch_size * fAttrHiddenSize;
               if (fAttrDirection == "backward") {
                  out << "\t" << "std::copy(" << OpName << "_hidden_state, " << OpName << "_hidden_state + "
                      << yh_size << ", tensor_" << fNY_h << ");\n";
               } else {
                  size_t offset = (seq_length - 1) * num_directions * batch_size * fAttrHiddenSize;
                  out << "\t" << "std::copy(" << OpName << "_hidden_state + " << offset << ", " << OpName
                      << "_hidden_state + " << offset << " + " << yh_size << ", tensor_" << fNY_h << ");\n";
               }
               if (num_directions == 2) {
                  out << "\t" << "std::copy(" << OpName << "_hidden_state + " << yh_size << ", " << OpName
                      << "_hidden_state + " << 2 * yh_size << ", tensor_" << fNY_h << " + " << yh_size << ");\n";
               }
            } else { // GRU with different sequence lengths
               if (fAttrDirection == "backward") {
                  out << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
                  out << "\t" << "\t" << "size_t offset = batch * " << fAttrHiddenSize << ";\n";
                  out << "\t" << "\t" << "std::copy(" << OpName << "_hidden_state + offset, " << OpName
                      << "_hidden_state + offset + " << fAttrHiddenSize << ", tensor_" << fNY_h << " + offset);\n";
                  out << "\t" << "}\n";
               } else {
                  out << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
                  out << "\t" << "\t" << "size_t seq = " << "tensor_" << fNSequence_lens << "[batch] - 1;\n";
                  out << "\t" << "\t" << "size_t offset = seq * " << num_directions * batch_size * fAttrHiddenSize
                      << " + batch * " << fAttrHiddenSize << ";\n";
                  out << "\t" << "\t" << "size_t yh_offset = batch * " << fAttrHiddenSize << ";\n";
                  out << "\t" << "\t" << "std::copy(" << OpName << "_hidden_state + offset, " << OpName
                      << "_hidden_state + offset + " << fAttrHiddenSize << ", tensor_" << fNY_h << " + yh_offset);\n";
                  out << "\t" << "}\n";
               }
               if (num_directions == 2) {
                  out << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
                  out << "\t" << "\t" << "size_t offset = " << batch_size * fAttrHiddenSize
                      << " + batch * " << fAttrHiddenSize << ";\n";
                  out << "\t" << "\t" << "size_t yh_offset = " << batch_size * fAttrHiddenSize
                      << " + batch * " << fAttrHiddenSize << ";\n";
                  out << "\t" << "\t" << "std::copy(" << OpName << "_hidden_state + offset, " << OpName
                      << "_hidden_state + offset + " << fAttrHiddenSize << ", tensor_" << fNY_h << " + yh_offset);\n";
                  out << "\t" << "}\n";
               }
            }
         }
      } else { // fAttrLayout=1
         if (!fNY.empty()) {
            // Copy hidden_state into Y
            for (size_t direction = 0; direction < num_directions; direction++) {
               out << "\t" << "for (size_t seq = 0; seq < " << seq_length << "; seq++) {\n";
               out << "\t" << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
               out << "\t" << "\t" << "\t" << "size_t offset = seq * " << num_directions * batch_size * fAttrHiddenSize
                   << " + " << direction * batch_size * fAttrHiddenSize << " + batch * " << fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "\t" << "size_t y_offset = batch * " << seq_length * num_directions * fAttrHiddenSize
                   << " + seq * " << num_directions * fAttrHiddenSize << " + " << direction * fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "\t" << "std::copy(" << OpName << "_hidden_state + offset, " << OpName
                   << "_hidden_state + offset + " << fAttrHiddenSize << ", tensor_" << fNY << " + y_offset);\n";
               out << "\t" << "\t" << "}\n";
               out << "\t" << "}\n";
            }
         }
         if (!fNY_h.empty()) {
            // Copy the hidden_state into Y_h
            if (fAttrDirection == "backward") {
               out << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
               out << "\t" << "\t" << "size_t offset = batch * " << fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "size_t yh_offset = batch * " << num_directions * fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "std::copy(" << OpName << "_hidden_state + offset, " << OpName
                   << "_hidden_state + offset + " << fAttrHiddenSize << ", tensor_" << fNY_h << " + yh_offset);\n";
               out << "\t" << "}\n";
            } else {
               out << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
               if (fNSequence_lens.empty()) {
                  out << "\t" << "\t" << "size_t seq = " << seq_length - 1 << ";\n";
               } else {
                  out << "\t" << "\t" << "size_t seq = " << "tensor_" << fNSequence_lens << "[batch] - 1;\n";
               }
               out << "\t" << "\t" << "size_t offset = seq * " << num_directions * batch_size * fAttrHiddenSize
                   << " + batch * " << fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "size_t yh_offset = batch * " << num_directions * fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "std::copy(" << OpName << "_hidden_state + offset, " << OpName
                   << "_hidden_state + offset + " << fAttrHiddenSize << ", tensor_" << fNY_h << " + yh_offset);\n";
               out << "\t" << "}\n";
            }
            if (num_directions == 2) {
               out << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
               out << "\t" << "\t" << "size_t offset = " << batch_size * fAttrHiddenSize << " + batch * "
                   << fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "size_t yh_offset = batch * " << num_directions * fAttrHiddenSize << " + "
                   << fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "std::copy(" << OpName << "_hidden_state + offset, " << OpName
                   << "_hidden_state + offset + " << fAttrHiddenSize << ", tensor_" << fNY_h << " + yh_offset);\n";
               out << "\t" << "}\n";
            }
         }
      }

      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
