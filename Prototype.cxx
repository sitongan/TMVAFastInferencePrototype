#include "RModel.hxx"
#include "RModelParser_ONNX.hxx"




#include <string>

void permute( float input[24], float output[24]){
	for (int id = 0; id < 24 ; id++){
		 output[id / 24 % 1 * 1 + id / 12 % 2 * 1 + id / 4 % 3 * 2 + id / 1 % 4 * 6] = input[id];
	}
}




int main(){

   constexpr int dim = 4;
   int shape[dim] ={1,2,3,4};
   constexpr int c_length = 24;
   int length=1;
   int sizeofindex[dim];
   for (int i = dim - 1; i>=0; i--){
      sizeofindex[i] = length;
      length *= shape[i];
   }


   float a[length];
   for (int i = 0; i < length; i++){
      a[i] = (float)(i +1);
   }

   int perm[dim] ={4, 3,2,1};
   int index_goto[dim];
   for (int i = 0; i < dim; i++){
      index_goto[perm[i] - 1] = i + 1;
   }
   int new_shape[dim];
   int t = 1;
   int new_sizeofindex[dim];
   for (int i = dim - 1; i>=0; i--){
      new_shape[i] = shape[perm[i] - 1];
      new_sizeofindex[i] = t;
      t *= new_shape[i];
   }


   using std::endl;

   std::stringstream out;
   out << "void permute( float input[" << length << "], float output[" << length << "]){" << "\n";
   out << "\t" << "for (int id = 0; id < " << length << " ; id++){\n";
   out << "\t\t output[";
   for (int i =0; i < dim; i++){
      out << "id / " << sizeofindex[i] << " % " << shape[i] << " * " << new_sizeofindex[index_goto[i] - 1];
      if (i != dim - 1) out << " + ";
   }
   out << "] = input[id];\n";
   out << "\t}\n";
   out << "}\n";
   std::cout << out.str();


   float input[c_length];
   float output[c_length];
   for (int i =0; i < c_length; i++){
      input[i] = (float)(i+1);
   }
   permute(input, output);
   for (int i =0; i < c_length; i++){
      std::cout << output[i] << "\t";
   }


   TMVA::Experimental::SOFIE::RModelParser_ONNX parser;
   TMVA::Experimental::SOFIE::RModel model = parser.Parse("./LinearNN.onnx");
   //TMVA::Experimental::SOFIE::RModel model2 = std::move(model);
   //model2.printRequiredInputTensors();
   //model2.printInitializedTensors();
   //model2.headInitializedTensors("18.bias");
   //model2.headInitializedTensors("0.weight");
}
