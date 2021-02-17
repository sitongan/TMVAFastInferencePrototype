#include "LinearNN.hxx"

#include <algorithm>
#include <iostream>

int main(){
   float inputss[6400];
   std::fill_n(inputss, 6400, 0.0);
   auto out = TMVA_SOFIE_LinearNN::infer(inputss);
   for (auto& i: out){
      std::cout << i << ",";
   }
   //free(inputss);

}
