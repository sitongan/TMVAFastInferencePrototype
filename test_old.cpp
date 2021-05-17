#include "Linear_event.hxx"

#include <algorithm>
#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <time.h>

int main(){
   const int batchsize =1;
   float inputss[100 * batchsize];
   std::fill_n(inputss, 100 * batchsize, 0.0);
/*
   for (int i = 0; i < 100 * batchsize; i++){
      srand(time(0));
      inputss[i] = rand();
   }
*/
   auto t1 = std::chrono::high_resolution_clock::now();
   auto out = TMVA_SOFIE_Linear_event::infer(inputss);
   auto t2 = std::chrono::high_resolution_clock::now();
   for (auto& i: out){
      std::cout << i << ",";
   }
   //free(inputss);
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << std::endl << duration << std::endl;
}
