#include "Linear_event.hxx"

#include <algorithm>
#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <numeric>


int main(){

#define FLASH_CACHE
   const int n = 1000;
   const int batchsize =1;
   float inputss[n][100 * batchsize];
   //std::fill_n(inputss, 100 * batchsize, 0.0);

for (int k = 0; k < n; ++k) {
   for (int i = 0; i < 100 * batchsize; i++){
      srand(time(0));
      inputss[k][i] = rand();
   }
}

std::vector<float> total_time;
const size_t bigger_than_cachesize = 10 * 1024 * 1024;

for (int k = 0; k < n; ++k ) {

#ifdef FLASH_CACHE
      std::vector<float> tmp(bigger_than_cachesize);
      // When you want to "flush" cache.
      for(int i = 0; i < bigger_than_cachesize; i++)
      {
         tmp[i] = rand();
      }
#endif

   auto t1 = std::chrono::high_resolution_clock::now();
   auto out = TMVA_SOFIE_Linear_event::infer(inputss[k]);
   auto t2 = std::chrono::high_resolution_clock::now();
   total_time.push_back(float(std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count()));
}

   //for (auto& i: out){
      //std::cout << i << ",";
   //}

   float sum = std::accumulate(total_time.begin(), total_time.end(), 0.0);
   float mean = sum / total_time.size();

   float sq_sum = std::inner_product(total_time.begin(), total_time.end(), total_time.begin(), 0.0);
   float std = std::sqrt(sq_sum / total_time.size() - mean * mean);

   std::cout << std::endl << mean  << std::endl;
   std::cout << std  << std::endl;
}
