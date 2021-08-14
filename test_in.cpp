#include "testCaseInstanceNorm_1.hxx"

#include <algorithm>
#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <time.h>

int main(){
   const int n =2*3*4*5;
   float inputss[n] = {
	   1.408230662345886230e-01,
-4.410425722599029541e-01,
-3.359223306179046631e-01,
-1.727990955114364624e-01,
-1.590707898139953613e+00,
1.861469030380249023e+00,
-6.418844219297170639e-03,
6.900002956390380859e-01,
-6.051822304725646973e-01,
-8.357438445091247559e-01,
1.282754540443420410e+00,
1.200208783149719238e+00,
-1.052471041679382324e+00,
3.685210049152374268e-01,
2.129749536514282227e+00,
1.001815080642700195e+00,
1.028962969779968262e+00,
-3.360367417335510254e-01,
-7.886439561843872070e-01,
-4.168069362640380859e-01,
7.912051677703857422e-01,
-6.796895861625671387e-01,
2.311003953218460083e-01,
-9.878946468234062195e-03,
4.607469439506530762e-01,
-7.995736002922058105e-01,
7.815433293581008911e-02,
-1.872568577527999878e-02,
-7.899420857429504395e-01,
1.033812284469604492e+00,
1.718611836433410645e+00,
5.294144749641418457e-01,
-1.257323026657104492e+00,
-2.539811134338378906e-01,
-1.121054530143737793e+00,
3.404809534549713135e-01,
-1.240813657641410828e-01,
9.024481773376464844e-01,
-1.276684552431106567e-01,
1.022624254226684570e+00,
2.731915712356567383e-01,
-9.022383093833923340e-01,
1.128219515085220337e-01,
-9.903874248266220093e-02,
3.339947462081909180e-01,
-7.006556391716003418e-01,
-6.747316718101501465e-01,
-5.020832419395446777e-01,
-4.641020298004150391e-01,
6.428984999656677246e-01,
-7.729179263114929199e-01,
-2.907449305057525635e-01,
-8.183640837669372559e-01,
-8.022791743278503418e-01,
-7.846617698669433594e-01,
-1.136696219444274902e+00,
3.871940672397613525e-01,
9.300022125244140625e-01,
3.081022739410400391e+00,
-1.170924782752990723e+00,
1.187241822481155396e-01,
1.071151733398437500e+00,
9.074761867523193359e-01,
1.185947895050048828e+00,
-6.206241250038146973e-01,
-2.451246380805969238e-01,
-1.671078920364379883e+00,
5.532880425453186035e-01,
-2.844502925872802734e+00,
7.071546316146850586e-01,
-4.856002628803253174e-01,
3.667388260364532471e-01,
1.640831828117370605e-01,
-1.334887027740478516e+00,
-1.212914824485778809e+00,
1.463807225227355957e+00,
-5.733821392059326172e-01,
-9.554402828216552734e-01,
2.922860383987426758e-01,
-1.045349836349487305e+00,
6.071310639381408691e-01,
-1.273798942565917969e-01,
5.507920980453491211e-01,
7.991607189178466797e-01,
-2.384920835494995117e+00,
1.493168592453002930e+00,
-1.502574801445007324e+00,
-1.522353440523147583e-01,
6.961275935173034668e-01,
1.793691664934158325e-01,
1.498691886663436890e-01,
7.074797991663217545e-03,
-3.135632276535034180e-01,
-3.340588808059692383e-01,
1.218666672706604004e+00,
2.557981610298156738e-01,
-1.003211021423339844e+00,
9.734609127044677734e-01,
-6.430810093879699707e-01,
-9.966270923614501953e-01,
2.424867630004882812e+00,
3.318164646625518799e-01,
1.127084612846374512e+00,
-6.351163387298583984e-01,
-1.214517474174499512e+00,
1.075468540191650391e+00,
2.557419538497924805e-01,
-4.977259635925292969e-01,
7.301319390535354614e-02,
-1.783542990684509277e+00,
7.384370267391204834e-02,
-6.110147759318351746e-02,
-1.106023073196411133e+00,
3.132668435573577881e-01,
1.057330727577209473e+00,
-3.548019230365753174e-01,
-1.962830185890197754e+00,
-7.950413823127746582e-01,
-2.373389452695846558e-01,
1.907506704330444336e+00
   };

   auto t1 = std::chrono::high_resolution_clock::now();
   auto out = TMVA_SOFIE_testCaseInstanceNorm_1::infer(inputss);
   auto t2 = std::chrono::high_resolution_clock::now();
   for (auto& i: out){
      std::cout << i << ",";
   }
   //free(inputss);
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << std::endl << duration << std::endl;
}