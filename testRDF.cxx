#include "Linear_RDF.hxx"
#include "ROOT/RDataFrame.hxx"

int main(){

   size_t npoints = 1000;
   ROOT::RDataFrame df(npoints);
   auto df2 = df.Define("a","1.0f").Define("b","1.0f").Define("c", "1.0f")
                .Define("p", TMVA_SOFIE_Linear_RDF::infer, {"a", "b", "c"});
   auto d1 = df2.Display("");
   d1 -> Print();

   return 0;
}
