CXX = g++
CPPFLAGS = -std=c++14 -MMD -MP -g
PROTOBUF = `pkg-config --cflags protobuf`
PROTOBUFL = `pkg-config --libs protobuf`
ROOTCONFIG =
ROOTCONFIG2 = `root-config --cflags --glibs`
# BLASDIR = /Users/sitongan/rootdev/BLAS-3.8.0
BLASDIR = /blas-3.8.0/BLAS-3.8.0
BLASFLAGS = -L${BLASDIR} -lblas
SRC = ${wildcard *.cxx}
SOFIEOBEJCT =
SOFIEHEADER =
SOFIE = $(SOFIEOBEJCT) $(SOFIEHEADER)

prototype: ${SRC:%.cxx=%.o}
	${CXX} -o prototype $^ ${CPPFLAGS} $(ROOTCONFIG) $(PROTOBUFL)

testinfer: test.cpp
	${CXX} -o testinfer test.cpp -std=c++14 -g $(BLASFLAG) -O3 -I ./eigen/

validate: test_old.cpp
	${CXX} -o testinfer test_old.cpp -std=c++14 -g $(BLASFLAG) -O3 -I ./eigen/

-include $(SRC:%.cxx=%.d)

%.o: %.cxx
	${CXX} ${CPPFLAGS} -c $< `root-config --cflags` $(PROTOBUF)

.phony: clean
clean:
	-rm *.d
	-rm *.o
