CXX = g++
CPPFLAGS = -std=c++14 -MMD -MP -g
PROTOBUF = `pkg-config --cflags protobuf`
PROTOBUFL = `pkg-config --libs protobuf`
ROOTCONFIG =
ROOTCONFIG2 = `root-config --cflags --glibs`
BLASDIR = /Users/sitongan/rootdev/BLAS-3.8.0
BLASFLAG = -L${BLASDIR} -lblas
SRC = ${wildcard *.cxx}
SOFIEOBEJCT =
SOFIEHEADER =
SOFIE = $(SOFIEOBEJCT) $(SOFIEHEADER)

prototype: ${SRC:%.cxx=%.o}
	${CXX} -o prototype $^ ${CPPFLAGS} $(ROOTCONFIG) $(PROTOBUFL)

testinfer: test.cpp LinearNN.hxx
	${CXX} -o testinfer test.cpp -std=c++14 -g $(BLASFLAG) -O3

-include $(SRC:%.cxx=%.d)

%.o: %.cxx
	${CXX} ${CPPFLAGS} -c $< `root-config --cflags` $(PROTOBUF)

.phony: clean
clean:
	-rm *.d
	-rm *.o
