CXX = g++
CPPFLAGS = -std=c++14 -MMD -MP -g
PROTOBUF = `pkg-config --cflags protobuf`
PROTOBUFL = `pkg-config --libs protobuf`
ROOTCONFIG =
ROOTCONFIG2 = `root-config --cflags --glibs`
BLASDIR = /usr/local/lib/BLAS-3.8.0
BLASFLAG = -L${BLASDIR} -lblas
SRC = ${wildcard *.cxx}
SOFIEOBEJCT =
SOFIEHEADER =
SOFIE = $(SOFIEOBEJCT) $(SOFIEHEADER)

prototype: ${SRC:%.cxx=%.o}
	${CXX} -o prototype $^ ${CPPFLAGS} $(ROOTCONFIG) $(PROTOBUFL)

test_rnn: test_rnn.cpp
	${CXX} -o test_rnn test_rnn.cpp -std=c++14 -g $(BLASFLAG) -O3

validate: test_old.cpp
	${CXX} -o testinfer test_old.cpp -std=c++14 -g $(BLASFLAG) -O3 -I ./eigen/

-include $(SRC:%.cxx=%.d)

%.o: %.cxx
	${CXX} ${CPPFLAGS} -c $< `root-config --cflags` $(PROTOBUF)

.phony: clean
clean:
	-rm *.d
	-rm *.o
