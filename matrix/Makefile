NVCC=nvcc
BIN=main
SRC=main.cu
CUDACXXFLAGS= -O2
   
$(BIN):
	$(NVCC) $(SRC) $(CUDACXXFLAGS) -o $(BIN)

all: $(BIN)

test:
	nvprof ./$(BIN)
clean:
	$(RM) $(BIN)
