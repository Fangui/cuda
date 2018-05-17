NVCC=nvcc
BIN=main
SRC=main.cu
CUDAFLAGS= 
   
$(BIN):
	$(NVCC) $(SRC) $(CUDAFLAGS) -o $(BIN)

all: $(BIN)

test:
	nvprof ./$(BIN)
clean:
	$(RM) $(BIN)
