# Compiler
NVCC = nvcc

# Flags
NVCC_FLAGS = -O3 -std=c++11 --gpu-architecture=sm_75
LIBS = -lcusolver -lcublas

# Target
TARGET = matrix_inversion

# Build
all: $(TARGET)

$(TARGET): matrix_inversion.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

# Run
run: $(TARGET)
	./$(TARGET)

# Clean
clean:
	rm -f $(TARGET)

.PHONY: all run clean