NVCC ?= nvcc
CUDA_ARCH ?= sm_89
CUDA_FLAGS ?= -O3 -std=c++17 -lineinfo

.PHONY: cuda-transpose-test cuda-transpose-bench offline-audit clean

build/matrix_transpose: cuda/memory/matrix_transpose.cu cuda/memory/transpose_kernels.cuh
	mkdir -p build
	$(NVCC) $(CUDA_FLAGS) -arch=$(CUDA_ARCH) cuda/memory/matrix_transpose.cu -o $@

cuda-transpose-test: build/matrix_transpose
	./build/matrix_transpose --mode test --sizes 1,7,31,32,33,47,63,65,129

cuda-transpose-bench: build/matrix_transpose
	./build/matrix_transpose --mode bench --n 8192 --warmup 5 --repeat 50 --samples 5 --json results/summary/transpose_8192.json

offline-audit:
	python scripts/offline_audit.py
	python -m unittest discover -s tests -p 'test_transpose_mapping_cpu.py' -v

clean:
	rm -rf build
