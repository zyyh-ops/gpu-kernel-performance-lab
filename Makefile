NVCC ?= nvcc
CUDA_ARCH ?= sm_89
CUDA_FLAGS ?= -O3 -std=c++17 -lineinfo

.PHONY: cuda-transpose-test cuda-transpose-bench cuda-double-buffer-test cuda-double-buffer-bench offline-audit release-offline gpu-smoke clean

build/matrix_transpose: cuda/memory/matrix_transpose.cu cuda/memory/transpose_kernels.cuh
	mkdir -p build
	$(NVCC) $(CUDA_FLAGS) -arch=$(CUDA_ARCH) cuda/memory/matrix_transpose.cu -o $@

cuda-transpose-test: build/matrix_transpose
	./build/matrix_transpose --mode test --sizes 1,7,31,32,33,47,63,65,129

cuda-transpose-bench: build/matrix_transpose
	./build/matrix_transpose --mode bench --n 8192 --warmup 5 --repeat 50 --samples 5 --json results/summary/transpose_8192.json

build/double_buffer: cuda/optimization/double_buffer.cu cuda/optimization/double_buffer_kernels.cuh
	mkdir -p build
	$(NVCC) $(CUDA_FLAGS) -arch=$(CUDA_ARCH) cuda/optimization/double_buffer.cu -o $@

cuda-double-buffer-test: build/double_buffer
	./build/double_buffer --mode test --test-sizes 1,7,15,16,17,31,32,33,47,48,65,129,256

cuda-double-buffer-bench: cuda-double-buffer-test
	./build/double_buffer --mode bench --n 4096 --extra-kib 0,16,24,32,44 --warmup 5 --repeat 20 --samples 5 --json results/summary/double_buffer_4096.json

offline-audit:
	python scripts/offline_audit.py
	python -m unittest discover -s tests -p 'test_*mapping_cpu.py' -v

release-offline:
	python scripts/release_check.py

gpu-smoke: cuda-transpose-test cuda-double-buffer-test
	python -m unittest discover -s tests -p 'test_matmul.py' -v
	python -m unittest discover -s tests -p 'test_paged_kv.py' -v

clean:
	rm -rf build
