CUDA            := nvcc
NVCCFLAGS       := -std=c++17 -O2 -I.
BUILD_DIR       := build
ARCHIVE_DIR     := archive/summer2025-assignment
OPS             := vector_add silu transpose rmsnorm reduction scan softmax topk gemm attention

OP              ?= vector_add
VARIANT         ?= naive

VARIANT_SRC     := ops/$(OP)/variants/$(VARIANT).cu
TEST_SRC        := ops/$(OP)/tests/test_$(OP).cu
BENCH_SRC       := ops/$(OP)/bench/bench_$(OP).cu
TEST_BIN        := $(BUILD_DIR)/$(OP)_$(VARIANT)_test
BENCH_BIN       := $(BUILD_DIR)/$(OP)_$(VARIANT)_bench

.DEFAULT_GOAL := help

.PHONY: help list-ops assignment test bench clean

help:
	@echo "Learning-CUDA Lab"
	@echo ""
	@echo "Targets:"
	@echo "  make list-ops"
	@echo "  make assignment [VERBOSE=true]"
	@echo "  make test OP=vector_add VARIANT=naive"
	@echo "  make bench OP=vector_add VARIANT=naive"
	@echo "  make clean"

list-ops:
	@printf '%s\n' $(OPS)

assignment:
	$(MAKE) -C $(ARCHIVE_DIR) VERBOSE=$(VERBOSE)

test:
	@test -f $(TEST_SRC) || (echo "Missing test source: $(TEST_SRC)" && exit 1)
	@test -f $(VARIANT_SRC) || (echo "Missing variant source: $(VARIANT_SRC)" && exit 1)
	@mkdir -p $(BUILD_DIR)
	$(CUDA) $(NVCCFLAGS) $(TEST_SRC) $(VARIANT_SRC) -o $(TEST_BIN)
	./$(TEST_BIN)

bench:
	@test -f $(BENCH_SRC) || (echo "Missing benchmark source: $(BENCH_SRC)" && exit 1)
	@test -f $(VARIANT_SRC) || (echo "Missing variant source: $(VARIANT_SRC)" && exit 1)
	@mkdir -p $(BUILD_DIR)
	$(CUDA) $(NVCCFLAGS) $(BENCH_SRC) $(VARIANT_SRC) -o $(BENCH_BIN)
	./$(BENCH_BIN)

clean:
	rm -rf $(BUILD_DIR)
