# *********************************************************************
# Learning-CUDA Makefile (最终修复版)
# *********************************************************************

CC              := nvcc
CFLAGS          := -std=c++17 -O0
TARGET          := test_kernels
STUDENT_SRC     := src/kernels.cu
STUDENT_OBJ     := $(STUDENT_SRC:.cu=.o)
TEST_OBJ        := tester/tester.o
TEST_VERBOSE_FLAG := --verbose
VERBOSE         :=

VERBOSE_ARG := $(if $(filter true True TRUE, $(VERBOSE)), $(TEST_VERBOSE_FLAG), )

.PHONY: all build run clean

all: build run

build: $(TARGET)
	@echo "=== Compiling and linking ==="
	$(CC) $(CFLAGS) -c $(STUDENT_SRC) -o $(STUDENT_OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(STUDENT_OBJ) $(TEST_OBJ)

run: $(TARGET)
	@echo "=== Running tests ==="
	./$(TARGET) $(VERBOSE_ARG) | tee test_output.tmp
	@if [ -n "$(VERBOSE_ARG)" ]; then \
		echo -e "\n=== Avg time 总和 ==="; \
		awk ' \
			/=== Running kthLargest Tests ===/ { flag="kth" } \
			/=== Running Attention Tests ===/ { flag="att" } \
			/Avg time:/ { \
				if (flag=="kth") kth_sum += $$3; \
				else if (flag=="att") att_sum += $$3; \
			} \
			END { \
				printf "kthLargest Tests 总时间: %.6f ms\n", kth_sum; \
				printf "Attention Tests 总时间: %.6f ms\n", att_sum; \
				printf "所有测试总时间: %.6f ms\n", kth_sum + att_sum \
			}' test_output.tmp; \
	fi
	@rm -f test_output.tmp

clean:
	@echo "=== Cleaning ==="
	rm -f $(TARGET) $(STUDENT_OBJ) test_output.tmp
