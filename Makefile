file_name := iou
ext_name := iou_cpp

all: clean format build test

build:
	$(info Assumes conda is used with a path ${CONDA_PREFIX})
	clang++ --std=c++17 ${file_name}.cpp -c -o ${ext_name}.o -I $(wildcard ${CONDA_PREFIX}/include/python*/) -fPIC
	clang++ -o ${ext_name}.so -shared ${ext_name}.o -lboost_numpy3 -lboost_python3

test:
	python -c "import numpy as np; import ${ext_name}; print(${ext_name}.iou(np.zeros((3,5)), np.zeros((4,5))))"

format:
	clang-format -i --style=Google ${file_name}.cpp

clean:
	rm -f *.o *.so || true
