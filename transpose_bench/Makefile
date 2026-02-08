PYTHON=python3

.PHONY: all build install bench roofline clean

all: build

build:
	$(PYTHON) setup.py build_ext --inplace

install:
	$(PYTHON) setup.py install

bench: build
	$(PYTHON) benchmark.py

roofline: build
	$(PYTHON) roofline.py

clean:
	rm -rf build *.so __pycache__ kernels/__pycache__
