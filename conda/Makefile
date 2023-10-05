VERSION?=
BUILD?=0
HDC_INDEX=../../hdc-conda-index

.PHONY: all
all: build

.PHONY: build
build:
	env VERSION=$(VERSION) BUILD=$(BUILD) conda build --no-test --no-anaconda-upload --output-folder ./build recipe

.PHONY: build-meta
build-meta:
	env VERSION=$(VERSION) BUILD=$(BUILD) conda build --no-test --no-anaconda-upload --output-folder ./build recipe-meta

.PHONY: refresh-index
refresh-index:
	conda index -s noarch -n hdc $(HDC_INDEX)
	cp ./build/noarch/*tar.bz2 $(HDC_INDEX)/noarch/
	conda index -s noarch -n hdc $(HDC_INDEX)
