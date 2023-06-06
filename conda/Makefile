BUCKET=wfp-vam-www
VERSION=0.1.2
BUILD=0
HDC_INDEX=../../hdc-conda-index

.PHONY: all
all: build

.PHONY: build
build:
	env VERSION=$(VERSION) BUILD=$(BUILD) conda build --no-test --no-anaconda-upload --output-folder ./build recipe

.PHONY: refresh-index
refresh-index:
	conda index -s noarch -n hdc $(HDC_INDEX)
	cp ./build/noarch/*tar.bz2 $(HDC_INDEX)/noarch/
	conda index -s noarch -n hdc $(HDC_INDEX)

sync-index:
	cd $(HDC_INDEX) && aws s3 sync --dryrun s3://$(BUCKET)/hdc/ .
	sleep 5
	cd $(HDC_INDEX) && aws s3 sync s3://$(BUCKET)/hdc/ .

.PHONY: upload
upload: refresh-index
	@echo Check Upload:
	cd $(HDC_INDEX) && aws s3 sync --exclude='**/.cache/*' --dryrun . s3://$(BUCKET)/hdc/
	@echo "Will do for real in 5 seconds"
	@sleep 5
	cd $(HDC_INDEX) && aws s3 sync --exclude='**/.cache/*' . s3://$(BUCKET)/hdc/