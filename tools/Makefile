# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

# If you want to use prebuild kaldi, make KALDI=<path/to/kaldi_root>
KALDI :=
# Specify cuda root path installed in your environment
CUDA_PATH := /usr/local/cuda
CUDA_VERSION := $(shell $(CUDA_PATH)/bin/nvcc --version | tail -n1 | awk '{print substr($$5,0,length($$5)-1)}')

# Get the pytorch install command
ifeq ($(CUDA_VERSION),10.2)
PYTORCH_INSTALL_CMD := miniconda3/envs/eend/bin/pip install torch==1.5.0 torchvision==0.6.0
else ifeq ($(CUDA_VERSION),10.1)
PYTORCH_INSTALL_CMD := miniconda3/envs/eend/bin/pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
else ifeq ($(CUDA_VERSION),9.2)
PYTORCH_INSTALL_CMD := miniconda3/envs/eend/bin/pip install torch==1.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
else
$(info CUDA $(CUDA_VERSION) not supported)
all:
endif

all: kaldi miniconda3/envs/eend/bin EEND env.sh

ifneq ($(strip $(KALDI)),)
kaldi:
	ln -s $(abspath $(KALDI)) kaldi
else
kaldi:
	git clone https://github.com/kaldi-asr/kaldi.git
	cd kaldi; git checkout 797905b0140f7169caf3d97c75a1a56a92f67d38
	cd kaldi/tools; $(MAKE)
	cd kaldi/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
endif

miniconda3.sh:
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $@

miniconda3: miniconda3.sh
	# -b: non-interactive install
	# -p: installed directory
	bash miniconda3.sh -b -p miniconda3

padertorch:
	git clone https://github.com/fgnt/padertorch.git

# virtual environment of python
miniconda3/envs/eend/bin: miniconda3 padertorch
	miniconda3/bin/conda update -y conda
	miniconda3/bin/conda env create -f environment.yml
	miniconda3/envs/eend/bin/python -m pip install --upgrade pip
	$(PYTORCH_INSTALL_CMD)
	# padertorch
	cd padertorch; git checkout fa6df6a0eae4fd4455c4d82644c2e082ac28ecb2
	cd padertorch; ../miniconda3/envs/eend/bin/python setup.py bdist_wheel
	miniconda3/envs/eend/bin/pip install padertorch/dist/padertorch-0.0.0-py3-none-any.whl

EEND:
	git clone https://github.com/hitachi-speech/EEND.git
	cd EEND; git checkout 9a0f211ce7e377eaea242490c3d7ec0f6adab8af
	# Add run_prepare_shared_eda.sh
	cd EEND; git show ddf4df4da55a493f550b6c45689b2d60a716c06d:egs/callhome/v1/run_prepare_shared_eda.sh > egs/callhome/v1/run_prepare_shared_eda.sh
	chmod u+x EEND/egs/callhome/v1/run_prepare_shared_eda.sh

update:
	miniconda3/bin/conda env update -f environment.yml

env.sh: EEND
	cp EEND/tools/env.sh.in env.sh
	echo "export LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$$LD_LIBRARY_PATH" >> env.sh
