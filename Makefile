root_dir:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

version?=latest
registry?=ethzasl/data-driven-dynamics
model?=quadrotor_model
log?=${root_dir}/resources/${model}.ulg
config?=${root_dir}/Tools/parametric_model/configs/${model}.yaml
data_selection?=none
selection_variable?=none
plot?=True

submodulesupdate:
	git submodule update --init --recursive

install-dependencies:
	pip3 install -r Tools/parametric_model/requirements.txt

install-full-depdencies: install-dependencies
	pip3 install -r Tools/parametric_model/visual_dataframe_selector/requirements.txt

docker-build:
	docker build -f docker/Dockerfile --tag ${registry}:${version} .

docker-push:
	docker push ${registry}:${version}

docker-publish: docker-build docker-push

docker-run:
	xhost local:root
	docker run -it --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
	${registry}:${version} /bin/bash

estimate-model:
	python3 Tools/parametric_model/generate_parametric_model.py \
	--config ${config} \
	--data_selection ${data_selection} \
	--selection_variable ${selection_variable} \
	--plot ${plot} \
	${log}

predict-model:
	python3 Tools/parametric_model/predict_model.py --config ${config} \
	--model_results ${model_results} \
	--data_selection ${data_selection} \
	--selection_variable ${selection_variable} \
	${log}

format:
	Tools/fix_code_style.sh .