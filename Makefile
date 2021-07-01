root_dir:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

version?=latest
registry?=ethzasl/data-driven-dynamics
model?=quadrotor_model
log?=${root_dir}/resources/${model}.ulg
config?=${root_dir}/Tools/parametric_model/configs/${model}.yaml
data_selection?=False

submodulesupdate:
	git submodule update --init --recursive

install-dependencies:
	pip3 install -r Tools/parametric_model/requirements.txt
	pip3 install -r Tools/parametric_model/visual_dataframe_selector/requirements.txt

docker-build:
	docker build -f docker/Dockerfile --tag ${registry}:${version} .

docker-push:
	docker push ${registry}:${version}

docker-publish: docker-build docker-push

docker-run:
	xhost local:root
	docker run -it --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ${registry}:${version} /bin/bash

estimate-model:
	python3 Tools/parametric_model/generate_parametric_model.py --model ${model} --config ${config} --data_selection ${data_selection} ${log} 

