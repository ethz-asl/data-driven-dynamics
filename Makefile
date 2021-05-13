root_dir:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

version?=latest
registry?=ethzasl/data-driven-dynamics
model?=quadrotor_model
log?=${root_dir}/resources/${model}.ulg

submodulesupdate:
	git submodule update --init --recursive

install-dependencies:
	pip3 install -r Tools/parametric_model/requirements.txt

docker-build:
	docker build -f docker/Dockerfile --tag ${registry}:${version} .

docker-push:
	docker push ${registry}:${version}

docker-publish: docker-build docker-push

docker-run:
	xhost local:root
	docker run -it --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ${registry}:${version} /bin/bash

estimate-model:
	python3 Tools/parametric_model/generate_parametric_model.py --model ${model} ${log}
