version?=latest
registry?=ethzasl/data-driven-dynamics

docker-build:
	docker build -f docker/Dockerfile --tag ${registry}:${version} .

docker-push:
	docker push ${registry}:${version}

docker-publish: docker-build docker-push

docker-run:
	xhost local:root
	docker run -it --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ${registry}:${version} /bin/bash
