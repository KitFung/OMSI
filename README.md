nvcr.io/nvidia/tensorrt:20.11-py3

docker run --gpus=all --rm -it -v $(pwd):/workplace nvcr.io/nvidia/tensorrt:20.11-py3 bash
