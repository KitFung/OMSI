# Requirement:

- nvidia driver >= 455
- docker with nvidia container support

# Setup:

specify ONNX or TRT runtime in main().
```
USE_ONNX = True
```
if gpu avaiable:
```
pip install onnxruntime-gpu
```
cpu only:
```
pip install onnxruntime
```


---

```
docker run --gpus=all --rm -it -v $(pwd):/workplace nvcr.io/nvidia/tensorrt:20.11-py3 bash

# In docker
cd /workplace
pip install -r requirements.txt

python inference.py omsi_conf.yaml
```

# Jetson Nano run
```
python3 inference.py ./configs/wingk/omsi_conf.yaml 2 ~/tiny-imagenet-200/train
```