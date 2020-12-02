# Requirement:

- nvidia driver >= 455
- docker with nvidia container support

# Setup:

---

```
docker run --gpus=all --rm -it -v $(pwd):/workplace nvcr.io/nvidia/tensorrt:20.11-py3 bash

# In docker
cd /workplace
pip install -r requirements.txt

python inference omsi_conf.yaml
```
