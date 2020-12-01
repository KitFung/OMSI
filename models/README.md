# Prepare pruned models

### going into distiller
```
conda activate distiller
cd distiller/examples/classifier_compression/
```


### pruning

mobilenet
```
ARCHNAME=mobilenet && CONFNAME='mobilenet.schedule_sensitivity.yaml' && \
python compress_classifier.py -a $ARCHNAME --lr 0.0045 -p 50 ~/tiny-imagenet-200 -j 4 -b 128 --epochs 200 --compress=~/pruned_models/$CONFNAME
```

resnet18
```
ARCHNAME=resnet18 && CONFNAME='resnet18.schedule_sensitivity.yaml' && \
python compress_classifier.py -a $ARCHNAME --lr 0.0045 -p 50 ~/tiny-imagenet-200 -j 4 -b 128 --epochs 200 --compress=~/pruned_models/$CONFNAME
```

inceptionv4
```
ARCHNAME=inceptionv4 && CONFNAME='inceptionv4.schedule_sensitivity.yaml' && \
python compress_classifier.py -a $ARCHNAME --lr 0.0045 -p 50 ~/tiny-imagenet-200 -j 4 -b 128 --epochs 200 --compress=~/pruned_models/$CONFNAME

```

### export-onnx
```
ARCHNAME=mobilenet && PTHNAME=mobilenet_34.67_66.84.pth.tar && python compress_classifier.py ~/tiny-imagenet-200 --arch $ARCHNAME --resume-from "~/pruned_models/$PTHNAME" --evaluate --export-onnx
```
