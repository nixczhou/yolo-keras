## How to install
```python
conda create -y -n keras-jax python=3.10
conda activate keras-jax
pip install -r requirements-jax-cuda.txt
```

## Change Backbone
Refer: https://keras.io/api/keras_cv/models/tasks/yolo_v8_detector/
```python
# Create the model with mobilenet backbone
model = keras_cv.models.YOLOV8Detector.from_preset(
    num_classes=20,
    bounding_box_format="xywh",
    preset="mobilenet_v3_large",
)
```

```python
# Can load pretrained weight if needed
model = keras_cv.models.YOLOV8Detector.from_preset(
    num_classes=20,
    bounding_box_format="xywh",
    preset="mobilenet_v3_large_imagenet",
    load_weights=True,
)
```

## Run Example Train
```python
python train.py
```

## Full Tutorial with Default Yolov8 Backbone
Link: https://keras.io/examples/vision/yolov8/
