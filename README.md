# Brief description of scripts

## Requirements
* python: 3.10
* pytorch
* scikit-learn
* mediapipe (pip)
* pyautogui
* numpy
* rerun (optional, for data visualisation)

## simplecursorcontrol.py
Controls mouse location with your index finger.

```python
python simplecursorcontrol.py
```
## datacollect.py
Collect data of your hand gestures. Hold down class number to keep collecting keypoint positions of hand in that class.

```python
python datacollect.py
```

## datatrain.py
Train data of your hand gestures. 

```python
python datatrain.py
```

## datalive.py
Use saved model on live camera stream.

```python
python datalive.py
```