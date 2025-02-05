# Brief description of scripts

## Requirements
* python: 3.10
* pytorch
* scikit-learn
* mediapipe (pip)
* pyautogui (pip)
* numpy
* rerun (optional, for data visualisation)

## aerialmouse.py
Controls mouse location with your index finger. Left click with extended thumb. Right click with same thumb motion, plus extended pinkie.

To be added: extra hand gestures for additional macros

```python
python aerialmouse.py
```

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
