# Tensorflow OOP
Object Oriented Programming with TensorFlow

## Dependency
* [Python 3.6.2+](https://www.python.org/downloads/release/python-362)
* [NumPy](https://github.com/numpy/numpy)
* [TensorFlow](https://github.com/tensorflow/tensorflow)

Installation command:
```
sudo pip install -r requirements.txt
```

## Import
For importing Python module location should be added to sys.path variable:
```python
import sys
include_dir = 'path_to_root_directory'
if include_dir not in sys.path:
    sys.path.append(include_dir)
import tensorflow_oop
```

## Example
Usage examples for MNIST located in folder 'example'.

## Test
Unittest scripts located in folder 'test'.