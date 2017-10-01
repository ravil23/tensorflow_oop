# TensorFlow OOP
Object Oriented Programming with TensorFlow

## Docker
Easiest way for playing with this library without installation on your host machine.

### Get docker image
Pull from DockerHub:
```
docker pull thedatascience/tensorflow_oop:1.0.1
```
or build local image:
```
docker build -t tensorflow_oop .
```

### Run container
```
docker run -p 6006:6006 -it --rm tensorflow_oop
```
P.S. Open http://localhost:6006 in your browser after running TensorBoard.

## Installation

### Dependency
* Python 2.7+ or 3.5+
* [NumPy](https://github.com/numpy/numpy)
* [TensorFlow](https://github.com/tensorflow/tensorflow)

Requirements installation command:
```
sudo pip install -r requirements.txt
```

### Install command
Library installation command:
```
python setup.py install
```

### Uninstall command
Library uninstallation commands:
```
python setup.py install --record files.txt
cat files.txt | xargs rm -rf
```

## Usage

### Import
For importing Python module location should be added to sys.path variable:
```python
import tensorflow_oop as tfoop
```
or if you a little lazy:
```python
from tensorflow_oop import *
```

### Example
Usage examples for MNIST located in folder 'example'. Run with '--help' option for more information.
```
python example/<EXAMPLE_NAME>.py
```

## Test
Unittest scripts located in folder 'test'.
```
python test/test_bag_of_words.py; \
python test/test_dataset.py; \
python test/test_sequence.py; \
python test/test_tripletset.py
```

## Documentation

### Read
Open in your browser file doc/html/index.html

### Generate
```
doxygen doc/config.txt
```

## ToDo
1. Add word2vec implementation