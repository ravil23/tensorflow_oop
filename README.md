# TensorFlow OOP
Object Oriented Programming with TensorFlow

## Installation

### Dependency
* Python 2.7+ or 3.5+
* [NumPy](https://github.com/numpy/numpy)
* [TensorFlow](https://github.com/tensorflow/tensorflow)

Requirements installation command:
```
sudo pip install -r requirements.txt
```

### Install
Library installation command:
```
python setup.py install
```

## Uninstall
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

### Example
Usage examples for MNIST located in folder 'example'. Run with '--help' option for more information.
```
python example/*.py
```

## Test
Unittest scripts located in folder 'test'.
```
python test/test_bag_of_words.py; \
python test/test_dataset.py; \
python test/test_sequence.py; \
python test/test_tripletset.py
```

## Generate documentation
```
doxygen doc/config.txt
```

## ToDo
1. Add word2vec implementation