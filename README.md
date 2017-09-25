# Tensorflow OOP
Object Oriented Programming with TensorFlow

## Install
Library installation command:
```
python setup.py install
```

## Dependency
* Python 2.7+ or 3.5+
* [NumPy](https://github.com/numpy/numpy)
* [TensorFlow](https://github.com/tensorflow/tensorflow)

Requirements installation command:
```
sudo pip install -r requirements.txt
```

## Import
For importing Python module location should be added to sys.path variable:
```python
import tensorflow_oop as tfoop
```

## Example
Usage examples for MNIST located in folder 'example'.

## Test
Unittest scripts located in folder 'test'.

## Uninstall
Library uninstallation commands:
```
python setup.py install --record files.txt
cat files.txt | xargs rm -rf
```

## ToDo
1. Add decorator for dataset last batch
2. Add decorator for sequences padding and dynamic lengths
3. Add indexes to batch
4. Add placeholders for margin and exlude hard properties
5. Add dropout
6. Add loading local variables
7. Add freeze
8. Add word2vec implementation