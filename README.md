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
1. Add decorator for sequences padding and dynamic lengths
2. Add placeholders for margin and exlude hard properties
3. Add dropout
4. Add loading local variables
5. Add word2vec implementation
6. Don't create new saver if exist