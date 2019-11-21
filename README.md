<img align="left" width="190" height="190" src="logo/logo_transparent.png">

# SimplyNN
<p align="left">
    <a href="LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg" /></a>
    <a href="http://hits.dwyl.io/digantamisra98/SimplyNN" alt="HitCount">
        <img src="http://hits.dwyl.io/digantamisra98/SimplyNN.svg" /></a>
</p>
<br>
<br>
<br>

**SimplyNN** is a TF.keras wrapper which allows to build simple network architectures with various parameter configuration in a single line. 

## Fully Connected FeedForward Network:

### Using default activation function:

```
input_shape = (28,28)
num_classes = 10
num_layers = 3
metrics = ['accuracy']
model = FNN(input_shape = input_shape, num_classes = num_classes, num_layers = num_layers, metrics = metrics, dense_bool = False, activation = 'relu', optimizer = 'Adam', dropout = True, batch_norm = True)
```

### Using custom activation function Mish: 

*Mish* class and function definition:

```
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow import keras

def mish(x):
	return keras.layers.Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)
    
class Mish(Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return mish(inputs)

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
```

Now, building the network with Mish activation function: 

```
input_shape = (28,28)
num_classes = 10
num_layers = 3
metrics = ['accuracy']
model = FNN(input_shape = input_shape, num_classes = num_classes, num_layers = num_layers, metrics = metrics, dense_bool = False, activation = Mish, optimizer = 'Adam', dropout = True)
```
