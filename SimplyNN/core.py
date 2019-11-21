""" SimplyNN: Core Module - Sequential Network Generator
    Author: Diganta Misra
    Data: 11/21/2019
    License: Apache License 2.0
    See https://github.com/digantamisra98/SimplyNN/blob/master/LICENSE for more details
"""

###Import Necessary Modules

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, models

def FNN(input_shape, num_classes, num_layers, metrics, base_dense_units = 64, dense_bool = True,  activation = 'relu', optimizer = 'SGD', loss = 'categorical_crossentropy', dropout = False, drop_value = 0.25, 
        batch_norm = False, custom_act = False, summary = True):
    """Constructs a Fully Connected Feed Forward Neural Network Architecture
    
        Input Parameters:
        
        1. 'input_shape' - The Input shape of the data. Must be in 'tuple' format.
        2. 'num_classes' - Number of classes the data is divided into. dtype - integer.
        3. 'base_dense_units' - Number of dense units in the dense layers. Default Value - 64. dtype - integer.
        4. 'dense_bool' - Accepts either 'True' or 'False' (boolean inputs). If 'True', assigns each dense layer in the network to have the same number of dense units. If 'False', assigns each dense layer in the network to have dense units in
                          increment multiplicative order of 2 over the value of base_dense_units. For example - 64, 128, 256, 512, 1024.
        5. 'num_layers' - Number of dense layers in the model architecture excluding the input and output layers. dtype - integer.
        6. 'activation' - Activation Function to be used in the network. Refer to https://www.tensorflow.org/api_docs/python/tf/keras/activations for available activation functions in TF.keras. To use a custom activation define the function and
                          pass the function name without quotations in this argument. For example - activation = mish.
        7. 'optimizer' - Optimizer to be used in the network. Refer to https://www.tensorflow.org/api_docs/python/tf/keras/optimizers for available optimizers in TF.keras. Additionally, you can pass in your own custom optimizer function name which
                         has been previously defined. For instance, optimizer = 'ranger'.
        8. 'loss' - Loss Function to be used in the network. Refer to https://www.tensorflow.org/api_docs/python/tf/keras/losses for available loss functions in TF.keras.
        9. 'metrics' - Network performance evaluation criteria metrics. Refer to https://www.tensorflow.org/api_docs/python/tf/keras/metrics for available metrics in TF.keras. dtype - list.
        10. 'dropout' - Uses dropout layer within each dense layer block. Accepts either 'True' or 'False'. Default value - 'False'.
        11. 'drop_value' - Dropout value for the dropout layers. Default - 0.25. dtype - float.
        12. 'batch_norm' - Uses Batch Normalization layer within each dense layer block. Accepts either 'True' or 'False'. Default value - 'False'.
        13. 'custom_act' - Use custom activation function as defined in the parameter 'activation'. Accepts either 'True' or 'False'. Default value - 'False'. Used for individual layer implementation for activation function. Please define the custom
                           activation as a layer inherited class.
        14. 'summary' - Prints the Model Summary. Accepts either 'True' or 'False'. Default value - 'True'.
        
        Returns:
        
        A compiled network model with the specified parameters as given above."""
    
    
    ### Initializes the final layer activation function. Sigmoid if Binary Classification else Softmax.
    if num_classes == 2:
        last_act = 'sigmoid'
    else:
        last_act = 'softmax'

    
    ### Initializes the dense units for the dense layers in the network.
    base = base_dense_units
    dense_units = [base_dense_units]

    if dense_bool == False:    
        for units in range(num_layers-1):
            base = base * 2
            dense_units.append(base)
    else:
        for units in range(num_layers-1):
            dense_units.append(base)


    ### Intializes the Optimizer.
    if optimizer == 'SGD':
        optim = tf.keras.optimizers.SGD()
    elif optimizer == 'Adam':
        optim = tf.keras.optimizers.Adam()
    elif optimizer == 'RMSprop':
        optim = tf.keras.optimizers.RMSprop()
    elif optimizer == 'Nadam':
        optim = tf.keras.optimizers.Nadam()
    elif optimizer == 'Adadelta':
        optim = tf.keras.optimizers.Adadelta()
    elif optimizer == 'Adagrad':
        optim = tf.keras.optimizers.Adagrad()
    elif optimizer == 'Adamax':
        optim = tf.keras.optimizers.Adamax()
    else:
        optim = optimizer

    
    ### Intializes the custom activation function. 
    if custom_act == True:
        act = activation + "()"

    print("Defining a " + str(num_layers) + " layered network initialized with " + activation + " and " + optimizer + " Optimization.")


    ### Constructs the sequential model
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    for layer in range(num_layers):
        if batch_norm == True and dropout == True:
            model.add(layers.Dense(dense_units[layer]))
            model.add(layers.normalization.BatchNormalization())
            if custom_act == False:
                model.add(tf.keras.layers.Activation(activation))
            else:
                model.add(act)
            model.add(layers.Dropout(drop_value))
        elif batch_norm == True and dropout == False:
            model.add(layers.Dense(dense_units[layer]))
            model.add(layers.normalization.BatchNormalization())
            if custom_act == False:
                model.add(tf.keras.layers.Activation(activation))
            else:
                model.add(act)
        elif batch_norm == False and dropout == True:
            model.add(layers.Dense(dense_units[layer], activation=activation))
            model.add(layers.Dropout(drop_value))
        else:
            model.add(layers.Dense(dense_units[layer], activation=activation))
        
    model.add(layers.Dense(num_classes, activation=last_act))

    ### Model Summary
    if summary == True:
        print(model.summary())
    
    ### Model Compilation
    model.compile(optimizer=optim,
              loss=loss,
              metrics=metrics)
    return model