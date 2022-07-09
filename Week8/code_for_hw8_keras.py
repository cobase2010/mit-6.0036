# This code file pertains to problems 3 onwards from homework 8
# Here, we use out-of-the-box neural network frameworks Keras and Tensorflow 
# to build and train our models

import pdb
import numpy as np
import itertools
# np.random.seed(0)
import math as m 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import mnist
# from tensorflow.keras import backend as K
import tensorflow.python.keras.backend as K
from tensorflow.keras.initializers import VarianceScaling
from matplotlib import pyplot as plt

from code_for_hw8_oop import ReLU

######################################################################
# Problem 3 - 2D data
######################################################################

def archs(classes):
    return [[Dense(input_dim=2, units=classes, activation="softmax")],
            [Dense(input_dim=2, units=10, activation='relu'),
             Dense(units=classes, activation="softmax")],
            [Dense(input_dim=2, units=100, activation='relu'),
             Dense(units=classes, activation="softmax")],
            [Dense(input_dim=2, units=10, activation='relu'),
             Dense(units=10, activation='relu'),
             Dense(units=classes, activation="softmax")],
            [Dense(input_dim=2, units=100, activation='relu'),
             Dense(units=100, activation='relu'),
             Dense(units=classes, activation="softmax")]]

# Read the simple 2D dataset files
def get_data_set(name):
    try:
        data = np.loadtxt(name, skiprows=0, delimiter = ' ')
    except:
        return None, None, None
    np.random.shuffle(data)             # shuffle the data
    # The data uses ROW vectors for a data point, that's what Keras assumes.
    _, d = data.shape
    X = data[:,0:d-1]
    Y = data[:,d-1:d]
    y = Y.T[0]
    classes = set(y)
    if classes == set([-1.0, 1.0]):
        print('Convert from -1,1 to 0,1')
        y = 0.5*(y+1)
    print('Loading X', X.shape, 'y', y.shape, 'classes', set(y))
    return X, y, len(classes)

######################################################################
# General helpers for Problems 3-5
######################################################################

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.keys = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        self.values = {}
        for k in self.keys:
            self.values['batch_'+k] = []
            self.values['epoch_'+k] = []

    def on_batch_end(self, batch, logs={}):
        for k in self.keys:
            bk = 'batch_'+k
            if k in logs:
                self.values[bk].append(logs[k])

    def on_epoch_end(self, epoch, logs={}):
        for k in self.keys:
            ek = 'epoch_'+k
            if k in logs:
                self.values[ek].append(logs[k])

    def plot(self, keys):
        for key in keys:
            plt.plot(np.arange(len(self.values[key])), np.array(self.values[key]), label=key)
        plt.legend()

def run_keras(X_train, y_train, X_val, y_val, X_test, y_test, model, epochs, split=0, verbose=True):
    
    # Define the optimization
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])
    N = X_train.shape[0]
    # Pick batch size
    batch = 32 if N > 1000 else 1     # batch size
    history = LossHistory()
    # Fit the model
    if X_val is None:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_split=split,
                  callbacks=[history], verbose=verbose)
    else:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_val, y_val),
                  callbacks=[history], verbose=verbose)
    # Evaluate the model on validation data, if any
    if X_val is not None or split > 0:
        val_acc, val_loss = history.values['epoch_val_accuracy'][-1], history.values['epoch_val_loss'][-1]
        print ("\nLoss on validation set:"  + str(val_loss) + " Accuracy on validation set: " + str(val_acc))
    else:
        val_acc = None
    # Evaluate the model on test data, if any
    if X_test is not None:
        test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=batch)
        print ("\nLoss on test set:"  + str(test_loss) + " Accuracy on test set: " + str(test_acc))
    else:
        test_acc = None
    return model, history, val_acc, test_acc

def dataset_paths(data_name):
    return ["data/data"+data_name+"_"+suffix+".csv" for suffix in ("train", "validate", "test")]

def reset_weights(model):
  for layer in model.layers: 
    if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            reset_weights(layer) #apply function recursively
            continue

    #where are the initializers?
    if hasattr(layer, 'cell'):
        init_container = layer.cell
    else:
        init_container = layer

    for key, initializer in init_container.__dict__.items():
        if "initializer" not in key: #is this item an initializer?
                continue #if no, skip it

        # find the corresponding variable, like the kernel or the bias
        if key == 'recurrent_initializer': #special case check
            var = getattr(init_container, 'recurrent_kernel')
        else:
            var = getattr(init_container, key.replace("_initializer", ""))

        var.assign(initializer(var.shape, var.dtype))
        #use the initializer

def reinitialize_model(model):
  weights = []
  initializers = []
  for layer in model.layers:
    if isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D)):
      weights += [layer.kernel, layer.bias]
      initializers += [layer.kernel_initializer, layer.bias_initializer]
    elif isinstance(layer, keras.layers.BatchNormalization):
      weights += [layer.gamma, layer.beta, layer.moving_mean, layer.moving_variance]
      initializers += [layer.gamma_initializer,
                       layer.beta_initializer,
                       layer.moving_mean_initializer,
                       layer.moving_variance_initializer]
  for w, init in zip(weights, initializers):
    w.assign(init(w.shape, dtype=w.dtype))

# The name is a string such as "1" or "Xor"
def run_keras_2d(data_name, layers, epochs, display=True, split=0.25, verbose=True, trials=1):
    # Model specification
    model = Sequential()
    for layer in layers:
        model.add(layer)
    print(model.summary())
    print('Keras FC: dataset=', data_name)
    (train_dataset, val_dataset, test_dataset) = dataset_paths(data_name)
    # Load the datasets
    X_train, y, num_classes = get_data_set(train_dataset)
    X_val, y2, _ = get_data_set(val_dataset)
    X_test, y3, _ = get_data_set(test_dataset)
    # Categorize the labels
    y_train = to_categorical(y, num_classes) # one-hot
    y_val = y_test = None
    if X_val is not None:
        y_val = to_categorical(y2, num_classes) # one-hot        
    if X_test is not None:
        y_test = to_categorical(y3, num_classes) # one-hot
    val_acc, test_acc = 0, 0
    for trial in range(trials):
        # Reset the weights
        # See https://github.com/keras-team/keras/issues/341
        reset_weights(model)
        # reinitialize_model(model)
        # the following code no longer works in tf2
        # session2 = tf.compat.v1.keras.backend.get_session()
        # session2 = K.get_session()
        # for layer in layers:
            # for v in layer.__dict__:
            #     v_arg = getattr(layer, v)
            #     if hasattr(v_arg, 'initializer'):
            #         initializer_func = getattr(v_arg, 'initializer')
            #         # initializer_func.run(session=session2)
            #         session2.run(initializer_func)
            
        # Run the model
        model, history, vacc, tacc, = \
               run_keras(X_train, y_train, X_val, y_val, X_test, y_test, model, epochs,
                         split=split, verbose=verbose)
        val_acc += vacc if vacc else 0
        test_acc += tacc if tacc else 0
        if display:
            # plot classifier landscape on training data
            plot_heat(X_train, y, model)
            plt.title('Training data')
            plt.show()
            if X_test is not None:
                # plot classifier landscape on testing data
                plot_heat(X_test, y3, model)
                plt.title('Testing data')
                plt.show()
            # Plot epoch loss
            history.plot(['epoch_loss', 'epoch_val_loss'])
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title('Epoch val_loss and loss')
            plt.show()
            # Plot epoch accuracy
            history.plot(['epoch_accuracy', 'epoch_val_accuracy'])
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.title('Epoch val_acc and acc')
            plt.show()
    if val_acc:
        print ("\nAvg. validation accuracy:"  + str(val_acc/trials))
    if test_acc:
        print ("\nAvg. test accuracy:"  + str(test_acc/trials))
    return X_train, y, model



######################################################################
# Helper functions for 
# OPTIONAL: Problem 4 - Weight Sharing
######################################################################

def generate_1d_images(nsamples,image_size,prob):
    Xs=[]
    Ys=[]
    for i in range(0,nsamples):
        X=np.random.binomial(1, prob, size=image_size)
        Y=count_objects_1d(X)
        Xs.append(X)
        Ys.append(Y)
    Xs=np.array(Xs)
    Ys=np.array(Ys)
    return Xs,Ys


#count the number of objects in a 1d array
def count_objects_1d(array):
    count=0
    for i in range(len(array)):
        num=array[i]
        if num==0:
            if i==0 or array[i-1]==1:
                count+=1
    return count

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))    


def filter_reg(weights):
    lam=1000
    filter_result = K.abs(K.sum(weights))
    return lam * filter_result

def get_image_data_1d(tsize,image_size,prob):
    #prob controls the density of white pixels
    #tsize is the size of the training and test sets
    vsize=int(0.2*tsize)
    X_train,Y_train=generate_1d_images(tsize,image_size,prob)
    X_val,Y_val=generate_1d_images(vsize,image_size,prob)
    X_test,Y_test=generate_1d_images(tsize,image_size,prob)
    #reshape the input data for the convolutional layer
    X_train=np.expand_dims(X_train,axis=2)
    X_val=np.expand_dims(X_val,axis=2)
    X_test=np.expand_dims(X_test,axis=2)
    data=(X_train,Y_train,X_val,Y_val,X_test,Y_test)
    return data

def train_neural_counter(layers,data,loss_func='mse',display=False):
    (X_train,Y_train,X_val,Y_val,X_test,Y_test)=data
    epochs=10
    batch=1
    
    model=Sequential()
    for layer in layers:
        model.add(layer)
    model.summary()    
    model.compile(loss=loss_func, optimizer=Adam())
    history = LossHistory()    
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, validation_data=(X_val, Y_val),callbacks=[history], verbose=True)
    err=model.evaluate(X_test,Y_test)
    ws=model.layers[-1].get_weights()[0]
    if display:
        plt.plot(ws)
        plt.show()
    return model,err

def problem_4b():
    imsize = 1024
    prob_white = 0.1

    num_filters = 1  # Your code
    kernel_size = 2  # Your code
    strides = 1  # Your code
    activation_conv = "relu"  # Your code

    (X_train,Y_train,X_val,Y_val,X_test,Y_test) = get_image_data_1d(1000,imsize,prob_white)

    layer1=keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, \
        strides=strides, use_bias=False, activation=activation_conv, batch_size=1, input_shape=(imsize,1), padding='same')

    activation_dense = "relu"  # Your code
    num_units = 1  # Your code
    layer3=Dense(units=num_units, activation=activation_dense, use_bias=False)

    layers=[layer1,Flatten(),layer3]

    # This is how we create the model using our layers
    model=Sequential()
    for layer in layers:
        model.add(layer)
            
    model.compile(loss='mse', optimizer=Adam()) 


    # Set the weights of the layers to desired values
    # We give you the lines to use for this part
    model.layers[0].set_weights([np.array([1/2,1/2]).reshape(2,1,1)])
    model.layers[-1].set_weights([np.ones(imsize).reshape(imsize,1)])

    print(model.summary())

    model.evaluate(X_test,Y_test)

def problem_4c():
    imsize = 1024
    prob_white = 0.1

    num_filters = 1  # Your code
    kernel_size = 2  # Your code
    strides = 1  # Your code
    activation_conv = "relu"  # Your code

    (X_train,Y_train,X_val,Y_val,X_test,Y_test) = get_image_data_1d(1000,imsize,prob_white)

    layer1=keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, \
        strides=strides, use_bias=False, activation=activation_conv, batch_size=1, input_shape=(imsize,1), padding='same')

    activation_dense = "relu"  # Your code
    num_units = 1  # Your code
    layer3=Dense(units=num_units, activation=activation_dense, use_bias=True)

    layers=[layer1,Flatten(),layer3]

    # This is how we create the model using our layers
    model=Sequential()
    for layer in layers:
        model.add(layer)
            
    model.compile(loss='mse', optimizer=Adam()) 


    # Set the weights of the layers to desired values
    # We give you the lines to use for this part
    model.layers[0].set_weights([np.array([1/2,1/2]).reshape(2,1,1)])
    model.layers[-1].set_weights([np.ones(imsize).reshape(imsize,1), np.array([-10]).reshape(1, )])


    print(model.summary())

    model.evaluate(X_test,Y_test)

def problem_4e():
    imsize = 1024
    prob_white = 0.1

    data=get_image_data_1d(1000, imsize, prob_white)
    trials=5
    for trial in range(trials):
    
        num_filters = 1  # Your code
        kernel_size = 2  # Your code
        strides = 1  # Your code
        activation_conv = "relu"  # Your code

        layer1=keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, kernel_regularizer=filter_reg,\
        strides=strides, use_bias=False, activation=activation_conv, batch_size=1, \
        input_shape=(imsize,1),padding='same')
        
        activation_dense = "relu"  # Your code
        num_units = 1  # Your code

        layer3=Dense(units=num_units, activation=activation_dense, use_bias=False)
        
        layers=[layer1,Flatten(),layer3]
        model,err = train_neural_counter(layers, data, 'mse')
        
        model.layers[0].get_weights()[0]
        np.mean(model.layers[-1].get_weights()[0])
        print(err)

def problem_4i():
    imsize = 128
    prob_white = 0.1

    data=get_image_data_1d(1000, imsize, prob_white)
    trials=5
    for trial in range(trials):
    
        num_filters = 1  # Your code
        kernel_size = 2  # Your code
        strides = 1  # Your code
        activation_conv = "relu"  # Your code

        layer1=keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size,\
        strides=strides, use_bias=False, activation=activation_conv, batch_size=1, \
        input_shape=(imsize,1),padding='same')
        
        layer2=keras.layers.MaxPool1D(pool_size=2)

        activation_dense = "relu"  # Your code
        num_units = 1  # Your code

        layer3=Dense(units=num_units, activation=activation_dense, use_bias=False)
        
        layers=[layer1, layer2, Flatten(),layer3]
        model,err = train_neural_counter(layers, data, 'mse')
        
        model.layers[0].get_weights()[0]
        np.mean(model.layers[-1].get_weights()[0])
        print(err)

# problem_4b()
# problem_4c()
# problem_4e()
# problem_4i()

######################################################################
# Problem 5
######################################################################

def shifted(X, shift):
    n = X.shape[0]
    m = X.shape[1]
    size = m + shift
    X_sh = np.zeros((n, size, size))
    plt.ion()
    for i in range(n):
        sh1 = np.random.randint(shift)
        sh2 = np.random.randint(shift)
        X_sh[i, sh1:sh1+m, sh2:sh2+m] = X[i, :, :]
        # If you want to see the shifts, uncomment
        #plt.figure(1); plt.imshow(X[i])
        #plt.figure(2); plt.imshow(X_sh[i])
        #plt.show()
        #input('Go?')
    return X_sh
  
def get_MNIST_data(shift=0):
    (X_train, y1), (X_val, y2) = mnist.load_data()
    if shift:
        size = 28+shift
        X_train = shifted(X_train, shift)
        X_val = shifted(X_val, shift)
    return (X_train, y1), (X_val, y2)

# Example Usage:
# train, validation = get_MNIST_data()

def run_keras_fc_mnist(train, test, layers, epochs, split=0.1, verbose=True, trials=1):
    (X_train, y1), (X_val, y2) = train, test
    # Flatten the images
    m = X_train.shape[1]
    X_train = X_train.reshape((X_train.shape[0], m*m))
    X_val = X_val.reshape((X_val.shape[0], m*m))
    # Categorize the labels
    num_classes = 10
    y_train = to_categorical(y1, num_classes)
    y_val = to_categorical(y2, num_classes)
    # Train, use split for validation
    val_acc, test_acc = 0, 0
    # Create model from layers
    model = Sequential()
    for layer in layers:
        model.add(layer)
    print(model.summary())
    for trial in range(trials):
        # Reset the weights
        # See https://github.com/keras-team/keras/issues/341
        reset_weights(model)
        # session = K.get_session()
        # session = tf.compat.v1.keras.backend.get_session()
        # for layer in layers:
        #     for v in layer.__dict__:
        #         v_arg = getattr(layer, v)
        #         if hasattr(v_arg, 'initializer'):
        #             initializer_func = getattr(v_arg, 'initializer')
        #             initializer_func.run(session=session)
        # Run the model
        model, history, vacc, tacc = \
                run_keras(X_train, y_train, X_val, y_val, None, None, model, epochs, split=split, verbose=verbose)
        val_acc += vacc if vacc else 0
        test_acc += tacc if tacc else 0
    if val_acc:
        print ("\nAvg. validation accuracy:"  + str(val_acc/trials))
    if test_acc:
        print ("\nAvg. test accuracy:"  + str(test_acc/trials))

def run_keras_cnn_mnist(train, test, layers, epochs, split=0.1, verbose=True, trials=1):
    # Load the dataset
    (X_train, y1), (X_val, y2) = train, test
    # Add a final dimension indicating the number of channels (only 1 here)
    m = X_train.shape[1]
    X_train = X_train.reshape((X_train.shape[0], m, m, 1))
    X_val = X_val.reshape((X_val.shape[0], m, m, 1))
    # Categorize the labels
    num_classes = 10
    y_train = to_categorical(y1, num_classes)
    y_val = to_categorical(y2, num_classes)

    # Create model from layers
    model = Sequential()
    for layer in layers:
        model.add(layer)


    model.build(input_shape=(None, m, m, 1))
    # print(model.summary())

    # Train, use split for validation
    val_acc, test_acc = 0, 0
    for trial in range(trials):
        # Reset the weights
        # See https://github.com/keras-team/keras/issues/341
        reinitialize_model(model)
        # session = K.get_session()
        # for layer in layers:
        #     for v in layer.__dict__:
        #         v_arg = getattr(layer, v)
        #         if hasattr(v_arg, 'initializer'):
        #             initializer_func = getattr(v_arg, 'initializer')
        #             initializer_func.run(session=session)
        # Run the model
        model, history, vacc, tacc = \
                run_keras(X_train, y_train, X_val, y_val, None, None, model, epochs, split=split, verbose=verbose)
        val_acc += vacc if vacc else 0
        test_acc += tacc if tacc else 0
    if val_acc:
        print ("\nAvg. validation accuracy:"  + str(val_acc/trials))
    if test_acc:
        print ("\nAvg. test accuracy:"  + str(test_acc/trials))

# Example usage:
train, validation = get_MNIST_data()
train_20, validation_20 = get_MNIST_data(shift=20) # Your code (fill in the shift)
# Scale the images
train2 =(train[0]/255.,train[1])  # Your code
validation2 = (validation[0]/255,validation[1])  # Your code
# Scale the images
train_20 = (train_20[0]/255., train_20[1])  # Your code
validation_20 = (validation_20[0]/255., validation_20[1])  # Your code
# 5B
# layers = [Dense(input_dim=784, units=10, activation='softmax')]

# 5H
# for num in [128,256,512,1024]:
#     layers = [Dense(input_dim=784, units=num, activation='relu'),\
#              Dense(units=10, activation="softmax")]  # Your code
#     run_keras_fc_mnist(train2, validation2, layers, 1, split=0.1, verbose=False, trials=5)

# 5I
# layers = [Dense(input_dim=784, units=512, activation='relu'), Dense(input_dim=512, units=256, activation='relu'), Dense(units=10, activation='softmax')]
# run_keras_fc_mnist(train2, validation2, layers, 1, split=0.1, verbose=False, trials=5)

# 5J

# layers = [Conv2D(filters=32, kernel_size=(3,3),strides=1, padding='valid', activation='relu'),\
#           MaxPooling2D(pool_size=(2, 2), padding='valid'),\
#           Conv2D(filters=64, kernel_size=(3,3),strides=1, padding='valid', activation='relu'),\
#           MaxPooling2D(pool_size=(2, 2),padding='valid'),\
#           Flatten(),\
#           Dense(units=128, activation='relu'),\
#           Dropout(rate=0.5),\
#           Dense(units=10, activation="softmax")]  # Your code  # Your code
# run_keras_cnn_mnist(train2, validation2, layers, 1, split=0.1, verbose=False, trials=5)
# Same pattern applies to the function: run_keras_cnn_mnist

# 5K
layers = [Dense(input_dim=48*48, units=512, activation='relu'), Dense(units=256, activation='relu'), Dense(units=10, activation='softmax')]
run_keras_fc_mnist(train_20, validation_20, layers, 1, split=0.1, verbose=False, trials=5)

layers = [Conv2D(filters=32, kernel_size=(3,3),strides=1, padding='valid', activation='relu'),\
          MaxPooling2D(pool_size=(2, 2), padding='valid'),\
          Conv2D(filters=64, kernel_size=(3,3),strides=1, padding='valid', activation='relu'),\
          MaxPooling2D(pool_size=(2, 2),padding='valid'),\
          Flatten(),\
          Dense(units=128, activation='relu'),\
          Dropout(rate=0.5),\
          Dense(units=10, activation="softmax")]  # Your code  # Your code
run_keras_cnn_mnist(train_20, validation_20, layers, 1, split=0.1, verbose=False, trials=5)


######################################################################
# Plotting Functions
######################################################################

def plot_heat(X, y, model, res = 200):
    eps = .1
    xmin = np.min(X[:,0]) - eps; xmax = np.max(X[:,0]) + eps
    ymin = np.min(X[:,1]) - eps; ymax = np.max(X[:,1]) + eps
    ax = tidyPlot(xmin, xmax, ymin, ymax, xlabel = 'x', ylabel = 'y')
    xl = np.linspace(xmin, xmax, res)
    yl = np.linspace(ymin, ymax, res)
    xx, yy = np.meshgrid(xl, yl, sparse=False)
    zz = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)
    im = ax.imshow(np.flipud(zz.reshape((res,res))), interpolation = 'none',
                   extent = [xmin, xmax, ymin, ymax],
                   cmap = 'viridis')
    plt.colorbar(im)
    for yi in set([int(_y) for _y in set(y)]):
        color = ['r', 'g', 'b'][yi]
        marker = ['X', 'o', 'v'][yi]
        cl = np.where(y==yi)
        ax.scatter(X[cl,0], X[cl,1], c = color, marker = marker, s=80,
                   edgecolors = 'none')
    return ax

def tidyPlot(xmin, xmax, ymin, ymax, center = False, title = None,
                 xlabel = None, ylabel = None):
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
    eps = .05
    plt.xlim(xmin-eps, xmax+eps)
    plt.ylim(ymin-eps, ymax+eps)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def plot_separator(ax, th, th_0):
    xmin, xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    pts = []
    eps = 1.0e-6
    # xmin boundary crossing is when xmin th[0] + y th[1] + th_0 = 0
    # that is, y = (-th_0 - xmin th[0]) / th[1]
    if abs(th[1,0]) > eps:
        pts += [np.array([x, (-th_0 - x * th[0,0]) / th[1,0]]) \
                                                        for x in (xmin, xmax)]
    if abs(th[0,0]) > 1.0e-6:
        pts += [np.array([(-th_0 - y * th[1,0]) / th[0,0], y]) \
                                                         for y in (ymin, ymax)]
    in_pts = []
    for p in pts:
        if (xmin-eps) <= p[0] <= (xmax+eps) and \
           (ymin-eps) <= p[1] <= (ymax+eps):
            duplicate = False
            for p1 in in_pts:
                if np.max(np.abs(p - p1)) < 1.0e-6:
                    duplicate = True
            if not duplicate:
                in_pts.append(p)
    if in_pts and len(in_pts) >= 2:
        # Plot separator
        vpts = np.vstack(in_pts)
        ax.plot(vpts[:,0], vpts[:,1], 'k-', lw=2)
        # Plot normal
        vmid = 0.5*(in_pts[0] + in_pts[1])
        scale = np.sum(th*th)**0.5
        diff = in_pts[0] - in_pts[1]
        dist = max(xmax-xmin, ymax-ymin)
        vnrm = vmid + (dist/10)*(th.T[0]/scale)
        vpts = np.vstack([vmid, vnrm])
        ax.plot(vpts[:,0], vpts[:,1], 'k-', lw=2)
        # Try to keep limits from moving around
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
    else:
        print('Separator not in plot range')

def plot_decision(data, cl, diff=False):
    layers = archs(cl)[0]
    X, y, model = run_keras_2d(data, layers, 10, trials=1, verbose=False, display=False)
    ax = plot_heat(X,y,model)
    W = layers[0].get_weights()[0]
    W0 = layers[0].get_weights()[1].reshape((cl,1))
    if diff:
        for i,j in list(itertools.combinations(range(cl),2)):
            plot_separator(ax, W[:,i:i+1] - W[:,j:j+1], W0[i:i+1,:] - W0[j:j+1,:])
    else:
        for i in range(cl):
            plot_separator(ax, W[:,i:i+1], W0[i:i+1,:])
    plt.show()

# layers = [Dense(input_dim=2, units=10, activation='relu'), Dense(units=2, activation='softmax')]

# 3G
# x, y, model = run_keras_2d("3class", archs(3)[0], 10, split=0.5, display=False, verbose=False, trials=1)
# weights = model.layers[0].get_weights()
# print(weights)
# z=[[-1,0], [1,0], [0,-11], [0,1], [-1,-1], [-1,1], [1,1], [1,-1]]

# prediction=model.predict(z)
# print(prediction)
# result = np.argmax(prediction,axis=1)
# print(result)



# layers2c = [Dense(input_dim=2, units=200, activation='relu'),
#              Dense(units=200, activation='relu'),
#              Dense(units=2, activation="softmax")]
# run_keras_2d("3", layers2c, 100, split=0.25, display=True, verbose=True, trials=1)

# 4.