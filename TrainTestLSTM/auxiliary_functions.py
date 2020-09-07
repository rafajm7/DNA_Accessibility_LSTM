import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, Conv1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
import keras
from keras.preprocessing.text import Tokenizer
import keras.backend as K
import pydot
import os
import graphviz
from itertools import product

def dict_product(input_dict):
    '''
    This function receives a dictionary where each value is a list and
    returns a list of dictionaries with all the possible combinations of values per key.

    Parameters:
        input_dict ({str:[int]}): The dictionary from which to extract the combinations

    Returns:
        The list of all possible combinations (list of dicts).
    '''

    return list(dict(zip(input_dict.keys(), values)) for values in product(*input_dict.values()))

def mean_confidence_interval(data, confidence=0.95):
    '''
    Function that builds a mean confidence interval around a given vector.

    Parameters:
        data (list(str)):The list from which to extract the mean confidence interval.

    Returns:
        A tuple with the mean value of the vector, and the minimum and maximum value of the interval
    '''

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def getKmers(sequence, k):
    '''
    Function that creates the k-mers of a given DNA sequence.

    Parameters:
        sequence (str): A str with the sequence from which to extract the k-mers
        k (int): The length of the k-mers

    Returns:
        A list with each element being a k-mer
    '''

    return [sequence[x:x+k].upper() for x in range(len(sequence) - k + 1)]

def specificity(y_true, y_pred):
    '''
    Keras custom metric function that returns the specificity of each epoch in training.

    Parameters:
        y_true: tensor with the true values of the output variable
        y_pred: tensor with the predicted values of the output variable

    Returns:
        The specificity of the predictions
    '''

    y_true_class = tf.cast(y_true >= 0.5, tf.float32)
    y_pred_class = tf.cast(y_pred >= 0.5, tf.float32)
    neg_y_true = 1 - y_true_class
    neg_y_pred = 1 - y_pred_class
    fp = K.sum(neg_y_true * y_pred_class)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

def sensitivity(y_true, y_pred):
    '''
    Keras custom metric function that returns the sensitivity of each epoch in training.

    Parameters:
        y_true: tensor with the true values of the output variable
        y_pred: tensor with the predicted values of the output variable

    Returns:
        The sensitivity of the predictions
    '''

    y_true_class = tf.cast(y_true >= 0.5, tf.float32)
    y_pred_class = tf.cast(y_pred >= 0.5, tf.float32)
    neg_y_pred = 1 - y_pred_class
    tp = K.sum(y_true_class * y_pred_class)
    fn = K.sum(y_true_class * neg_y_pred)
    sensitivity = tp / (tp + fn + K.epsilon())
    return sensitivity

def create_lstm_clf_onehot(nodes1, dropout, lr, input_x_shape, two_layers = False, nodes2 = 32):
    '''
    Function that creates and compiles an LSTM model for use with classification and one hot encoding.

    Parameters:
        nodes1 (int): the number of nodes in the first layer of the network
        dropout (float): the amount of dropout between layers
        lr (float): the learning rate applied in the model optimizer
        input_x_shape (int): the number of one-hot sequences in the input
        two_layers (bool): whether the current model will have two layers or not
        nodes2 (int): the number of nodes in the second layer (if required)

    Returns:
        The LSTM model to be trained
    '''

    model = Sequential()
    if two_layers:
        model.add(Bidirectional(LSTM(nodes1, return_sequences = True, input_shape = (input_x_shape,4))))
    else:
        model.add(Bidirectional(LSTM(nodes1, input_shape = (input_x_shape,4))))
    model.add(Dropout(dropout))
    if two_layers:
        model.add(Bidirectional(LSTM(nodes2)))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation = "sigmoid"))
    print(model)
    opt = Adam(learning_rate=lr)
    model.compile(optimizer = opt, loss = "binary_crossentropy",
                  metrics = ["accuracy", specificity, sensitivity])
    return model

def create_lstm_clf_embedding(input_dim, output_dim, input_length, nodes1, dropout, lr, two_layers = False, nodes2 = 32):
    '''
    Function that creates and compiles an LSTM model for use with classification and embedding encoding

    Parameters:
        input_dim (int): the size of the vocabulary obtained from the tokenizer step
        output_dim (int): the size of the output dimension of the embedding
        input_length (int): length of the k-mer sequences
        nodes1 (int): the number of nodes in the first layer of the network
        dropout (float): the amount of dropout between layers
        lr (float): the learning rate applied in the model optimizer
        two_layers (bool): whether the current model will have two layers or not
        nodes2 (int): the number of nodes in the second layer (if required)

    Returns:
        The LSTM model to be trained
    '''

    model = Sequential()
    model.add(Embedding(input_dim = input_dim, output_dim = output_dim, input_length=input_length, name = "embedding_layer"))
    if two_layers:
        model.add(Bidirectional(LSTM(nodes1, return_sequences = True)))
    else:
        model.add(Bidirectional(LSTM(nodes1)))
    model.add(Dropout(dropout))
    if two_layers:
        model.add(Bidirectional(LSTM(nodes2)))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation = "sigmoid"))
    print(model)
    opt = Adam(learning_rate=lr)
    model.compile(optimizer = opt, loss = "binary_crossentropy",
                  metrics = ["accuracy", specificity, sensitivity])
    return model

def create_lstm_regr_onehot(nodes1, dropout, lr, input_x_shape):
    '''
    Function that creates and compiles an LSTM model for use with regression and one hot encoding.

    Parameters:
        nodes1 (int): the number of nodes in the first layer of the network
        dropout (float): the amount of dropout between layers
        lr (float): the learning rate applied in the model optimizer
        input_x_shape (int): the number of one-hot sequences in the input

    Returns:
        The LSTM model to be trained
    '''

    model = Sequential()
    model.add(Bidirectional(LSTM(nodes1, input_shape = (input_x_shape,4))))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation = "sigmoid"))
    print(model)
    opt = Adam(learning_rate=lr)
    model.compile(optimizer = opt, loss = mean_relative_error)
    return model

def create_lstm_regr_embedding(input_dim, output_dim, input_length, nodes1, dropout, lr, two_layers = False, nodes2 = 32):
    '''
    Function that creates and compiles an LSTM model for use with regression and embedding encoding

    Parameters:
        input_dim (int): the size of the vocabulary obtained from the tokenizer step
        output_dim (int): the size of the output dimension of the embedding
        input_length (int): length of the k-mer sequences
        nodes1 (int): the number of nodes in the first layer of the network
        dropout (float): the amount of dropout between layers
        lr (float): the learning rate applied in the model optimizer
        two_layers (bool): whether the current model will have two layers or not
        nodes2 (int): the number of nodes in the second layer (if required)

    Returns:
        The LSTM model to be trained
    '''

    model = Sequential()
    model.add(Embedding(input_dim = input_dim, output_dim = output_dim, input_length=input_length, name = "embedding_layer"))
    if two_layers:
        model.add(Bidirectional(LSTM(nodes1, return_sequences = True)))
    else:
        model.add(Bidirectional(LSTM(nodes1)))
    model.add(Dropout(dropout))
    if two_layers:
        model.add(Bidirectional(LSTM(nodes2)))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation = "sigmoid"))
    print(model)
    opt = Adam(learning_rate=lr)
    model.compile(optimizer = opt, loss = mean_relative_error)
    return model

def create_conv_clf_onehot(filters, kernel_size, dropout, lr, input_x_shape):
    '''
    Function that creates and compiles a convolutional model for use with classification and one hot encoding.

    Parameters:
        filters (int): the number of filters to use in training
        kernel_size (int): the size of the kernel to apply in training
        dropout (float): the amount of dropout between layers
        lr (float): the learning rate applied in the model optimizer
        input_x_shape (int): the number of one-hot sequences in the input

    Returns:
        The Convolutional model to be trained
    '''

    model = Sequential()
    model.add(Conv1D(filters = filters, kernel_size = kernel_size, activation = "relu", input_shape=(input_x_shape,4)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    #model.add(Dense(10, activation = "relu"))
    #model.add(Dropout(dropout))
    model.add(Dense(1, activation = "sigmoid"))
    opt = Adam(learning_rate=lr)
    model.compile(optimizer = opt, loss = "binary_crossentropy",
                      metrics = ["accuracy", specificity, sensitivity])
    return model

def create_conv_clf_embedding(input_dim, output_dim, input_length, filters, kernel_size, dropout, lr):
    '''
    Function that creates and compiles a convolutional model for use with classification and embedding encoding

    Parameters:
        input_dim (int): the size of the vocabulary obtained from the tokenizer step
        output_dim (int): the size of the output dimension of the embedding
        input_length (int): length of the k-mer sequences
        filters (int): the number of filters to use in training
        kernel_size (int): the size of the kernel to apply in training
        dropout (float): the amount of dropout between layers
        lr (float): the learning rate applied in the model optimizer

    Returns:
        The Convolutional model to be trained
    '''

    model = Sequential()
    model.add(Embedding(input_dim = input_dim, output_dim = output_dim, input_length=input_length, name = "embedding_layer"))
    model.add(Conv1D(filters = filters, kernel_size = kernel_size, activation = "relu"))
    #model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(1, activation = "sigmoid"))
    print(model)
    opt = Adam(learning_rate=lr)
    model.compile(optimizer = opt, loss = "binary_crossentropy",
                  metrics = ["accuracy", specificity, sensitivity])
    return model

def create_conv_regr_onehot(filters, kernel_size, dropout, lr, input_x_shape):
    '''
    Function that creates and compiles a convolutional model for use with regression and one hot encoding.

    Parameters:
        filters (int): the number of filters to use in training
        kernel_size (int): the size of the kernel to apply in training
        dropout (float): the amount of dropout between layers
        lr (float): the learning rate applied in the model optimizer
        input_x_shape (int): the number of one-hot sequences in the input

    Returns:
        The Convolutional model to be trained
    '''

    model = Sequential()
    model.add(Conv1D(filters = filters, kernel_size = kernel_size, activation = "relu", input_shape=(input_x_shape,4)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    #model.add(Dense(10, activation = "relu"))
    #model.add(Dropout(dropout))
    model.add(Dense(1, activation = "sigmoid"))
    opt = Adam(learning_rate=lr)
    model.compile(optimizer = opt, loss = mean_relative_error, metrics = ["mse"])
    return model

def create_conv_regr_embedding(input_dim, output_dim, input_length, filters, kernel_size, dropout, lr):
    '''
    Function that creates and compiles a convolutional model for use with regression and embedding encoding

    Parameters:
        input_dim (int): the size of the vocabulary obtained from the tokenizer step
        output_dim (int): the size of the output dimension of the embedding
        input_length (int): length of the k-mer sequences
        filters (int): the number of filters to use in training
        kernel_size (int): the size of the kernel to apply in training
        dropout (float): the amount of dropout between layers
        lr (float): the learning rate applied in the model optimizer

    Returns:
        The Convolutional model to be trained
    '''

    model = Sequential()
    model.add(Embedding(input_dim = input_dim, output_dim = output_dim, input_length=input_length, name = "embedding_layer"))
    model.add(Conv1D(filters = filters, kernel_size = kernel_size, activation = "relu", input_shape=(int(sys.argv[5]),4)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    #model.add(Dense(10, activation = "relu"))
    #model.add(Dropout(dropout))
    model.add(Dense(1, activation = "sigmoid"))
    opt = Adam(learning_rate=lr)
    model.compile(optimizer = opt, loss = mean_relative_error, metrics = ["mse"])
    return model


def create_plots_classification(history, folder, name):
    '''
    Function that creates and saves the plots of the accuracy, sensitivity, specificity and loss per epoch
    after the execution of the classification models

    Parameters:
        history: the object returned from the keras `fit` function that contains the results
                 of the model per epoch for each metric in its attribute `history`
        folder (str): the directory in which to save each plot
        name (str): the name of the current model, composed of the combination of the tuning parameters
    '''

    if not os.path.exists(folder):
        os.mkdir(folder)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='center right')
    plt.savefig(folder+'/Acc-'+ name + '.png')
    plt.clf()

    plt.plot(history.history['specificity'])
    plt.plot(history.history['val_specificity'])
    plt.plot(history.history['sensitivity'])
    plt.plot(history.history['val_sensitivity'])
    plt.title('Model sensitivity vs specificity')
    plt.ylabel('Spec vs Sens')
    plt.xlabel('epoch')
    plt.legend(['train_spec', 'val_spec', 'train_sens', 'val_sens'], loc='center right')
    plt.savefig(folder+ '/SpecVsSens-'+ name + '.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(folder+ '/Loss-'+ name + '.png')
    plt.clf()

def create_plots_regression(history, folder, name, r2_scores_val, r2_scores_train):
    '''
    Function that creates and saves the plots of the loss and R^2 in train and validation per epoch
    after the execution of the regression models

    Parameters:
        history: the object returned from the keras `fit` function that contains the results
                 of the model per epoch for each metric in its attribute `history`
        folder (str): the directory in which to save each plot
        name (str): the name of the current model, composed of the combination of the tuning parameters
    '''

    if not os.path.exists(folder):
        os.mkdir(folder)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(folder+ '/Loss-'+ name + '.png')
    plt.clf()

    plt.plot(range(1,51), r2_scores_train, marker='o', label = "train")
    plt.plot(range(1,51), r2_scores_val, marker='o', label = "val")
    plt.title('R2 scores train vs val')
    plt.legend(['r2_train', 'r2_val'], loc='upper left')
    plt.savefig(folder+ '/R2_scores-'+ name + '.png')
    plt.clf()

def random_undersample(X_train, Y_train, encoding_type = "one-hot"):
    '''
    Function that, given an input and output set, reduces the number of samples of the
    majority class randomly

    Parameters:
        X_train: set with the input sequences
        Y_train: array with the output variable
        encoding_type (str): string indicating the type of the encoding applied to the input

    Returns:
        The new input and outputs sets obtained after the undersampling
    '''

    undersampler = RandomUnderSampler()
    if encoding_type == "one-hot":
        X_under, Y_under = undersampler.fit_resample(X_train, Y_train)
        X_under = to_categorical(X_under)
    elif encoding_type == "embedding":
        X_under, Y_under = undersampler.fit_resample(np.array(X_train).reshape(-1,1), Y_train)
        X_under = np.squeeze(X_under)
    else:
        print("Wrong type of encoding selected")
        return
    # Reorder training set randomly
    Y_train_u = pd.Series(Y_under).sample(len(Y_under), random_state = 1)
    X_train_u = X_under[Y_train_u.index]
    return X_train_u, Y_train_u.values

class OutputObserver(tf.keras.callbacks.Callback):
    """
    Class thta builds a callback to observe the output of the network
    """

    def __init__(self, x_val, x_train, logs = {}):
        self.pred_val = []
        self.pred_train = []
        self.val_data = x_val
        self.train_data = x_train

    def on_epoch_end(self, epoch, logs={}):
        self.pred_val.append(self.model.predict(self.val_data))
        self.pred_train.append(self.model.predict(self.train_data))
