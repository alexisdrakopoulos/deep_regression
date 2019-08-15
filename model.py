from sys import argv
import os
from datetime import datetime
from os.path import isfile
from warnings import warn

# assign a single GPU if so chosen by user in args
try:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # assign GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = argv[1]
except IndexError:
    pass

# Keras/machine learning imports
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras import losses
from keras import metrics
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, Callback

# from keras.backend.tensorflow_backend import set_session
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# #config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)

# import keras
# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# keras.backend.tensorflow_backend.set_session(sess)

# # if multi-gpu is detected
# if "," in argv[1]:
#   print(f"Using GPUs: {argv[1]}")
#   from keras.utils import multi_gpu_model


def experimental_log(model_iteration,
                     model_type,
                     activation,
                     dropout,
                     batchnorm,
                     batchnorm_order,
                     batch_size,
                     epochs,
                     learning_rate,
                     sparse_data):
    """
    Writes individual log files of each experiment as .txt
    Inputs are self-explanatory and pre-called inside the cnn_regressor
    """

    today = datetime.now().strftime("%d/%m/%Y %H:%M")

    file_name = "data/logs/model_logs/log_" + str(model_iteration)

    if os.path.exists(file_name):
        warn("File exists, appending current system second to name", UserWarning)
        file_name += "_" + datetime.now().second + ".txt"
    else:
        file_name += ".txt"

    with open(file_name, 'w+') as file:
        file.write(f"Date: {today} \n")
        file.write(f"Iteration: {model_iteration} \n")
        file.write(f"Model Type: {model_type} \n")
        file.write(f"Activation Function: {activation} \n")
        file.write(f"Batch size: {batch_size} \n")
        file.write(f"Dropout: {dropout} \n")
        file.write(f"Batchnorm: {batchnorm} \n")
        file.write(f"Batchnorm order: {batchnorm_order} \n")
        file.write(f"Epochs: {epochs} \n")
        file.write(f"Learning Rate: {learning_rate} \n")
        file.write(f"Sparse Data: {sparse_data}")


def create_directories():

    directories = ["data/models", "data/logs/model_logs", "data/logs/training_logs",
                   "data/predictions"]

    for el in directories:
        if not os.path.exists(el):
            print(f"Building directory: {el}")
            os.mkdir(el)


def load_data(model_type):
    """
    Imports data from numpy files saved to disk
    """

    data = np.load("data/ising/ising_data.npz")

    return data["training_data"], data["training_labels"], data["validation_data"], data["validation_labels"]


def convert_to_categories(values, bins):
    """
    Converts the continuous labels into evenly spread bins,
    uses the numpy linspace functionality.
    Can be slow, does around 500,000 iters/second.

    Inputs:
        values - numpy array of continuous labels (1D)
        bins - int value of number of bins to pass to numpy linspace

    Outputs:
        cats - the new 1D numpy array of categories from 0 to n in original order
    """

    vals = np.linspace(1, 4, bins)
    cats = np.zeros(len(values))

    print(f"Converting all labels to {bins} categories.")
    print(f"Categories are: {vals}")
    for idi, i in enumerate(values):
        for idj, j in enumerate(vals):
            if i < j:
                cats[idi] = idj
                break

    return cats


def cnn_regressor(model_iteration=1,
                  model_type="regression",
                  activation="relu",
                  dropout=False,
                  batchnorm=False,
                  batchnorm_order="before",
                  save_model=True,
                  batch_size=128,
                  epochs=25,
                  learning_rate=0.00002,
                  sparse_data=False,
                  predictions=False,
                  logging_loss=True):
    """
    CNN Regression model using vggnet16 and dropout layers

    Inputs:
        model_iteration - The experiment run to keep track of models
        model_type - regression or classification
        activation - relu or lrelu depending on activation to use in model
        dropout - Bool on whether to use dropout in between each set of layers
        batchnorm - Bool on whether to use batchnorm in between each set of layers
        batchnorm_order - before or after activations
        save_model - Bool on whether to save final model to disk, best model is saved
        batch_size - The batch size to use
        epochs - The number of epochs to run through, auto-stop procedure exists
        predictions - Bool whether to run the predictions

    Outputs:
        best model is saved to disk, and if save_model is True final model also saved
        a csv file containing training data is also saved to disk.
    """

    # Check args
    if batchnorm_order not in ("before", "after"):
        raise ValueError("batch_norm arg needs to be 'before' or 'after'.")

    if model_type not in ("regression", "classification"):
        raise ValueError("model_type arg needs to be 'regression' or 'classification'")

    if activation not in ("relu", "lrelu"):
        raise ValueError("Sorry at the moment only relu and lrelu is implemented.")

    # Convert args back to bools and ints from argparse command line
    model_iteration = int(model_iteration)
    dropout = eval(dropout.capitalize())
    batchnorm = eval(batchnorm.capitalize())
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    sparse_data = eval(sparse_data.capitalize())

    # Logging data to txt file
    experimental_log(model_iteration,
                     model_type,
                     activation,
                     dropout,
                     batchnorm,
                     batchnorm_order,
                     batch_size,
                     epochs,
                     learning_rate,
                     sparse_data)

    # Setting file names
    if model_type == "regression":
        add_type = "reg"
    elif model_type == "classification":
        add_type = "clas"

    training_file = "data/logs/training_logs/training_" + str(model_iteration) + ".csv"
    final_model = "data/models/fmodel_" + str(model_iteration) + "_" + add_type + ".h5"
    best_model = "data/models/bmodel_" + str(model_iteration) + "_" + add_type + ".h5"

    # input image dimensions
    img_rows, img_cols = 128, 128
    input_shape = (img_rows, img_cols, 1)

    # Import the data
    print("Starting Model")
    print("Importing Data and Labels")
    training_data, training_labels, val_data, val_labels = load_data(model_type)

    if sparse_data:
        print("Sparse data, only 50,000 training samples being used")
        training_data = training_data[:50_000]
        training_labels = training_labels[:50_000]
        val_data = val_data[:10_000]
        val_labels = val_labels[:10_000]

    # Reshape the data for channels last
    training_data = training_data.reshape(training_data.shape[0], img_rows, img_cols, 1)
    val_data = val_data.reshape(val_data.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)


    # Encode labels if classification is being used
    if model_type == "classification":
        num_classes = 10
        val_labels = convert_to_categories(val_labels, num_classes)
        training_labels = convert_to_categories(training_labels, num_classes)
        val_labels = to_categorical(val_labels.astype(int), num_classes)
        training_labels = to_categorical(training_labels.astype(int), num_classes)


    def activation_function(activation):
        """
        custom build activation function to return relu or lrelu

        Inputs:
            activation - string, either relu or lrelu
        Outputs:
            returns an activation to add to keras model
        """

        if activation == "relu":
            return Activation("relu")

        elif activation == "lrelu":
            return LeakyReLU()


    def final_activation(model,
                         pooling=True,
                         dropout=False,
                         batchnorm=False,
                         batchnorm_order="before"):
        """
        To shorten the conditionals required for the final activation
        at the end of each set of convolutional operations.

        Inputs:
            model - the keras model to add the layers to
            pooling - bool as to whether to pool, set False if dense
        Outputs:
            adds layers such as batchnormalization, pooling and dropout
        """

        if batchnorm and batchnorm_order == "before":
            model.add(BatchNormalization())

        model.add(activation_function(activation=activation))

        if batchnorm and batchnorm_order == "after":
            model.add(BatchNormalization())

        if pooling:
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        if dropout:
            model.add(Dropout(0.25))


    print("Running CNN")
    print('Training data shape:', training_data.shape)
    print(training_data.shape[0], 'training samples')
    print(val_data.shape[0], 'test samples')

    model = Sequential()

    # First set of Convolutions
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
    model.add(activation_function(activation=activation))
    model.add(Conv2D(64, (3, 3), padding='same'))
    final_activation(model=model, pooling=True,
                     dropout=dropout, batchnorm=batchnorm,
                     batchnorm_order=batchnorm_order)

    # Second set of Convolutions
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(activation_function(activation=activation))
    model.add(Conv2D(128, (3, 3), padding='same'))
    final_activation(model=model, pooling=True,
                     dropout=dropout, batchnorm=batchnorm,
                     batchnorm_order=batchnorm_order)

    # Third set of Convolutions
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(activation_function(activation=activation))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(activation_function(activation=activation))
    model.add(Conv2D(256, (3, 3), padding='same'))
    final_activation(model=model, pooling=True,
                     dropout=dropout, batchnorm=batchnorm,
                     batchnorm_order=batchnorm_order)

    # Fourth set of Convolutions
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(activation_function(activation=activation))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(activation_function(activation=activation))
    model.add(Conv2D(512, (3, 3), padding='same'))
    final_activation(model=model, pooling=True,
                     dropout=dropout, batchnorm=batchnorm,
                     batchnorm_order=batchnorm_order)

    # Fifth set of Convolutions
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(activation_function(activation=activation))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(activation_function(activation=activation))
    model.add(Conv2D(512, (3, 3), padding='same'))
    final_activation(model=model, pooling=True,
                     dropout=dropout, batchnorm=batchnorm,
                     batchnorm_order=batchnorm_order)

    # Fully connected Layers
    model.add(Flatten())
    model.add(Dense(4096))
    final_activation(model=model, pooling=False,
                     dropout=dropout, batchnorm=batchnorm,
                     batchnorm_order=batchnorm_order)
    model.add(Dense(4096))
    final_activation(model=model, pooling=False,
                     dropout=False, batchnorm=False,
                     batchnorm_order=batchnorm_order)

    # Linear regression if regression is chosen
    if model_type == "regression":
        model.add(Dense(1000))
        model.add(Dense(1))

    # softmax including batchnorm/dropout if selected
    elif model_type == "classification":
        model.add(Dense(1000))
        final_activation(model=model, pooling=False,
                         dropout=False, batchnorm=False)
        model.add(Dense(num_classes, activation="softmax"))

    optimizer = Adam(lr=learning_rate)

    # Losses and compiler
    if model_type == "regression":
        loss = losses.mean_squared_error
        metric = ["mse", tf.keras.losses.mape]
    elif model_type == "classification":
        loss = losses.categorical_crossentropy
        metric = ["accuracy"]


    # Check multigpu:
    # if "," in argv[1]:
    #   model = multi_gpu_model(model, gpus=len(argv[1].split(",")))

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metric)

    # Call Backs for early stopping, logging, checkpoints and lr reduction
    csv_logger = CSVLogger(training_file)
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.0002,
                               patience=12,
                               verbose=1,
                               mode='min')
    mcp_save = ModelCheckpoint(best_model,
                               save_best_only=True,
                               monitor='val_loss',
                               mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.25,
                                       patience=4,
                                       verbose=1,
                                       min_delta=0.0002,
                                       mode='min',
                                       min_lr=1e-7)

    # Custom callback for per-batch metrics and loss
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            if model_type == "classification":
                self.accuracy = []
            elif model_type == "regression":
                self.mape = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            if model_type == "classification":
                self.accuracy.append(logs.get('acc'))
            elif model_type == "regression":
                self.mape.append(logs.get("mean_absolute_percentage_error"))

    loss_history = LossHistory()

    # callbacks for the keras model
    callbacks = [loss_history,
                 early_stop,
                 mcp_save,
                 reduce_lr_loss,
                 csv_logger]

    history = model.fit(training_data, training_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        #verbose=2,
                        validation_data=(val_data, val_labels),
                        callbacks=callbacks)

    if logging_loss:
        print("Saving per batch losses")
        if model_type == "classification":
            loss_histories = np.vstack((np.array(loss_history.losses),
                                        np.array(loss_history.accuracy)))
        elif model_type == "regression":
            loss_histories = np.vstack((np.array(loss_history.losses),
                                        np.array(loss_history.mape)))

        loss_histories_name = "data/logs/loss_histories/" + str(model_iteration) + "loss_histories"
        np.save(loss_histories_name, loss_histories)


    print("Running evaluation on validation data")
    score = model.evaluate(val_data, val_labels, verbose=0)
    print(score)

    if save_model:
        print("Saving model")
        model.save(final_model)

    if predictions:
        print("Running predictions on test data")
        test_data = np.load("data/ising/test_data.npz")
        test_data = test_data["data"]
        predictions = model.predict(test_data)
        prediction_name = "predictions_" + str(model_iteration) + "_" + str(model_type) + ".npy"
        np.save(prediction_name, predictions)


print("Training Network")
# predictions = cnn_regressor(training_data=train_x,
#                             training_labels=train_y,
#                             val_data=val_x,
#                             val_labels=val_y,
#                             model_iteration=1,
#                             model_type="regression",
#                             activation="relu",
#                             dropout=False,
#                             batchnorm=False,
#                             epochs=25,
#                             batch_size=32)

# Quick hack to get sparse data working
# try:
#     sparse_data = eval(argv[10])
# except:
#     sparse_data = False

create_directories()
predictions = cnn_regressor(model_iteration=argv[2],
                            model_type=argv[3],
                            activation=argv[4],
                            dropout=argv[5],
                            batchnorm=argv[6],
                            batchnorm_order=argv[7],
                            batch_size=argv[8],
                            learning_rate=argv[9],
                            sparse_data=argv[10])
