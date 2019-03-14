from keras.models import Sequential, load_model
from keras.layers.core import Dense, Reshape, Flatten, Dropout
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.activations import softmax
from keras.losses import binary_crossentropy
from keras.optimizers import RMSprop, Adam, SGD
from keras.initializers import Orthogonal
import keras.backend as K
import numpy as np
import pandas as pd


def load_relations(relations_path):
    relations = pd.read_hdf(relations_path).values
    return relations


def load_labels(labels_path):
    labels = pd.read_hdf(labels_path).values
    return labels


# Metrics for each individual class
def make_metrics(class_id, one_hot):
    def precision(y_true, y_pred):
        if not one_hot:
            return K.mean(K.equal(y_true[:, class_id], K.round(y_pred)[:, class_id]))
        else:
            return K.mean(K.equal(K.round(y_true)[:, class_id], K.cast(K.equal(K.argmax(y_pred, 1), class_id), "float32")))

    def tp(y_true, y_pred):
        if not one_hot:
            true_positives = K.sum(y_true[:, class_id] * y_pred[:, class_id])
        else:
            true_positives = K.sum(y_true[:, class_id] * K.cast(K.equal(K.argmax(y_pred, 1), class_id), "float32"))

        return true_positives

    def pp(y_true, y_pred):
        possible_positives = K.sum(y_true[:, class_id])
        return possible_positives

    def recall(y_true, y_pred):
        return tp(y_true, y_pred) / (pp(y_true, y_pred) + K.epsilon())

    def fscore(y_true, y_pred, beta=1):
        prec = precision(y_true, y_pred)
        rec = recall(y_true, y_pred)
        beta_sq = beta ** 2
        fbeta_score = (1 + beta_sq) * (prec * rec) / \
            (beta_sq * prec + rec + K.epsilon())
        return fbeta_score

    def binary_entropy(y_true, y_pred):
        return binary_crossentropy(y_true[:, class_id], y_pred[:, class_id])

    return precision, recall, fscore, binary_entropy

class RelationClassifier:
    def __init__(self):
        self.model = None

    def new(self, input_dim, relation_count, one_hot=False, filters=32, max_filters=128,
            subtract_embeddings=False, dropout=False, learn_rate=0.001, optimizer="rmsprop",
            kernel_size=3, lr_decay=0):
        """Creates a new model expecting an input vector of shape `(-, input_dim)`
        making predictions for `relation_count` classes"""
        self.model = RelationClassifier._get_model(input_dim, relation_count, one_hot,
                                                   filters, max_filters, subtract_embeddings,
                                                   dropout, learn_rate, optimizer, kernel_size,
                                                   lr_decay)

    def save(self, path):
        """Saves the model to `path`"""
        self.model.compile(optimizer=RMSprop(), loss="binary_crossentropy",
                           metrics=["binary_accuracy"])
        self.model.save(path)

    def load(self, path):
        """Loads the model from `path`"""
        self.model = load_model(path)

        # Check if the output is one-hot by looking if the activation of
        # the last layer is softmax
        one_hot = self.model.layers[-1].activation == softmax

        all_metrics = ["categorical_accuracy" if one_hot else "binary_accuracy"]
        for class_id in range(self.model.layers[-1].output_shape[1]):
            all_metrics += make_metrics(class_id, one_hot)

        self.model.compile(optimizer=RMSprop(), loss="binary_crossentropy",
                           metrics=all_metrics)

    def train(self, relations, labels, batch_size=256, validation_split=0.1, epochs=10,
              val_data=None, verbose=1):
        """Trains the model given `relations` of shape `(-, input_dim)` and
        `labels` of shape `(-, relation_count)`"""
        if self.model is None:
            raise ValueError(
                "A model must be created (.new) or loaded (.load) first.")

        self.model.fit(relations, labels, batch_size=batch_size,
                       validation_split=validation_split if val_data is None else 0,
                       epochs=epochs, shuffle=True, validation_data=val_data,
                       verbose=verbose)

    def replace_last_layer(self, output_dim, one_hot):
        """Replaces the last layer of the model and adds a new output layer with size `output_dim`"""
        if self.model is None:
            raise ValueError(
                "A model must be created (.new) or loaded (.load) first.")

        # Remove last layer
        self.model.layers.pop()
        self.model.outputs = [self.model.layers[-1].output]
        self.model.layers[-1].outbound_nodes = []

        # Make layers untrainable
        for layer in self.model.layers:
            layer.trainable = False

        # Add new output layer
        self.model.add(Dense(output_dim, activation="softmax" if one_hot else "sigmoid",
                       kernel_initializer="orthogonal"))

        # Restore metrics
        all_metrics = ["categorical_accuracy" if one_hot else "binary_accuracy"]
        for class_id in range(self.model.layers[-1].output_shape[1]):
            all_metrics += make_metrics(class_id, one_hot)

        # Compile model
        self.model.compile(optimizer=RMSprop(), loss="binary_crossentropy",
                           metrics=all_metrics)

    def predict(self, relations):
        """Returns the class predictions for `relations`"""
        if self.model is None:
            raise ValueError(
                "A model must be created (.new) or loaded (.load) first.")
        return self.model.predict(relations)

    @staticmethod
    def _get_model(input_dim, output_dim, one_hot, filters=32, max_filters=128, subtract_embeddings=False, dropout=False,
                   learn_rate=0.001, optimizer="rmsprop", kernel_size=3, lr_decay=0):
        if optimizer == "rmsprop":
            optimizer = RMSprop(lr=learn_rate, decay=lr_decay)
        elif optimizer == "adam":
            optimizer = Adam(lr=learn_rate, decay=lr_decay)
        elif optimizer == "sgd":
            optimizer = SGD(lr=learn_rate, momentum=0.9, decay=lr_decay)
        else:
            raise ValueError("Invalid argument optimizer")

        initializer = Orthogonal(np.sqrt(2))

        model = Sequential()

        dropout = dropout if isinstance(dropout, float) else (0.5 if dropout else 0)

        if not subtract_embeddings:
            # Reshape to 2D, do 2D conv and reshape to 1D
            model.add(Reshape(input_shape=(input_dim,),
                              target_shape=(2, input_dim // 2, 1)))

            model.add(Conv2D(filters=filters, kernel_size=[2, 7], strides=[1, 2],
                            padding="VALID", activation="relu", kernel_initializer=initializer))

            model.add(BatchNormalization())
        else:
            # Reshape to add single channel
            model.add(Reshape(input_shape=(input_dim,),
                              target_shape=(input_dim, 1)))
            model.add(Conv1D(filters=filters, kernel_size=7, strides=2,
                            padding="VALID", activation="relu", kernel_initializer=initializer))
            model.add(BatchNormalization())

        model.add(Reshape((-1, filters)))
        
        if dropout:
            model.add(Dropout(dropout))

        # Half conv with kernel size 3 until not greater than 3
        while model.layers[-1].output_shape[1] > 3:
            filters = min(filters * 2, max_filters)

            model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=2,
                             padding="SAME", activation="relu", kernel_initializer=initializer))

            model.add(BatchNormalization())
            if dropout:
                model.add(Dropout(dropout))

        # Conv valid so output is 1 if necessary
        if model.layers[-1].output_shape[1] != 1:
            filters = min(filters * 2, max_filters)
            model.add(Conv1D(filters=filters, kernel_size=(model.layers[-1].output_shape[1],),
                             padding="VALID", activation="relu", kernel_initializer=initializer))
            model.add(BatchNormalization())
            if dropout:
                model.add(Dropout(dropout))

        # Dense sigmoid output
        model.add(Flatten())
        #model.add(Dropout(0.5))
        model.add(Dense(output_dim, activation="softmax" if one_hot else "sigmoid",
                        kernel_initializer="orthogonal"))

        print(model.summary())

        all_metrics = ["categorical_accuracy" if one_hot else "binary_accuracy"]
        for class_id in range(output_dim):
            all_metrics += make_metrics(class_id, one_hot)

        model.compile(optimizer=optimizer, loss="categorical_crossentropy" if one_hot else "binary_crossentropy",
                      metrics=all_metrics)

        return model
