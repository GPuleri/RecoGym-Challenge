import numpy as np
from recogym import build_agent_init, Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, InputLayer, SimpleRNN, Reshape
from keras.utils import to_categorical
import tensorflow as tf

keras_nn_args = {
    'random_seed': np.random.randint(2 ** 31 - 1),
}


class KerasNNModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(KerasNNModelBuilder, self).__init__(config)

    def build(self):
        class KerasNNFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super(KerasNNFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class KerasNNModel(Model):
            def __init__(self, config, model):
                super(KerasNNModel, self).__init__(config)
                self.model = model

            def act(self, observation, features):
                # X is a vector of organic counts
                predictions = self.model.predict(features)
                # Take the one you think is most likely to give a click
                action = np.argmax(predictions)
                ps_all = np.zeros(self.config.num_products)
                ps_all[action] = 1.0

                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': 1.0,
                        'ps-a': ps_all,
                    },
                }

        # Get data
        features, actions, deltas, pss = self.train_data()

        # Extract data properly
        X = features  # NxP vector where rows are users, columns are counts of organic views
        N = X.shape[0]  # Number of bandit feedback samples
        P = X.shape[1]  # Number of items
        A = actions  # Vector of length N - indicating the action that was taken
        y = deltas  # Vector of length N - indicating whether a click occurred
        # Explicitly mask - drop non-clicks
        mask = y == 1
        X = X[mask]
        A = A[mask]
        y = y[mask]
        pss = pss[mask]

        n_clicks = np.sum(deltas)

        # Explicitly build one-hot matrix for actions
        A_one_hot = np.zeros((n_clicks, P))
        A_one_hot[np.arange(n_clicks), A] = 1

        network = Sequential()
        network.add(InputLayer(input_shape=X[0].shape))
        network.add(Dense(X.shape[1], activation=tf.nn.relu))
        network.add(Reshape((1, X.shape[1])))
        network.add(SimpleRNN(X.shape[1]))
        network.add(Dense(A_one_hot.shape[1], activation=tf.nn.softmax))

        network.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        network.fit(X, A_one_hot, epochs=10, class_weight=1 / pss, verbose=1)

        return (
            KerasNNFeaturesProvider(self.config),
            KerasNNModel(self.config, network)
        )


class KerasNNAgent(ModelBasedAgent):
    """
    Scikit-Learn-based logistic regression Agent.
    """

    def __init__(self, config=Configuration(keras_nn_args)):
        super(KerasNNAgent, self).__init__(
            config,
            KerasNNModelBuilder(config)
        )


agent = build_agent_init('KerasNNAgent', KerasNNAgent, {**keras_nn_args})
