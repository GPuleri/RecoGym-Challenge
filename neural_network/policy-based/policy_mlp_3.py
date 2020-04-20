import numpy as np
from recogym import build_agent_init, Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)

from keras.models import Sequential
from keras.layers import Dense, InputLayer
import tensorflow as tf

import random

keras_nn_args = {
    'random_seed': np.random.randint(2 ** 31 - 1),
    "online_training_batch": 200,
    "training_factor": 0.01,
    "online_training": True,
    "epsilon": 0.01
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
                self.online_data = []
                self.actions = []
                self.online_rewards = np.empty(self.config.online_training_batch)
                self.online_count = 0
                self.class_weights = []

            def reset_online(self):
                self.online_data = []
                self.online_rewards = np.empty(self.config.online_training_batch, dtype=np.int)
                self.actions = []
                self.online_count = 0
                self.class_weights = []

            def train_online(self, features, action, reward):

                self.actions.append(action["a"])
                self.class_weights.append(action["ps"])
                self.online_rewards[self.online_count % self.config.online_training_batch] = reward
                self.online_data.append(features)
                self.online_count += 1

                if self.online_count == self.config.online_training_batch:
                    features = np.array(self.online_data).reshape((self.config.online_training_batch, P))
                    actions = np.array(self.actions, dtype=np.dtype(int))
                    clicks = np.sum(self.online_rewards, dtype=np.dtype(int))
                    actions_one_hot = np.zeros((clicks, P))
                    actions_one_hot[np.arange(clicks), actions] = 1

                    sample_weights = np.ones(len(features)) * [self.config.training_factor * i for i in range(1, len(features) + 1)]

                    print(f"Features:\n{features}")
                    print(f"Actions:\n{actions}")
                    print(f"Actions one hot:\n{actions_one_hot}")
                    print(f"Sample weights:\n{sample_weights}")
                    print(f"Class weights:\n{self.class_weights}")

                    self.model.fit(features, actions_one_hot, epochs=10, sample_weight=sample_weights, class_weight=1 / np.array(self.class_weights, dtype=np.dtype(int)), verbose=1)

                    self.reset_online()

            def act(self, observation, features):
                # X is a vector of organic counts
                predictions = self.model.predict(features)
                # Take the one you think is most likely to give a click
                if random.random() <= self.config.epsilon:
                    action = random.randrange(P)
                else:
                    action = np.argmax(predictions)
                # print(f"Action taken: {action}")
                ps_all = np.zeros(self.config.num_products)
                ps_all[action] = 1.0

                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': predictions[0][action],
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
        network.add(Dense(X.shape[1], activation=tf.nn.relu))
        network.add(Dense(A_one_hot.shape[1], activation=tf.nn.softmax))

        network.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        network.fit(X, A_one_hot, epochs=10, class_weight=1 / pss, verbose=1)        # Train a model

        return (
            KerasNNFeaturesProvider(self.config),
            KerasNNModel(self.config, network)
        )


class KerasNNAgent(ModelBasedAgent):
    """
    Scikit-Learn-based logistic regression Agent.
    """

    def __init__(self, config=Configuration(keras_nn_args)):
        self.previous_features = None
        self.previous_action = None
        self.model = None
        super(KerasNNAgent, self).__init__(
            config,
            KerasNNModelBuilder(config)
        )

    def act(self, observation, reward, done):
        if self.model is None:
            self.feature_provider, self.model = self.model_builder.build()

        if self.config.online_training and reward == 1:
            self.model.train_online(self.previous_features, self.previous_action, reward)

        self.feature_provider.observe(observation)

        features = self.feature_provider.features(observation)
        action_dict = self.model.act(observation, features)

        self.previous_features = features
        self.previous_action = action_dict

        return {
            "t": observation.context().time(),
            "u": observation.context().user(),
            **action_dict
        }



agent = build_agent_init('KerasNNAgent', KerasNNAgent, {**keras_nn_args})
