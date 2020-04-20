import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy import sparse
from sklearn.neural_network import MLPClassifier

import random

class LinearAutoEncoder(nn.Module):

    def __init__(self, num_products, embedding_size):
        super(LinearAutoEncoder, self).__init__()
        self.encoder = nn.Linear(num_products, embedding_size)
        self.decoder = nn.Linear(embedding_size, num_products)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


def train_lin_encoder(data, embedding_size):
    training_data = torch.from_numpy(data).float()
    model = LinearAutoEncoder(data.shape[1], embedding_size)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    early_stopper = EarlyStopper()

    n_epochs = 30000  # this should certainly be enough for convergence
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(training_data)
        loss = F.mse_loss(output, training_data)
        loss.backward()
        optimizer.step()
        early_stopper(loss.item(), model)
        if early_stopper.early_stop:
            break
    model.eval()
    return model


class EarlyStopper:

    def __init__(self, patience=100, delta=0):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.min_loss = np.Inf
        self.best_score = None

    def __call__(self, loss, model):

        score = -loss

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


from recogym import Configuration, build_agent_init
from recogym.agents.abstract import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
)
from recogym.agents import ViewsFeaturesProvider


def arange_kronecker(features, actions, num_actions):
    """ compute kronecker product of each feature with one-hot encoded action """
    n = actions.size
    num_features = features.shape[-1]
    data = np.broadcast_to(features, (n, num_features)).ravel()
    ia = num_features * np.arange(n + 1)
    ja = np.ravel(num_features * actions[:, np.newaxis] + np.arange(num_features))
    return sparse.csr_matrix((data, ja, ia), shape=(n, num_features * num_actions), dtype=np.float32)


class MLPModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(MLPModelBuilder, self).__init__(config)

    def build(self):
        class MLPFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super(MLPFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class MLPModel(Model):
            def __init__(self, config, model):
                super(MLPModel, self).__init__(config)
                self.model = model
                self.online_data = []
                self.actions = []
                self.online_rewards = np.empty(self.config.online_training_batch)
                self.online_count = 0
                self.first_time = True
                self.autoencoder = None

            def reset_online(self):
                self.online_data = []
                self.online_rewards = np.empty(self.config.online_training_batch, dtype=np.int)
                self.actions = []

            def train_online(self, features, action, reward):
                """ This method does the online training (in batches) based on the reward we got for the previous
                action """
                self.online_data.append(features)
                self.actions.append(action)
                self.online_rewards[self.online_count % self.config.online_training_batch] = reward
                self.online_count += 1

                if self.online_count % self.config.online_training_batch == 0:
                    # print("Starting online training")
                    if self.first_time:
                        self.autoencoder = train_lin_encoder(np.array(self.online_data).reshape((self.config.online_training_batch, P)), self.config.embedding_size)
                        self.first_time = False

                    with torch.no_grad():
                        features_to_encode = np.array(self.online_data).reshape((self.config.online_training_batch, P))
                        data = torch.from_numpy(features_to_encode).float()
                        online_matrix = self.autoencoder.encode(data).numpy()

                    online_matrix = arange_kronecker(online_matrix, np.array(self.actions), P)
                    # print(f"Online training shape: {online_matrix.shape}")

                    self.model = self.model.partial_fit(online_matrix, self.online_rewards, classes=[0, 1])

                    self.reset_online()

            def act(self, observation, features):

                A = np.arange(P)

                if self.online_count < self.config.online_training_batch:
                    action = np.argmax(features)
                else:
                    if random.random() < self.config.epsilon:
                        action = random.randrange(self.config.num_products)
                    elif self.config.epsilon <= random.random() < self.config.fallback_threshold:
                        action = np.argmax(features)
                    else:
                        with torch.no_grad():
                            data = torch.from_numpy(features)
                            features = self.autoencoder.encode(data.float()).numpy()

                        actions = np.arange(P)
                        kronecker = arange_kronecker(features, actions, P)

                        predictions = self.model.predict_proba(kronecker)[:, 1]
                        action = np.argmax(predictions)

                ps_all = np.zeros(self.config.num_products)
                ps_all[action] = 1.0

                return {
                    **super().act(observation, features),
                    **{
                        "a": action,
                        "ps": 1.0,
                        "ps-a": ps_all
                    }
                }

        features, actions, deltas, pss = self.train_data()

        X = features
        N = X.shape[0]
        P = X.shape[1]
        A = actions
        y = deltas

        model = MLPClassifier(activation="logistic", solver="adam", hidden_layer_sizes=(P,), learning_rate="constant", alpha=0.0001)

        return MLPFeaturesProvider(self.config), MLPModel(self.config, model)


test_agent_args = {
    'random_seed': np.random.randint(2 ** 31 - 1),
    'num_products': 100,
    'online_training_batch': 50,
    'online_training': True,
    'epsilon': 0.01,
    'fallback_threshold': 0.04,
    'embedding_size': 2
}


class TestAgent(ModelBasedAgent):
    def __init__(self, config=Configuration(test_agent_args)):
        super(TestAgent, self).__init__(config, MLPModelBuilder(config))
        self.previous_features = None
        self.previous_action = None
        self.feature_count = 0

    def act(self, observation, reward, done):
        if self.model is None:
            self.feature_provider, self.model = self.model_builder.build()

        self.feature_provider.observe(observation)
        features = self.feature_provider.features(observation)

        if self.config.online_training and reward is not None:
            self.model.train_online(self.previous_features, self.previous_action, reward)

        action_dict = self.model.act(observation, features)

        self.previous_features = features
        self.feature_count += 1
        self.previous_action = action_dict["a"]

        return {
            "t": observation.context().time(),
            "u": observation.context().user(),
            **action_dict
        }


agent = build_agent_init("MLPAgent", TestAgent, {**test_agent_args})





