import numpy as np
from scipy import sparse
from recogym import build_agent_init, Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightgbm as lgb


test_agent_args = {
    "random_seed": np.random.randint(2 ** 31 - 1),
    "num_products": 10,
    "fallback_threshold": 0.00,
    "online_training": True,
    "online_training_batch": 100,
    "refit_decay_rate": 0.9
}

def arange_kronecker(features, actions, num_actions):
    """ compute kronecker product of each feature with one-hot encoded action """
    n = actions.size
    num_features = features.shape[-1]
    data = np.broadcast_to(features, (n, num_features)).ravel()
    ia = num_features * np.arange(n+1)
    ja = np.ravel(num_features*actions[:, np.newaxis] + np.arange(num_features))
    return sparse.csr_matrix((data, ja, ia), shape=(n, num_features*num_actions), dtype=np.float32)

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
        if epoch % 100 == 0:
            print(f"{epoch}: {loss.item()}")
        early_stopper(loss.item(), model)
        if early_stopper.early_stop:
            print(f"Early stopping at: {epoch}, Loss: {loss.item()}")
            break
    model.eval()
    return model


class EarlyStopper:

    def __init__(self, patience=20, delta=0):
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
            print(f"Patience counter: {self.counter}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class LightGBMAgentModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        class LightGBMAgentFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super().__init__(config)
                print(f"Number of products: {self.config.num_products}")

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class LightGBMAgentModel(Model):
            def __init__(self, config, model):
                super().__init__(config)
                self.model = model
                self.times_acted = 0
                self.online_matrix = sparse.dok_matrix(
                    (self.config.online_training_batch, P * P)
                )
                self.online_rewards = np.empty(self.config.online_training_batch)
                self.online_count = 0

            def act(self, observation, features):
                # Show progress
                self.times_acted += 1
                if self.times_acted % 1000 == 0:
                    print(f"{self.times_acted} acts", end="\r")

                A = np.arange(self.config.num_products)

                with torch.no_grad():
                    data = torch.from_numpy(features).float()
                    features = autoencoder.encode(data)

                # We want a prediction for every action
                # matrix = sparse.kron(features, sparse.eye(P), format="csr")
                matrix = arange_kronecker(features.numpy(), A, self.config.num_products)
                # Get prediction for every action
                predictions = self.model.predict(matrix)

                # Check if we need to fallback to most viewed organic
                if np.max(predictions) >= self.config.fallback_threshold:
                    # Take the one you think is most likely to give a click
                    action = np.argmax(predictions)
                    ps_all = np.zeros(self.config.num_products)
                    ps_all[action] = 1.0
                else:
                    # Fallback to
                    action = np.argmax(features)
                    ps_all = np.zeros(self.config.num_products)
                    ps_all[action] = 1.0

                return {
                    **super().act(observation, features),
                    **{"a": action, "ps": 1.0, "ps-a": ps_all},
                }

        # Get data
        features, actions, deltas, pss = self.train_data()

        # NxP vector where rows are users, columns are counts of organic views
        X = features

        N = X.shape[0]  # Number of bandit feedback samples
        P = X.shape[1]  # Number of items
        A = actions  # Vector of length N - indicating the action that was taken
        y = deltas  # Vector of length N - indicating whether a click occurred

        # Initialize a sparse DOK matrix and fill it up. This has the same result as a row by row kronecker product
        # of the features by A (one-hot-encoded)
        print(f"Building training matrix ({N},{P * P}) ...")

        autoencoder = train_lin_encoder(X, embedding_size=2)

        with torch.no_grad():
            data = torch.from_numpy(X).float()
            training_matrix = autoencoder.encode(data)

        training_matrix = arange_kronecker(training_matrix.numpy(), A, P)

        # Now that the matrix is built, switch from DOK to CSR
        # print("\nSwitching training matrix to CSR format")
        # training_matrix = training_matrix.tocsr()

        lgb_params = {
            "objective": "binary",
            "num_leaves": 2,  # default 31
            "max_bin": 2,  # default 255
            "min_data_in_leaf": 3,  # default 20
            "boost_from_average": True,
            "verbosity": -1
        }

        print("Training model with the following parameters:")
        for k, v in lgb_params.items():
            print(f'{k.ljust(30, " ")}{v}')

        training_dataset = lgb.Dataset(training_matrix, label=y, free_raw_data=False)
        model = None
        model = lgb.train(
            lgb_params,
            training_dataset,
            num_boost_round=100,
            init_model=model,
            keep_training_booster=True,
        )

        return (
            LightGBMAgentFeaturesProvider(self.config),
            LightGBMAgentModel(self.config, model)
        )

class TestAgent(ModelBasedAgent):
    """
    LightGBM Gradient Boosting Agent.
    """

    def __init__(self, config=Configuration(test_agent_args)):
        super().__init__(config, LightGBMAgentModelBuilder(config))


agent = build_agent_init("LightGBMAgent", TestAgent, {**test_agent_args})

