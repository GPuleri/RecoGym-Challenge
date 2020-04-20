import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy import sparse
from sklearn.neural_network import MLPClassifier


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

    n_epochs = 30000 # this should certainly be enough for convergence
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
    ia = num_features * np.arange(n+1)
    ja = np.ravel(num_features*actions[:, np.newaxis] + np.arange(num_features))
    return sparse.csr_matrix((data, ja, ia), shape=(n, num_features*num_actions), dtype=np.float32)

mlp_args = {
    'random_seed': np.random.randint(2 ** 31 - 1),
    'num_products': 10,
    'online_training_batch': 500,
    'online_training': True
}

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
                # self.online_matrix = sparse.dok_matrix(
                #     (self.config.online_training_batch, P * P)
                # )
                self.online_data = []
                self.online_rewards = np.empty(self.config.online_training_batch)
                self.online_count = 0
                self.first_time = True
                self.autoencoder = None

            def reset_online(self):
                # self.online_matrix = sparse.dok_matrix(
                #     (self.config.online_training_batch, P * P), dtype=np.int
                # )
                self.online_data = []
                self.online_rewards = np.empty(self.config.online_training_batch, dtype=np.int)
                self.online_count = 0

            def train_online(self, features, action, reward):
                """ This method does the online training (in batches) based on the reward we got for the previous
                action """

                # print("Inside online training")
                # print(f"Features:\n{features}\nFeatures shape: {features.shape}")

                # Update our online batch
                # for j in range(P * P):
                #     if action == j % P and features[0, j // P] != 0:
                #         self.online_matrix[(self.online_count, j)] = features[0, j // P]
                self.online_data.append(features)
                self.online_rewards[self.online_count] = reward
                self.online_count += 1


                # print(f"Online training batch:\n{self.online_matrix}\nOnline batch shape:{self.online_matrix.shape}")

                if self.online_count == self.config.online_training_batch:
                    # online_data = self.online_matrix.toarray()
                    print("Starting online training")
                    if self.first_time:
                        print("Training online autoencoder")
                        train_encoder = np.array(self.online_data).reshape((500, 10))
                        self.autoencoder = train_lin_encoder((train_encoder), embedding_size=2)
                        self.first_time = False

                    with torch.no_grad():
                        print("Encoding")
                        features_to_encode = np.array(self.online_data).reshape((500, 10))
                        data = torch.from_numpy(features_to_encode).float()
                        online_matrix = self.autoencoder.encode(data).numpy()
                    print("Encoding done")
                    online_matrix = arange_kronecker(online_matrix, self.online_rewards, self.config.num_products)
                    # print(f"Encoded online batch info: {online_matrix.shape}")
                    # exit(1)
                    # print(f"Online rewards:\n{self.online_rewards}")
                    print(f"Online training shape: {online_matrix.shape}")
                    self.model = self.model.partial_fit(online_matrix, self.online_rewards)

                    self.reset_online()

            def act(self, observation, features):
                print("Acting")

                A = np.arange(self.config.num_products)
                # print(f"Features in act: {features}")
                with torch.no_grad():
                    data = torch.from_numpy(features)
                    features = autoencoder.encode(data.float()).numpy()

                # print(f"Acting on: {features.numpy().shape}")
                kronecker = arange_kronecker(features, A, self.config.num_products)
                print(f"Predicting with shape: {kronecker.shape}")

                predictions = self.model.predict_proba(kronecker)[:, 1]
                action = np.argmax(predictions)

                # print(f"Data:\n{kronecker}")
                # print(f"Action: {action}")

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

        print(f"Training data:\n{features}\nShape: {features.shape}")

        X = features
        N = X.shape[0]
        P = X.shape[1]
        A = actions
        y = deltas

        # matrix = arange_kronecker(X, A, P)
        # print(f"Sparse features:\n{matrix}\nShape: {matrix.shape}")

        autoencoder = train_lin_encoder(X, embedding_size=2)
        with torch.no_grad():
            # print(f"Data before encoding:\n{X}\n{type(X)}")
            data = torch.from_numpy(X).float()
            training_matrix = autoencoder.encode(data).numpy()
            # print(f"Data after encoding:\n{training_matrix}\n{type(training_matrix)}")
            # exit(1)

        training_matrix = arange_kronecker(training_matrix, A, P)

        # print(f"Sparse encoded features:\n{training_matrix}\nShape: {training_matrix.shape}")

        # print(f"Starting MLP training with data: {type(training_matrix)}")
        # print(f"Organic views encoded shape: {training_matrix.shape}, Actions taken: {y.shape}")

        # print(f"Offline rewards:\n{y[:100]}")

        print(f"Training with shape:{training_matrix.shape}")

        model = MLPClassifier(activation="logistic", solver="adam", hidden_layer_sizes=(P,)).fit(training_matrix, y)

        return MLPFeaturesProvider(self.config), MLPModel(self.config, model)


class MLPAgent(ModelBasedAgent):
    def __init__(self, config=Configuration(mlp_args)):
        super(MLPAgent, self).__init__(config, MLPModelBuilder(config))
        self.previous_features = None
        self.previous_action = None

    def act(self, observation, reward, done):
        if self.model is None:
            self.feature_provider, self.model = self.model_builder.build()

        if self.config.online_training and reward is not None:
            self.model.train_online(self.previous_features, self.previous_action, reward)

        self.feature_provider.observe(observation)

        features = self.feature_provider.features(observation)
        action_dict = self.model.act(observation, features)

        self.previous_features = features
        self.previous_action = action_dict["a"]

        return {
            "t": observation.context().time(),
            "u": observation.context().user(),
            **action_dict
        }


agent = build_agent_init("MLPAgent", MLPAgent, {**mlp_args})






