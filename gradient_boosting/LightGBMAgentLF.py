import random
import numpy as np
from scipy import sparse
from recogym import build_agent_init, Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider,
)

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

import torch
from torch import nn, optim

import pandas as pd

test_agent_args = {
    "random_seed": np.random.randint(2 ** 31 - 1),
    "num_products": 10,  # mandatory, will be overwritten
    "online_training": True,  # whether we should do online training
    "online_training_batch": 100,  # records to gather before starting an online training session
    "epsilon": 0.0,  # for epsilon-greedy online learning
    "retrain_auto_encoder_after": 300000,
    "latent_factors": 20,
    "auto_encoder_epochs": 1000
}


def arange_kronecker(features, actions, num_actions):
    """ compute kronecker product of each feature with one-hot encoded action """
    n = actions.size
    num_features = features.shape[-1]
    data = np.broadcast_to(features, (n, num_features)).ravel()
    ia = num_features * np.arange(n+1)
    ja = np.ravel(num_features*actions[:, np.newaxis] + np.arange(num_features))
    return sparse.csr_matrix((data, ja, ia), shape=(n, num_features*num_actions), dtype=np.float32)


def bin_data(data):
    #bins = np.array([0.0, 1.0, np.inf])
    #return np.digitize(data, bins)
    #return data.astype(np.bool)
    return data


class AutoEncoder(nn.Module):
    """
    Interesting loss functions for this problem:
    - torch.nn.CrossEntropyLoss
    - torch.nn.CosineEmbeddingLoss
    """
    def __init__(self, num_input_products, num_output_factors):
        midway = (num_input_products + num_output_factors) // 2
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_input_products, midway, bias=True),
            nn.ReLU(),
            nn.Linear(midway, num_output_factors, bias=True),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_output_factors, midway, bias=True),
            nn.ReLU(),
            nn.Linear(midway, num_input_products, bias=True),
            nn.Sigmoid(),
        )
        self.criterion = nn.MSELoss(reduction='mean')
        # self.criterion = nn.BCEWithLogitsLoss()
        # self.optimizer = optim.SGD(self.parameters(), lr=1e-4)
        # self.optimizer = optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)
        self.optimizer = optim.SGD(self.parameters(), lr=0.1, weight_decay=0.1)
        self.loss = None
        self.trainings_done = 0

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, x):
        y_pred = self(x)
        self.loss = self.criterion(y_pred, x)

        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.trainings_done += 1


class LightGBMAgentModel(Model):
    def __init__(self, config, x, a, y):
        super().__init__(config)
        self.model = None
        self.num_features = x.shape[1]
        self.times_acted = 0
        self.batch_x = x
        self.batch_a = a
        self.batch_y = y
        self.batch_size = 0
        self.full_x = None
        self.full_a = None
        self.full_y = None
        self.full_size = 0
        self.auto_encoder = None
        random.seed(2468)

    def reset_online(self):
        self.batch_x = np.empty((self.config.online_training_batch, self.num_features), dtype='float64')
        self.batch_a = np.empty(self.config.online_training_batch, dtype='int64')
        self.batch_y = np.empty(self.config.online_training_batch, dtype='int64')
        self.batch_size = 0

    def train_batch(self, features, actions, rewards):
        # Set training parameters
        lgb_params = {
            "objective": "binary",
            "num_leaves": 2,  # default 31
            "max_bin": 2,  # default 255
            "min_data_in_leaf": 0,  # default 20
            "boost_from_average": True,
            "verbosity": -1
        }
        """
        print("Training model with the following parameters:")
        for k, v in lgb_params.items():
            print(f'{k.ljust(30, " ")}{v}')
        """
        # Convert to sparse CSR
        sparse_data = arange_kronecker(features, actions, self.config.num_products)

        # Convert to LightGBM Dataset
        dataset = lgb.Dataset(sparse_data, label=rewards, free_raw_data=False)

        # Build and train booster
        self.model = lgb.train(lgb_params, dataset, num_boost_round=100)

    def add_batch_to_full(self):
        self.batch_x = bin_data(self.batch_x)

        # Add last batch to the full set
        if self.full_x is None and self.full_y is None and self.full_a is None:
            self.full_x = self.batch_x
            self.full_a = self.batch_a
            self.full_y = self.batch_y
        else:
            self.full_x = np.vstack([self.full_x, self.batch_x])
            self.full_a = np.concatenate([self.full_a, self.batch_a])
            self.full_y = np.concatenate([self.full_y, self.batch_y])

    def update_data(self, features, action, reward):
        # Store for statistics
        global_stats.append(reward)

        # Update our online batch
        self.batch_x[self.batch_size] = features[0]
        self.batch_a[self.batch_size] = action
        self.batch_y[self.batch_size] = reward
        self.batch_size += 1
        self.full_size += 1

        '''
        if self.full_size > 100000:
            # Encode our features
            with torch.no_grad():
                encoded_features = self.auto_encoder.encoder(torch.from_numpy(self.full_x).float()).numpy()
                do_grid_search(arange_kronecker(encoded_features, self.full_a, self.config.num_products), self.full_y)
            exit()
        '''
        # Check if batches are big enough to do a training session
        if self.batch_size == self.config.online_training_batch:
            # Process batch
            self.add_batch_to_full()

        if self.full_size % self.config.retrain_auto_encoder_after == 0:
            # Rebuild a new auto-encoder
            print(f'Retraining auto-encoder {self.full_size}')
            self.build_auto_encoder()

        if self.batch_size == self.config.online_training_batch:
            # Encode our features
            with torch.no_grad():
                encoded_features = self.auto_encoder.encoder(torch.from_numpy(self.full_x).float()).numpy()

            # Rebuild booster
            self.train_batch(encoded_features, self.full_a, self.full_y)
            self.reset_online()

    def build_auto_encoder(self):
        self.auto_encoder = AutoEncoder(self.config.num_products, self.config.latent_factors)
        early_stopper = EarlyStopper()

        for epoch in range(1000):
            self.auto_encoder.fit(torch.from_numpy(self.full_x).float())
            # early_stopper(self.auto_encoder.loss.item(), self.auto_encoder)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}: loss {self.auto_encoder.loss.item()}")
            # if early_stopper.early_stop:
            #     print(f"Early stopping at: {epoch}, Loss: {self.auto_encoder.loss.item()}")
            #     break

    def act(self, observation, features):
        # Show progress
        if self.times_acted and self.times_acted % 10000 == 0:
            running_stats = global_stats[-10000:]
            ctr_running = sum(running_stats) / 10000
            ctr_cumulative = sum(global_stats) / len(global_stats)
            print(
                f'Times acted: {self.times_acted} | Running CTR: {ctr_running:.2%} ({len(running_stats)}) | Total CTR: {ctr_cumulative:.2%} ({len(global_stats)})')

        self.times_acted += 1

        # Decide if we want to explore or exploit
        if random.random() < self.config.epsilon:
            # Explore: pick a random product
            action = random.randrange(self.config.num_products)
        else:
            # Exploit: take our best guess
            with torch.no_grad():
                encoded_features = self.auto_encoder.encoder(torch.from_numpy(bin_data(features)).float()).numpy()
            # encoded_features = features

            # We want a prediction for every action
            matrix = sparse.kron(sparse.eye(encoded_features.shape[1]), encoded_features, format="csr")

            # Get prediction for every action
            predictions = self.model.predict(matrix)

            # Take best action
            action = np.argmax(predictions)

        ps_all = np.zeros(self.config.num_products)
        ps_all[action] = 1.0

        return {
            **super().act(observation, features),
            **{"a": action, "ps": 1.0, "ps-a": ps_all},
        }


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
            #print(f"Patience counter: {self.counter}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class LightGBMAgentFeaturesProvider(ViewsFeaturesProvider):
    def __init__(self, config):
        super().__init__(config)
        print(f"Number of products: {self.config.num_products}")

    def features(self, observation):
        base_features = super().features(observation)
        return base_features.reshape(1, self.config.num_products)


class LightGBMAgentModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        # Get data
        offline_features, offline_actions, offline_rewards, offline_pss = self.train_data()

        # do_grid_search(training_matrix, y)
        model = LightGBMAgentModel(self.config, offline_features, offline_actions, offline_rewards)
        model.add_batch_to_full()

        # Train the auto-encoder
        model.build_auto_encoder()

        with torch.no_grad():
            encoded_features = model.auto_encoder.encoder(torch.from_numpy(bin_data(offline_features)).float()).numpy()

        # do_grid_search(arange_kronecker(encoded_features, offline_actions, self.config.num_products), offline_rewards)
        # exit()

        model.train_batch(encoded_features, offline_actions, offline_rewards)
        model.reset_online()

        return (
            LightGBMAgentFeaturesProvider(self.config),
            model,
        )


def do_grid_search(X, y):
    """ Does a grid search to find a good combination for the most important LightGBM hyper parameters given
    training set X and labels y """
    lgb_param_grid = {
        "boosting_type": ["gbdt"],
        "num_leaves": range(2, 21, 2),
        "min_data_in_leaf": range(0, 21, 2),
        "max_bin": range(2, 30, 2),
    }
    lgb_classifier = lgb.LGBMClassifier(
        objective="binary", n_estimators=100, verbosity=-1
    )
    gcv = GridSearchCV(lgb_classifier, lgb_param_grid, cv=3, n_jobs=-1, verbose=1)
    gcv.fit(X, y)
    print(f"\nBest estimator:\n{gcv.best_estimator_}")
    print("Best params:")
    for k, v in gcv.best_params_.items():
        print(f"{str(k).ljust(25)}\tBest: {v}")
    exit()


class TestAgent(ModelBasedAgent):
    """
    LightGBM Gradient Boosting Agent.
    """

    def __init__(self, config=Configuration(test_agent_args)):
        self.previous_features = None
        self.previous_action = None
        super().__init__(config, LightGBMAgentModelBuilder(config))

    def act(self, observation, reward, done):
        """ We're overloading this method so we can do online training on the previous observation whenever we get
        a new one """

        # Build model first if not yet done
        if self.model is None:
            assert self.feature_provider is None
            self.feature_provider, self.model = self.model_builder.build()

        # Now that we have the reward, train based on previous features and reward we got for our action
        if self.config.online_training and reward is not None:
            self.model.update_data(
                self.previous_features, self.previous_action, reward
            )

        # Update the feature provider with this new observation
        self.feature_provider.observe(observation)

        # Get the new features
        features = self.feature_provider.features(observation)
        a_ps_psa_dict = self.model.act(observation, features)

        # Update previous feature set for next online learning session
        self.previous_features = features
        self.previous_action = a_ps_psa_dict["a"]

        return {
            "t": observation.context().time(),
            "u": observation.context().user(),
            **a_ps_psa_dict,
        }


global_stats = []
agent = build_agent_init("LightGBMAgent", TestAgent, {**test_agent_args})

if __name__ == "__main__":
    import gym
    from recogym import env_1_args
    from recogym.bench_agents import test_agent

    num_products = 10
    num_offline_users = 20
    num_online_users = 200

    agent = TestAgent(
        Configuration({
            "random_seed": np.random.randint(2 ** 31 - 1),
            "num_products": num_products,
            "fallback_threshold": 0.00,
            "online_training": True,
            "online_training_batch": 100,
            "epsilon": 0.01,
            "latent_factors": 2
        })
    )

    env_1_args["random_seed"] = 71
    env_1_args["num_products"] = num_products
    env = gym.make("reco-gym-v1")
    env.init_gym(env_1_args)

    print(
        test_agent(
            env, agent, num_offline_users=num_offline_users, num_online_users=num_online_users
        )
    )

    """
    correct_predictions = {}
    for edge in np.arange(0.0, 1.0, 0.01):
        correct_predictions[edge] = 0

    for prob, click in global_stats:
        for edge in np.arange(0.0, 1.0, 0.01):
            if prob < edge + 0.01:
                correct_predictions[edge] += click
                break

    hfreq, hedges = np.histogram([stat[0] for stat in global_stats], np.arange(0, 1.1, 0.01))
    print("\n*** PROBABILITIES HISTOGRAM ***")
    print(f"Total of {len(global_stats)} samples")
    print(f"Number of zero probabilities: {len([x for x in hfreq if x == 0])}")
    for freq, edge in zip(hfreq, hedges):
        print(f"{round(edge, 2)} : {freq} (Share: {freq/len(global_stats):.2%} | Correct predictions: {correct_predictions[edge]} or {correct_predictions[edge]/freq:.2%})")
    """
