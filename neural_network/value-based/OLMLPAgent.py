import numpy as np

from scipy import sparse
from sklearn.neural_network import MLPClassifier
import random


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


def preprocess(data):
    data = data + 1.0
    top = np.zeros_like(data)

    for row in range(top.shape[0]):
        cols = np.argwhere(data[row, :] == np.max(data[row, :]))
        top[row, cols] = 1.0

    return np.hstack((data, top))

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
                self.times_acted = 0

            def reset_online(self):
                self.online_data = []
                self.online_rewards = np.empty(self.config.online_training_batch, dtype=np.int)
                self.actions = []

            def train_online(self, features, action, reward):
                """ This method does the online training (in batches) based on the reward we got for the previous
                action """
                global_stats.append(reward)
                self.online_data.append(features)
                self.actions.append(action)
                self.online_rewards[self.online_count % self.config.online_training_batch] = reward
                self.online_count += 1

                if self.online_count % self.config.online_training_batch == 0:
                    online_matrix = arange_kronecker(preprocess(np.array(self.online_data).reshape(len(self.online_data), P)), np.array(self.actions), P)

                    self.model = self.model.partial_fit(online_matrix, self.online_rewards, classes=[0, 1])

                    self.reset_online()

            def act(self, observation, features):
                if self.times_acted and self.times_acted % 10000 == 0:
                    running_stats = global_stats[-10000:]
                    ctr_running = sum(running_stats) / 10000
                    ctr_cumulative = sum(global_stats) / len(global_stats)
                    print(
                        f'Times acted: {self.times_acted} | Running CTR: {ctr_running:.2%} ({len(running_stats)}) | Total CTR: {ctr_cumulative:.2%} ({len(global_stats)}) | Alpha: {self.model.get_params()["alpha"]}')

                self.times_acted += 1
                A = np.arange(P)
                if self.online_count < self.config.online_training_batch:
                    action = np.argmax(features)
                else:
                    if random.random() < self.config.epsilon:
                        action = random.randrange(P)
                    elif self.config.epsilon <= random.random() < self.config.fallback_threshold:
                        action = np.argmax(features)
                    else:

                        actions = np.arange(P)
                        features_preprocessed = preprocess(features)
                        kronecker = arange_kronecker(features_preprocessed, actions, P)

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

        X_preprocessed = preprocess(X)

        training_matrix = arange_kronecker(X_preprocessed, A, P)

        model = MLPClassifier(activation="logistic", solver="adam", hidden_layer_sizes=(P,)).fit(training_matrix, y)

        return MLPFeaturesProvider(self.config), MLPModel(self.config, model)


test_agent_args = {
    'random_seed': np.random.randint(2 ** 31 - 1),
    'num_products': 100,
    'online_training_batch': 50,
    'online_training': True,
    'epsilon': 0.01,
    'fallback_threshold': 0.04
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

global_stats = []
agent = build_agent_init("MLPAgent", TestAgent, {**test_agent_args})




