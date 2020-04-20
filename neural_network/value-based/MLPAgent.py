import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
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

            def act(self, observation, features):

                A = np.arange(P)

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

        training_matrix = arange_kronecker(X, A, P)

        model = MLPClassifier(activation="logistic", solver="adam", hidden_layer_sizes=(P,)).fit(training_matrix, y)

        return MLPFeaturesProvider(self.config), MLPModel(self.config, model)


test_agent_args = {
    'random_seed': np.random.randint(2 ** 31 - 1),
    'num_products': 100,
}


class TestAgent(ModelBasedAgent):
    def __init__(self, config=Configuration(test_agent_args)):
        super(TestAgent, self).__init__(config, MLPModelBuilder(config))


global_stats = []
agent = build_agent_init("MLPAgent", TestAgent, {**test_agent_args})




