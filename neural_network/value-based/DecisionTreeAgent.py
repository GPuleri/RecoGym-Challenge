import numpy as np
from recogym import build_agent_init, Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import dok_matrix, eye, kron
import logging
import time
from scipy import sparse

decision_tree_args = {
    'random_seed': np.random.randint(2 ** 31 - 1),
    'num_products': 1000
}

def arange_kronecker(features, actions, num_actions):
    """ compute kronecker product of each feature with one-hot encoded action """
    n = actions.size
    num_features = features.shape[-1]
    # print(f"Actions:{n}, Features:{num_features}")
    # print(f"Features shape: {features.shape}")
    data = np.broadcast_to(features, (n, num_features)).ravel()
    ia = num_features * np.arange(n + 1)
    ja = np.ravel(num_features * actions[:, np.newaxis] + np.arange(num_features))
    return sparse.csr_matrix((data, ja, ia), shape=(n, num_features * num_actions), dtype=np.float32)

class DecisionTreeModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(DecisionTreeModelBuilder, self).__init__(config)

    def build(self):
        class DecisionTreeFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super(DecisionTreeFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class DecisionTreeModel(Model):
            def __init__(self, config, model):
                super(DecisionTreeModel, self).__init__(config)
                self.model = model

            def act(self, observation, features):
                X = features
                A = np.eye(P)

                actions = np.arange(P)
                kronecker = arange_kronecker(features, actions, P)

                for i in kronecker:
                    logging.debug(i)

                predictions = self.model.predict_proba(kronecker)[:, 1]

                logging.debug("Decision Tree Predictions %s" % predictions)

                action = np.argmax(predictions)

                logging.debug("Action taken: %s" % action)

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

        # for row, i in zip(X, range(5)):
        #     print(row)

        A_to_one_hot = np.zeros((N, P))
        A_to_one_hot[np.arange(N), A] = 1

        training_matrix = dok_matrix((N, P * P), dtype="uint8")
        for i in range(N):
            if i % (N // 10) == 0:
                print(f"{i * 100 / (N // 10 * 10)} %", end='\r')
            for j in range(P * P):
                if A[i] == j % P and X[i][j // P] != 0:
                    training_matrix[(i, j)] = X[i][j // P]

        training_matrix = training_matrix.tocsr()

        model = DecisionTreeClassifier().fit(training_matrix, y)

        return (DecisionTreeFeaturesProvider(self.config), DecisionTreeModel(self.config, model))

class DecisionTreeAgent(ModelBasedAgent):
    def __init__(self, config=Configuration(decision_tree_args)):
        super(DecisionTreeAgent, self).__init__(config, DecisionTreeModelBuilder(config))

agent = build_agent_init("DecisionTreeAgent", DecisionTreeAgent, {**decision_tree_args})

if __name__ == '__main__':
    import recogym, gym
    from recogym import env_1_args
    from recogym.bench_agents import test_agent


    env_1_args['random_seed'] = 42
    env_1_args["num_products"] = 5
    env = gym.make("reco-gym-v1")
    env.init_gym(env_1_args)

    tree_agent = DecisionTreeAgent(Configuration(env_1_args))

    print(test_agent(env, tree_agent))




