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

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.DEBUG)

decision_tree_args = {
    'random_seed': np.random.randint(2 ** 31 - 1),
    'num_products': 1000
}

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

                logging.info("Currently acting...")

                # kronecker = np.kron(X, A)
                start = time.time()
                kronecker = kron(features.astype("uint8"), eye(P, dtype="uint8"), format="csr")
                end = time.time()

                logging.info("Kronecker calculation took %s" % (end - start))

                logging.info("Internals of the kron matrix:")

                for i in kronecker:
                    logging.debug(i)

                predictions = model.predict_proba(kronecker)[:, 1]

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

        # print("Features\n", features)

        X = features
        N = X.shape[0]
        P = X.shape[1]
        A = actions
        y = deltas

        # for row, i in zip(X, range(5)):
        #     print(row)

        A_to_one_hot = np.zeros((N, P))
        A_to_one_hot[np.arange(N), A] = 1

        # training_matrix = []
        # for x, a in zip(X, A_to_one_hot):
        #     training_matrix.append(np.kron(x, a))
        # training_matrix = np.asarray(training_matrix)

        # training_matrix = csr_matrix(training_matrix)


        training_matrix = dok_matrix((N, P * P), dtype="uint8")
        for i in range(N):
            if i % (N // 10) == 0:
                print(f"{i * 100 / (N // 10 * 10)} %", end='\r')
            for j in range(P * P):
                if A[i] == j % P and X[i][j // P] != 0:
                    training_matrix[(i, j)] = X[i][j // P]

        training_matrix = training_matrix.tocsr()

        # for row in training_matrix:
        #     print(row)

        # print("Dense Training matrix\n", training_matrix)
        # print("A value from dense matrix ", training_matrix[0][55])


        # print("Sparse Training matrix\n", training_matrix)

        model = DecisionTreeClassifier().fit(training_matrix, y)

        # print("# of classes: ", model.n_classes_)
        # print("Classes: ", model.classes_)
        # exit(1)

        return (DecisionTreeFeaturesProvider(self.config), DecisionTreeModel(self.config, model))

class DecisionTreeAgent(ModelBasedAgent):
    def __init__(self, config=Configuration(decision_tree_args)):
        super(DecisionTreeAgent, self).__init__(config, DecisionTreeModelBuilder(config))

agent = build_agent_init("DecisionTreeAgent", DecisionTreeAgent, {**decision_tree_args})

