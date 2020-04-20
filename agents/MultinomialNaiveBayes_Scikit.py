import numpy as np
from random import choices
from recogym import build_agent_init, Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)

from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm

multinomial_naive_bayes_args = {
    'random_seed': np.random.randint(2 ** 31 - 1),
}


class MultinomialNaiveBayesModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(MultinomialNaiveBayesModelBuilder, self).__init__(config)

    def build(self):
        class MultinomialNaiveBayesFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super(MultinomialNaiveBayesFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class MultinomialNaiveBayesModel(Model):
            def __init__(self, config, model):
                super(MultinomialNaiveBayesModel, self).__init__(config)
                self.model = model

            def act(self, observation, features):
                # X is a vector of organic counts
                X = features

                # We want a prediction for every action
                A = np.eye(P)

                # Get prediction for every action
                predictions = model.predict_proba(np.kron(X,A))[:,1]

                action = choices(range(self.config.num_products), weights=predictions).pop()
                ps_all = predictions
                ps_all[action] = predictions[action]

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
        X = features    # NxP vector where rows are users, columns are counts of organic views
        N = X.shape[0]  # Number of bandit feedback samples
        P = X.shape[1]  # Number of items
        A = actions     # Vector of length N - indicating the action that was taken
        y = deltas      # Vector of length N - indicating whether a click occurred, TARGET

        # Explicitly build one-hot matrix for actions
        A_one_hot = np.zeros((N,P))
        A_one_hot[np.arange(N), A] = 1


        # TODO - this really doesn't scale, maybe sparsify?
        training_matrix = []
        for x, a in zip(X, A_one_hot):
            training_matrix.append(np.kron(x,a))
        training_matrix = np.asarray(training_matrix)

        # Train a model
        model = MultinomialNB().fit(training_matrix, y)

        return (
            MultinomialNaiveBayesFeaturesProvider(self.config),
            MultinomialNaiveBayesModel(self.config, model)
        )

class MultinomialNaiveBayesAgent(ModelBasedAgent):
    """
    Scikit-Learn-based logistic regression Agent.
    """
    def __init__(self, config = Configuration(multinomial_naive_bayes_args)):
        super(MultinomialNaiveBayesAgent, self).__init__(
            config,
            MultinomialNaiveBayesModelBuilder(config)
        )

agent = build_agent_init('MultinomialNaiveBayesAgent', MultinomialNaiveBayesAgent, {**multinomial_naive_bayes_args})
