import numpy as np
from recogym import build_agent_init, Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

logistic_regression_sklearn_args = {
    'random_seed': np.random.randint(2 ** 31 - 1),
}

class LogisticRegression_SKLearnModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(LogisticRegression_SKLearnModelBuilder, self).__init__(config)

    def build(self):
        class LogisticRegression_SKLearnFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super(LogisticRegression_SKLearnFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class LogisticRegression_SKLearnModel(Model):
            def __init__(self, config, model):
                super(LogisticRegression_SKLearnModel, self).__init__(config)
                self.model = model

            def act(self, observation, features):
                # X is a vector of organic counts
                X = features

                # We want a prediction for every action
                A = np.eye(P)
            
                # Get prediction for every action
                predictions = model.predict_proba(np.kron(X,A))[:,1]

                # Take the one you think is most likely to give a click
                action = np.argmax(predictions)
                ps_all = np.zeros(self.config.num_products)
                ps_all[action] = 1.0

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
        y = deltas      # Vector of length N - indicating whether a click occurred

        # Explicitly build one-hot matrix for actions
        A_one_hot = np.zeros((N,P))
        A_one_hot[np.arange(N), A] = 1

        # TODO - this really doesn't scale, maybe sparsify?
        training_matrix = []
        for x,a in zip(X,A_one_hot):
            training_matrix.append(np.kron(x,a))
        training_matrix = np.asarray(training_matrix)

        # Train a model
        model = LogisticRegression(solver = 'lbfgs').fit(training_matrix,y)

        return (
            LogisticRegression_SKLearnFeaturesProvider(self.config),
            LogisticRegression_SKLearnModel(self.config, model)
        )

class LogisticRegression_SKLearnAgent(ModelBasedAgent):
    """
    Scikit-Learn-based logistic regression Agent.
    """
    def __init__(self, config = Configuration(logistic_regression_sklearn_args)):
        super(LogisticRegression_SKLearnAgent, self).__init__(
            config,
            LogisticRegression_SKLearnModelBuilder(config)
        )

agent = build_agent_init('LogisticRegression_SKLearnAgent', LogisticRegression_SKLearnAgent, {**logistic_regression_sklearn_args})
