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
                predictions = self.model.predict_proba(features)

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

        # Explicitly mask - drop non-clicks
        mask = y == 1
        X = X[mask]
        A = A[mask]
        y = y[mask]
        pss = pss[mask]
        
        n_clicks = np.sum(deltas)

        # Explicitly build one-hot matrix for actions
        A_one_hot = np.zeros((n_clicks,P))
        A_one_hot[np.arange(n_clicks), A] = 1

        # Train a model
        model = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial').fit(X, A, sample_weight = 1 / pss)

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
