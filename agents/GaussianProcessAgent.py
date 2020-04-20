import numpy as np
from recogym import build_agent_init, Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)

from sklearn.gaussian_process import GaussianProcessClassifier

model_sklearn_args = {
    # 'num_products': 100,
    'random_seed': np.random.randint(2 ** 31 - 1),
}


class GaussianProcessAgentModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(GaussianProcessAgentModelBuilder, self).__init__(config)

    def build(self):
        class GaussianProcessAgentFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super(GaussianProcessAgentFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class GaussianProcessAgentModel(Model):
            def __init__(self, config, model):
                super(GaussianProcessAgentModel, self).__init__(config)
                self.model = model

            def act(self, observation, features):
                # X is a vector of organic counts
                X = features

                # We want a prediction for every action
                A = np.eye(P)

                # Get prediction for every action
                predictions = model.predict_proba(np.kron(X, A))[:, 1]

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
        X = features  # NxP vector where rows are users, columns are counts of organic views
        N = X.shape[0]  # Number of bandit feedback samples
        P = X.shape[1]  # Number of items
        A = actions  # Vector of length N - indicating the action that was taken
        y = deltas  # Vector of length N - indicating whether a click occurred

        # Explicitly build one-hot matrix for actions
        A_one_hot = np.zeros((N, P))
        A_one_hot[np.arange(N), A] = 1

        # TODO - this really doesn't scale, maybe sparsify?
        training_matrix = []

        for x, a in zip(X, A_one_hot):
            training_matrix.append(np.kron(x, a))

        training_matrix = np.asarray(training_matrix)

        np.savetxt('test.out', training_matrix, delimiter=',', fmt='%1f')

        '''
            random_state : int, RandomState instance or None, optional (default: None)
                The generator used to initialize the centers. If int, 
                random_state is the seed used by the random number generator; 
                If RandomState instance, random_state is the random number generator; 
                If None, the random number generator is the RandomState instance used by np.random.
                
            multi_class : string, default
                Specifies how multi-class classification problems are handled. 
                Supported are “one_vs_rest” and “one_vs_one”. In “one_vs_rest”, 
                one binary Gaussian process classifier is fitted for each class, 
                which is trained to separate this class from the rest. 
                In “one_vs_one”, one binary Gaussian process classifier is fitted for each pair of classes, 
                which is trained to separate these two classes. 
                The predictions of these binary predictors are combined into multi-class predictions. 
                Note that “one_vs_one” does not support predicting probability estimates.
                Default is "one_vs_rest"

        '''

        # Train a model
        model = GaussianProcessClassifier().fit(training_matrix, y)

        return (
            GaussianProcessAgentFeaturesProvider(self.config),
            GaussianProcessAgentModel(self.config, model)
        )


class GaussianProcessAgent(ModelBasedAgent):
    """
    Scikit-Learn-based SVM Agent.
    """

    def __init__(self, config=Configuration(model_sklearn_args)):
        super(GaussianProcessAgent, self).__init__(
            config,
            GaussianProcessAgentModelBuilder(config)
        )


agent = build_agent_init('GaussianProcessAgent',
                         GaussianProcessAgent,
                         {**model_sklearn_args})

