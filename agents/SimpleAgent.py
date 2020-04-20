import numpy as np
from recogym import build_agent_init, Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)
from sklearn.svm import SVC

model_sklearn_args = {
    'random_seed': np.random.randint(2 ** 31 - 1),
}


class SimpleAgentModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(SimpleAgentModelBuilder, self).__init__(config)

    def build(self):
        class SimpleAgentFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super(SimpleAgentFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class SimpleAgentModel(Model):
            def __init__(self, config, model):
                super(SimpleAgentModel, self).__init__(config)
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

        print('Shape of training_matrix: {}'.format(training_matrix.shape))

        np.savetxt('test.out', training_matrix, delimiter=',', fmt='%1f')

        '''
            kernel : string, optional (default=’rbf’)
            Specifies the kernel type to be used in the algorithm.
            It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
            If none is given, ‘rbf’ will be used.
            If a callable is given it is used to pre-compute the kernel matrix from data matrices;
            that matrix should be an array of shape (n_samples, n_samples).

            coef0 : float, optional (default=0.0)
            Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

            tol : float, optional (default=1e-3)
            Tolerance for stopping criterion.

            class_weight : {dict, ‘balanced’}, optional
            Set the parameter C of class i to class_weight[i]*C for SVC. 
            If not given, all classes are supposed to have weight one. 
            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to 
            class frequencies in the input data as n_samples / (n_classes * np.bincount(y))

            max_iter : int, optional (default=-1)
            Hard limit on iterations within solver, or -1 for no limit.

            random_state : int, RandomState instance or None, optional (default=None)
            The seed of the pseudo random number generator used when shuffling the data for probability estimates. 
            If int, random_state is the seed used by the random number generator; 
            If RandomState instance, random_state is the random number generator; 
            If None, the random number generator is the RandomState instance used by np.random.

        '''

        # Train a model
        model = SVC(C=1.0,
                    kernel='sigmoid',
                    coef0=0.0,
                    probability=True,
                    tol=1e-3,
                    class_weight=None,
                    max_iter=-1,
                    random_state=42) \
            .fit(training_matrix, y)

        print('Model: {}'.format(model))

        return (
            SimpleAgentFeaturesProvider(self.config),
            SimpleAgentModel(self.config, model)
        )


class SimpleAgent(ModelBasedAgent):
    """
    Scikit-Learn-based SVM Agent.
    """

    def __init__(self, config=Configuration(model_sklearn_args)):
        super(SimpleAgent, self).__init__(
            config,
            SimpleAgentModelBuilder(config)
        )


agent = build_agent_init('SimpleAgent',
                         SimpleAgent,
                         {**model_sklearn_args})

