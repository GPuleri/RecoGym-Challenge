import numpy as np
import random

from recogym import build_agent_init, Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)

from scipy import sparse
from sklearn.linear_model import LogisticRegression

test_agent_args = {
    'num_products': 10,
    'random_seed': np.random.randint(2 ** 31 - 1),
    'num_latent_factors': 20,
    "fallback_threshold": 0.00,
    "online_training": True,
    "online_training_batch": 100,
    'epsilon': 0.05,
    'training_weight_factor': 2,
}


def arange_kronecker(features, actions, num_actions):
    """ compute kronecker product of each feature with one-hot encoded action """
    n = actions.size
    num_features = features.shape[-1]
    data = np.broadcast_to(features, (n, num_features)).ravel()
    ia = num_features * np.arange(n + 1)
    ja = np.ravel(num_features * actions[:, np.newaxis] + np.arange(num_features))
    return sparse.csr_matrix((data, ja, ia), shape=(n, num_features * num_actions), dtype=np.float32)


def preprocess(data):
    # 0s are not liked
    data = data + 1.0
    # Distinctly identify top-viewed item too
    top = np.zeros_like(data)
    for row in range(top.shape[0]):
        cols = np.argwhere(data[row, :] == np.max(data[row, :]))
        top[row, cols] = 1.0
    # Return hstack
    return np.hstack((data, top))


class LogisticRegression_SKLearnFeaturesProvider(ViewsFeaturesProvider):
    def __init__(self, config):
        super().__init__(config)

    def features(self, observation):
        base_features = super().features(observation)
        return base_features.reshape(1, self.config.num_products)


class LogisticRegression_SKLearnModel(Model):
    def __init__(self, config, model, x, a, y):
        super().__init__(config)
        self.model = model

        self.num_features = x.shape[1]

        self.batch_x = x.astype('uint16')
        self.batch_a = a.astype('uint16')
        self.batch_y = y.astype('bool')

        self.batch_size = 0
        self.num_iterations = 0

    def act(self, observation, features):

        # Egreedy - explore  vs exploit
        if random.random() >= self.config.epsilon:
            # We want a prediction for every action
            matrix = sparse.kron(sparse.eye(features.shape[1]), preprocess(features), format="csr")

            # Get prediction for every action
            predictions = self.model.predict_proba(matrix)[:, 1]

            action = np.argmax(predictions)
        else:
            # Pick a random item
            action = np.random.choice(self.config.num_products)

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

    def reset_batches(self):
        self.batch_x = np.empty((self.config.online_training_batch, self.num_features), dtype='uint16')
        self.batch_a = np.empty(self.config.online_training_batch, dtype='uint16')
        self.batch_y = np.empty(self.config.online_training_batch, dtype='bool')

        self.batch_size = 0

    def update_data(self, features, action, reward):
        # single record

        # store record
        self.batch_x[self.batch_size] = features[0]
        self.batch_a[self.batch_size] = action
        self.batch_y[self.batch_size] = reward

        self.num_iterations += 1
        self.batch_size += 1

        if self.batch_size == self.config.online_training_batch:

            if any(self.batch_y):
                # Partial fit of model in batches
                kronecker_product = arange_kronecker(preprocess(self.batch_x), self.batch_a, self.config.num_products)

                # num_rows = kronecker_product.shape[0]
                # weights = np.ones(num_rows) * [self.config.training_weight_factor * (i + 1) for i in range(num_rows)]

                # Partial fit on batch of data
                self.model.fit(
                    X=kronecker_product,
                    y=self.batch_y
                    # ,sample_weight=weights
                )

            self.reset_batches()


class LogisticRegression_SKLearnModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(LogisticRegression_SKLearnModelBuilder, self).__init__(config)

    def build(self):
        # Get data
        offline_features, offline_actions, offline_rewards, offline_pss = self.train_data()

        kronecker_product = arange_kronecker(preprocess(offline_features), offline_actions, self.config.num_products)

        # Train model
        """
        solver : str, {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, optional (default=’liblinear’).
            Algorithm to use in the optimization problem.

            For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
            For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; 
            ‘liblinear’ is limited to one-versus-rest schemes.
            ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
            ‘liblinear’ and ‘saga’ also handle L1 penalty
            ‘saga’ also supports ‘elasticnet’ penalty
            ‘liblinear’ does not handle no penalty

        max_iter : int, optional (default=100)
            Maximum number of iterations taken for the solvers to converge.

        warm_start : bool, optional (default=False)
            When set to True, reuse the solution of the previous call to fit as initialization, 
            otherwise, just erase the previous solution. Useless for liblinear solver. See the Glossary.

            New in version 0.17: warm_start to support lbfgs, newton-cg, sag, saga solvers



        .fit
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples. If not provided, then each sample is given unit weight.

            New in version 0.17: sample_weight support to LogisticRegression.
        """

        model = LogisticRegression(
            solver='lbfgs',
            max_iter=4000,
            warm_start=True
        ).fit(
            X=kronecker_product,
            y=offline_rewards
        )

        return (
            LogisticRegression_SKLearnFeaturesProvider(self.config),
            LogisticRegression_SKLearnModel(self.config, model, offline_features, offline_actions, offline_rewards)
        )


class TestAgent(ModelBasedAgent):
    """
    Logistic regression agent.
    """

    def __init__(self, config=Configuration(test_agent_args)):
        self.previous_features = None
        self.previous_action = None
        super().__init__(config, LogisticRegression_SKLearnModelBuilder(config))

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


agent = build_agent_init('LogisticRegression_SKLearnAgent_Egreedy', TestAgent, test_agent_args)
