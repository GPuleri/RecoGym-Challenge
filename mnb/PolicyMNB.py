import numpy as np
from recogym import build_agent_init, Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)
from sklearn.naive_bayes import MultinomialNB

mnb_args = {
    'random_seed': np.random.randint(2 ** 31 - 1),
    "online_training_batch": 2000,
    "training_factor": 0.01,
    "online_training": True,
    "epsilon": 0.01
}

class MNBBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(MNBBuilder, self).__init__(config)

    def build(self):
        class MNBFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super(MNBFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class MNBModel(Model):
            def __init__(self, config, model):
                super(MNBModel, self).__init__(config)
                self.model = model
                self.online_data = []
                self.actions = []
                self.online_rewards = np.empty(self.config.online_training_batch)
                self.online_count = 0
                self.class_weights = []

            def reset_online(self):
                self.online_data = []
                self.online_rewards = np.empty(self.config.online_training_batch, dtype=np.int)
                self.actions = []
                self.online_count = 0
                self.class_weights = []

            def train_online(self, features, action, reward):

                self.actions.append(action["a"])
                self.class_weights.append(action["ps"])
                self.online_rewards[self.online_count % self.config.online_training_batch] = reward
                self.online_data.append(features)
                self.online_count += 1


                if self.online_count == self.config.online_training_batch:
                # if TRUE:
                    features = np.array(self.online_data).reshape((self.config.online_training_batch, P))
                    # print(features.shape)
                    actions = np.array(self.actions, dtype=np.dtype(int))
                    clicks = np.sum(self.online_rewards, dtype=np.dtype(int))
                    # actions_one_hot = np.zeros(clicks)
                    # actions_one_hot[np.arange(clicks), actions] = 1
                    # print(actions_one_hot.shape)
                    # print(actions_one_hot)
                    sample_weights = np.ones(len(features)) * [self.config.training_factor * i for i in range(1, len(features) + 1)]

                    self.model.partial_fit(features, actions)

                    self.reset_online()

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
        X = features  # NxP vector where rows are users, columns are counts of organic views
        N = X.shape[0]  # Number of bandit feedback samples
        P = X.shape[1]  # Number of items
        A = actions  # Vector of length N - indicating the action that was taken
        y = deltas  # Vector of length N - indicating whether a click occurred

        # Explicitly mask - drop non-clicks
        mask = y == 1
        X = X[mask]
        A = A[mask]
        y = y[mask]
        pss = pss[mask]

        n_clicks = np.sum(deltas)

        # Explicitly build one-hot matrix for actions
        A_one_hot = np.zeros((n_clicks, P))
        A_one_hot[np.arange(n_clicks), A] = 1
        # print(X.shape)
        # print(A.shape)
        # Train a model
        model = MultinomialNB(alpha=0.1).fit(X, A)

        return (
            MNBFeaturesProvider(self.config),
            MNBModel(self.config, model)
        )


class MNBAgent(ModelBasedAgent):
    """
    Scikit-Learn-based logistic regression Agent.
    """

    def __init__(self, config=Configuration(mnb_args)):
        self.previous_features = None
        self.previous_action = None
        self.model = None
        super(MNBAgent, self).__init__(
            config,
            MNBBuilder(config)
        )

    def act(self, observation, reward, done):
        if self.model is None:
            self.feature_provider, self.model = self.model_builder.build()

        if self.config.online_training and reward == 1:
            self.model.train_online(self.previous_features, self.previous_action, reward)

        self.feature_provider.observe(observation)

        features = self.feature_provider.features(observation)
        action_dict = self.model.act(observation, features)

        self.previous_features = features
        self.previous_action = action_dict

        return {
            "t": observation.context().time(),
            "u": observation.context().user(),
            **action_dict
        }


agent = build_agent_init('MNBAgent', MNBAgent,
                         {**mnb_args})