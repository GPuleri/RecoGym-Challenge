import sys
import random
import numpy as np
from scipy import sparse
from recogym import build_agent_init, Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider,
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate

test_agent_args = {
    "random_seed": np.random.randint(2 ** 31 - 1),
    "num_products": 10,
    "fallback_threshold": 0.05,
    "online_training": True,
    "online_training_batch": 25,
}


class CrossConfirmationMNBAgentModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        class CrossConfirmationMNBAgentFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super().__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class CrossConfirmationMNBAgentModel(Model):
            def __init__(self, config, model, full_x, full_y):
                super().__init__(config)
                self.model = model
                self.times_acted = 0
                self.online_matrix = sparse.dok_matrix(
                    (self.config.online_training_batch, P * P)
                )
                self.online_rewards = np.empty(self.config.online_training_batch)
                self.online_count = 0
                self.full_x = full_x
                self.full_y = full_y

            def reset_online(self):
                self.online_matrix = sparse.dok_matrix(
                    (self.config.online_training_batch, P * P)
                )
                self.online_rewards = np.empty(self.config.online_training_batch)
                self.online_count = 0

            def train_online(self, features, action, reward):
                """ This method does the online training (in batches) based on the reward we got for the previous
                action """

                # Update our online batch
                for j in range(P):
                    self.online_matrix[(self.online_count, action * P + j)] = features[0, j]

                self.online_rewards[self.online_count] = reward
                self.online_count += 1
                # Check if we want to do a training session
                if self.online_count == self.config.online_training_batch:

                    # Reached batch size, completely retrain our booster on cumulative data
                    online_dataset = self.online_matrix.tocsr()
                    online_label = self.online_rewards
                    self.model[0] = model[0].partial_fit(online_dataset, online_label)
                    self.model[1] = model[1].partial_fit(online_dataset, online_label)
                    self.model[2] = model[2].partial_fit(online_dataset, online_label)

                    # Reset our online batch data
                    self.reset_online()

            def act(self, observation, features):
                # Show progress
                self.times_acted += 1

                # We want a prediction for every action
                matrix = sparse.kron(sparse.eye(P), features, format="csr")

                # Get prediction for every action
                predictions_0 = self.model[0].predict_proba(matrix)[:, 1]
                predictions_1 = self.model[1].predict_proba(matrix)[:, 1]
                predictions_2 = self.model[2].predict_proba(matrix)[:, 1]

                predictions = np.array([predictions_0, predictions_1, predictions_2])
                predictions = np.mean(predictions, axis=0)
                action = np.argmax(predictions)
                ps_all = np.zeros(self.config.num_products)
                ps_all[action] = 1.0

                # Store for statistics
                global_stats.append(np.max(predictions))

                return {
                    **super().act(observation, features),
                    **{"a": action, "ps": 1.0, "ps-a": ps_all},
                }

        # Get data
        features, actions, deltas, pss = self.train_data()

        # NxP vector where rows are users, columns are counts of organic views
        X = features

        N = X.shape[0]  # Number of bandit feedback samples
        P = X.shape[1]  # Number of items
        A = actions  # Vector of length N - indicating the action that was taken
        y = deltas  # Vector of length N - indicating whether a click occurred

        # Initialize a sparse DOK matrix and fill it up. This has the same result as a row by row kronecker product
        # of the features by A (one-hot-encoded)
        data = np.broadcast_to(features, (N, P)).ravel()
        ia = P * np.arange(N + 1)
        ja = np.ravel(P * actions[:, np.newaxis] + np.arange(P))
        training_matrix = sparse.csr_matrix((data, ja, ia), shape=(N, P * P))


        models = list()
        roc_auc = list()
        # precision = list()
        k_crosses = list()
        training_dataset = training_matrix.tocsr()
        for i in (3, 5, 10):
            for j in (0.01, 0.1, 0.3, 0.5, 1.0):
                general_model = MultinomialNB(alpha=j)
                scores = cross_validate(general_model, training_dataset, y,  return_estimator=True, cv=i,
                                        scoring=('roc_auc', 'precision_micro'))
                models = models + list(scores['estimator'])
                roc_auc = roc_auc + list(scores['test_roc_auc'])
                # precision = precision + list(scores['test_precision_micro'])
                k_crosses = k_crosses + [i] * i

        models = np.array(list(models))
        roc_auc = np.array(roc_auc)
        # precision = np.array(precision)

        indexes = (-roc_auc).argsort()[:3]
        # models = np.take(models, indexes)
        # precision = np.take(precision, indexes)
        # print(precision)
        # indexes = (-precision).argsort()[:5]
        # print(np.take(precision, indexes))
        model = np.take(models, indexes)

        return (
            CrossConfirmationMNBAgentFeaturesProvider(self.config),
            CrossConfirmationMNBAgentModel(self.config, model, training_matrix, y),
        )


class TestAgent(ModelBasedAgent):

    def __init__(self, config=Configuration(test_agent_args)):
        self.previous_features = None
        self.previous_action = None
        super().__init__(config, CrossConfirmationMNBAgentModelBuilder(config))

    def act(self, observation, reward, done):
        """ We're overloading this method so we can do online training on the previous observation whenever we get
        a new one """

        # Build model first if not yet done
        if self.model is None:
            assert self.feature_provider is None
            self.feature_provider, self.model = self.model_builder.build()

        # Now that we have the reward, train based on previous features and reward we got for our action
        if self.config.online_training and reward is not None:
            self.model.train_online(
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


global_stats = []
agent = build_agent_init("CrossConfirmationMNB", TestAgent, {**test_agent_args})