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

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

test_agent_args = {
    "random_seed": np.random.randint(2 ** 31 - 1),
    "num_products": 10,  # mandatory, will be overwritten
    "online_training": True,  # whether we should do online training
    "online_training_batch": 100,  # records to gather before starting an online training session
    "epsilon": 0.0  # for epsilon-greedy online learning
}


class LightGBMAgentModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        class LightGBMAgentFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super().__init__(config)
                print(f"Number of products: {self.config.num_products}")

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class LightGBMAgentModel(Model):
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
                random.seed(2468)

            def reset_online(self):
                self.online_matrix = sparse.dok_matrix(
                    (self.config.online_training_batch, P * P)
                )
                self.online_rewards = np.empty(self.config.online_training_batch)
                self.online_count = 0

            def train_online(self, features, action, reward):
                """ This method does the online training (in batches) based on the reward we got for the previous
                action """

                # Store for statistics
                global_stats.append(reward)

                # Update our online batch
                for j in range(P):
                    self.online_matrix[(self.online_count, action * P + j)] = features[0, j]

                self.online_rewards[self.online_count] = reward
                self.online_count += 1

                # Check if we want to do a training session
                if self.online_count == self.config.online_training_batch:
                    # Stack matrices
                    self.full_x = sparse.vstack([self.full_x, self.online_matrix])
                    self.full_y = np.concatenate([self.full_y, self.online_rewards])

                    # Reached batch size, completely retrain our booster on cumulative data
                    online_dataset = lgb.Dataset(self.full_x.tocsr(), label=self.full_y, free_raw_data=False)
                    self.model = lgb.train(lgb_params, online_dataset, num_boost_round=100)

                    # Print progress
                    """
                    print(
                        f"Booster iteration: {self.model.current_iteration()} | Data: {online_dataset.num_data()} | CTR: {np.sum(self.online_rewards) / self.online_rewards.shape[0]}"
                    )
                    """
                    # Do grid search on a very large batch (gives same results as small batch)
                    # if 150000 <= online_dataset.num_data() <= 150000 + self.config.online_training_batch:
                    #     do_grid_search(self.full_x.tocsr(), self.full_y)

                    # Reset our online batch data
                    self.reset_online()

            def act(self, observation, features):
                # Show progress
                if self.times_acted and self.times_acted % 10000 == 0:
                    running_stats = global_stats[-10000:]
                    ctr_running = sum(running_stats) / 10000
                    ctr_cumulative = sum(global_stats) / len(global_stats)
                    print(f'Times acted: {self.times_acted} | Running CTR: {ctr_running:.2%} ({len(running_stats)}) | Total CTR: {ctr_cumulative:.2%} ({len(global_stats)})')

                self.times_acted += 1

                # Decide if we want to explore or exploit
                if random.random() < self.config.epsilon:
                    # Explore: pick a random product
                    action = random.randrange(self.config.num_products)
                else:
                    # Exploit: take our best guess
                    # We want a prediction for every action
                    matrix = sparse.kron(sparse.eye(P), features, format="csr")

                    # Get prediction for every action
                    predictions = self.model.predict(matrix)

                    # Take best action
                    action = np.argmax(predictions)

                ps_all = np.zeros(self.config.num_products)
                ps_all[action] = 1.0

                return {
                    **super().act(observation, features),
                    **{"a": action, "ps": 1.0, "ps-a": ps_all},
                }

        # Get data
        offline_features, offline_actions, offline_deltas, offline_pss = self.train_data()

        # NxP vector where rows are users, columns are counts of organic views
        X = offline_features

        N = X.shape[0]  # Number of bandit feedback samples
        P = X.shape[1]  # Number of items
        # A = offline_actions  # Vector of length N - indicating the action that was taken
        y = offline_deltas  # Vector of length N - indicating whether a click occurred

        # Initialize a sparse DOK matrix and fill it up. This has the same result as a row by row kronecker product
        # of the features by A (one-hot-encoded)
        print(f"Building training matrix ({N},{P*P}) ...")
        data = np.broadcast_to(offline_features, (N, P)).ravel()
        ia = P * np.arange(N + 1)
        ja = np.ravel(P * offline_actions[:, np.newaxis] + np.arange(P))
        training_matrix = sparse.csr_matrix((data, ja, ia), shape=(N, P * P))

        # do_grid_search(training_matrix, y)

        """Best params: (GOOD SET)
        boosting_type            	Best: gbdt
        max_bin                  	Best: 5
        min_data_in_leaf         	Best: 5
        num_leaves               	Best: 5
        
        Best params: (Narrowed down)
        boosting_type            	Best: gbdt
        max_bin                  	Best: 2
        min_data_in_leaf         	Best: 3
        num_leaves               	Best: 2
        """
        lgb_params = {
            "objective": "binary",
            "num_leaves": 2,  # default 31
            "max_bin": 2,  # default 255
            "min_data_in_leaf": 3,  # default 20
            "boost_from_average": True,
            "verbosity": -1
        }

        print("Training model with the following parameters:")
        for k, v in lgb_params.items():
            print(f'{k.ljust(30, " ")}{v}')

        training_dataset = lgb.Dataset(training_matrix.tocsr(), label=y, free_raw_data=False)
        model = None
        model = lgb.train(
            lgb_params,
            training_dataset,
            num_boost_round=100,
            init_model=model,
            keep_training_booster=True,
        )

        return (
            LightGBMAgentFeaturesProvider(self.config),
            LightGBMAgentModel(self.config, model, training_matrix, y),
        )


def do_grid_search(X, y):
    """ Does a grid search to find a good combination for the most important LightGBM hyper parameters given
    training set X and labels y """
    lgb_param_grid = {
        "boosting_type": ["gbdt", "dart"],
        "num_leaves": range(2, 11),
        "min_data_in_leaf": range(11),
        "max_bin": range(2, 15),
    }
    lgb_classifier = lgb.LGBMClassifier(
        objective="binary", n_estimators=100, verbosity=-1
    )
    gcv = GridSearchCV(lgb_classifier, lgb_param_grid, cv=3, n_jobs=-1, verbose=1)
    gcv.fit(X, y)
    print(f"\nBest estimator:\n{gcv.best_estimator_}")
    print("Best params:")
    for k, v in gcv.best_params_.items():
        print(f"{str(k).ljust(25)}\tBest: {v}")
    exit()


class TestAgent(ModelBasedAgent):
    """
    LightGBM Gradient Boosting Agent.
    """

    def __init__(self, config=Configuration(test_agent_args)):
        self.previous_features = None
        self.previous_action = None
        super().__init__(config, LightGBMAgentModelBuilder(config))

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
agent = build_agent_init("LightGBMAgent", TestAgent, {**test_agent_args})
