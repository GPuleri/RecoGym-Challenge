import numpy as np
from scipy import sparse
from recogym import build_agent_init, Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider,
)

from sklearn.ensemble import (
    GradientBoostingClassifier,
)

from sklearn.base import BaseEstimator

from sklearn.model_selection import GridSearchCV, StratifiedKFold

pirate_agent_args = {
    "random_seed": np.random.randint(2 ** 31 - 1),
    #"random_seed": 42,
    "fallback_threshold": 0.0,
    "online_training": False
}


class PirateAgentModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(PirateAgentModelBuilder, self).__init__(config)

    def build(self):
        class PirateAgentFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super(PirateAgentFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class PirateAgentModel(Model):
            def __init__(self, config, model):
                super(PirateAgentModel, self).__init__(config)
                self.model = model
                self.times_acted = 0
                self.positive_class_index = np.where(model.classes_ == 1)[0][0]

            def train_online(self, features, action, reward):
                """ This method does the online training based on reward we got for the previous action """
                assert reward is not None
                assert features is not None
                assert action is not None

                action_one_hot = np.zeros((1, P), dtype="uint8")
                action_one_hot[0, action] = 1

                matrix = sparse.kron(features.astype("uint8"), action_one_hot, format="csr")
                y = np.array([reward], dtype="float16")

                self.model.fit(matrix, y)

            def act(self, observation, features):
                # Show progress
                self.times_acted += 1
                if self.times_acted % 1000 == 0:
                    print(f"{self.times_acted} acts", end="\r")

                # We want a prediction for every action
                matrix = sparse.kron(
                    features.astype("uint8"), sparse.eye(P, dtype="uint8"), format="csr"
                )

                # Get prediction for every action
                predictions = model.predict_proba(matrix)
                predictions = predictions[:, self.positive_class_index]

                # Store for statistics
                global_stats.append(np.max(predictions))

                # Check if we need to fallback to most viewed organic
                if np.max(predictions) > self.config.fallback_threshold:
                    # Take the one you think is most likely to give a click
                    action = np.argmax(predictions)
                    ps_all = np.zeros(self.config.num_products)
                    ps_all[action] = 1.0
                else:
                    # Fallback to
                    action = np.argmax(features)
                    ps_all = np.zeros(self.config.num_products)
                    ps_all[action] = 1.0

                return {
                    **super().act(observation, features),
                    **{"a": action, "ps": 1.0, "ps-a": ps_all},
                }

        # Get data
        features, actions, deltas, pss = self.train_data()

        # NxP vector where rows are users, columns are counts of organic views
        X = features.astype("uint8")

        N = X.shape[0]  # Number of bandit feedback samples
        P = X.shape[1]  # Number of items
        A = actions  # Vector of length N - indicating the action that was taken
        y = deltas.astype(
            "float16"
        )  # Vector of length N - indicating whether a click occurred

        # Initialize a sparse DOK matrix and fill it up. This has the same result as a row by row kronecker product
        # of the features by A (one-hot-encoded)
        print(f"Building training matrix ({N},{P*P}) ...")
        training_matrix = sparse.dok_matrix((N, P * P), dtype="uint8")
        for i in range(N):
            if i % (N // 10) == 0:
                print(f"{i * 100 / (N // 10 * 10)} %", end="\r")
            for j in range(P * P):
                if A[i] == j % P and X[i][j // P] != 0:
                    training_matrix[(i, j)] = X[i][j // P]

        # Now that the matrix is built, switch from DOK to CSR
        print("\nSwitching training matrix to CSR format")
        training_matrix = training_matrix.tocsr()
        """
        ### BEGIN GRID SEARCH ###
        gp_param_grid = {
            "learning_rate": [0.01, 0.02, 0.03, 0.1, 0.2],
            "max_features": [None, "sqrt", "log2"],
            "max_depth": [None, 1, 2, 3, 4],
            "n_estimators": [50, 75, 100, 125]
        }
        gcv = GridSearchCV(GradientBoostingClassifier(), gp_param_grid, cv=5, n_jobs=-1, verbose=1)
        gcv.fit(training_matrix, y)
        print(f'\nBest estimator:\n{gcv.best_estimator_}')
        print("Best params:")
        for k, v in gcv.best_params_.items():
            print(f"{str(k).ljust(25)}\tBest: {v}")
        exit()
        ### END GRID SEARCH ###

        # Train GradientBoostingClassifier
        # Best params:
        # learning_rate            	Best: 0.02
        # max_depth                	Best: 2
        # max_features             	Best: 5
        # n_estimators             	Best: 50
        """
        gb_params = {
            "learning_rate": 0.01,
            "max_depth": None,
            "max_features": None,
            "n_estimators": 50
        }

        #gb_params = {}
        print("Training model with the following parameters:")
        for k, v in gb_params.items():
            print(f'{k.ljust(30, " ")}{v}')

        model = GradientBoostingClassifier(**gb_params).fit(training_matrix, y)
        model.set_params(**{"warm_start": True})

        return (
            PirateAgentFeaturesProvider(self.config),
            PirateAgentModel(self.config, model),
        )


class PirateAgent(ModelBasedAgent):
    """
    Scikit-Learn-based GradientBoosting Agent.
    """
    def __init__(self, config=Configuration(pirate_agent_args)):
        self.previous_features = None
        self.previous_action = None
        super(PirateAgent, self).__init__(config, PirateAgentModelBuilder(config))


    def act(self, observation, reward, done):
        """ We're overloading this method so we can do online training on the previous observation whenever we get
        a new one """

        # Build model first if not yet done
        if self.model is None:
            assert (self.feature_provider is None)
            self.feature_provider, self.model = self.model_builder.build()

        # Now that we have the reward, train based on previous features and reward we got for our action
        #if self.config.online_training and reward is not None:
        #    self.model.train_online(self.previous_features, self.previous_action, reward)

        # Update the feature provider with this new observation
        self.feature_provider.observe(observation)

        # Get the new features
        features = self.feature_provider.features(observation)
        a_ps_psa_dict = self.model.act(observation, features)

        # Update previous feature set for next online learning session
        self.previous_features = features
        self.previous_action = a_ps_psa_dict['a']

        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            **a_ps_psa_dict,
        }


global_stats = []
agent = build_agent_init("PirateAgent", PirateAgent, {**pirate_agent_args})


###
# Test & train locally, this is not used when submitting but helps for local debugging
#

if __name__ == "__main__":
    import gym
    from recogym import env_1_args
    from recogym.bench_agents import test_agent

    num_products = 10
    num_users = 300

    pirate_agent = PirateAgent(
        Configuration({"random_seed": 33, "num_products": num_products, "fallback_threshold": 0.0, "online_training": False})
    )

    env_1_args["random_seed"] = 71
    env_1_args["num_products"] = num_products
    env = gym.make("reco-gym-v1")
    env.init_gym(env_1_args)

    print(
        test_agent(
            env, pirate_agent, num_offline_users=num_users, num_online_users=num_users
        )
    )

    hfreq, hedges = np.histogram(global_stats, np.arange(0, 1.1, 0.1))
    print("\n*** PROBABILITIES HISTOGRAM ***")
    print(f"Total of {len(global_stats)} samples")
    print(f"Number of zero probabilities: {len([x for x in hfreq if x == 0])}")
    for freq, edge in zip(hfreq, hedges):
        print(f"{round(edge, 1)} : {freq}")
