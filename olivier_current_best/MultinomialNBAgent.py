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

from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.naive_bayes import MultinomialNB

test_agent_args = {
    "random_seed": np.random.randint(2 ** 31 - 1),
    "num_products": 10,  # mandatory, will be overwritten
    "online_training": True,  # whether we should do online training
    "online_training_batch": 100,  # records to gather before starting an online training session
    "grid_search_interval": 50000,  # Every x batches, do a grid_search optimization
    "epsilon": 0,  # for epsilon-greedy online learning
}


def arange_kronecker(features, actions, num_actions):
    """ compute kronecker product of each feature with one-hot encoded action """
    n = actions.size
    num_features = features.shape[-1]
    data = np.broadcast_to(features, (n, num_features)).ravel()
    ia = num_features * np.arange(n+1)
    ja = np.ravel(num_features*actions[:, np.newaxis] + np.arange(num_features))
    return sparse.csr_matrix((data, ja, ia), shape=(n, num_features*num_actions), dtype='uint16')

def preprocess(data):
    # 0s are not liked
    data = data + 1.0
    # Distinctly identify top-viewed item too
    top = np.zeros_like(data)
    for row in range(top.shape[0]):
        cols = np.argwhere(data[row,:] == np.max(data[row,:]))
        top[row,cols] = 1.0
    # Return hstack
    return np.hstack((data,top))
    #bins = np.array([0.0, 1.0, np.inf])
    #np.digitize(data, bins)# - 1.0
    #return data.astype(np.bool)
    #return data


class MultinomialNBAgentModel(Model):
    def __init__(self, config, x, a, y):
        super().__init__(config)
        self.model = None
        self.num_features = x.shape[1]
        self.times_acted = 0
        self.batch_x = x.astype('uint16')
        self.batch_a = a.astype('uint16')
        self.batch_y = y.astype('bool')
        self.batch_size = 0
        self.full_x = None
        self.full_a = None
        self.full_y = None
        self.batches_done = 0
        self.positive_class_index = 1
        random.seed(2468)

        self.action2count = [0] * config.num_products

    def add_batch_to_full(self):
        # Add last batch to the full set
        self.full_x = self.batch_x if self.full_x is None else np.vstack([self.full_x, self.batch_x])
        self.full_a = self.batch_a if self.full_a is None else np.concatenate([self.full_a, self.batch_a])
        self.full_y = self.batch_y if self.full_y is None else np.concatenate([self.full_y, self.batch_y])

    def reset_batches(self):
        self.batch_x = np.empty((self.config.online_training_batch, self.num_features), dtype='uint16')
        self.batch_a = np.empty(self.config.online_training_batch, dtype='uint16')
        self.batch_y = np.empty(self.config.online_training_batch, dtype='bool')
        self.batch_size = 0
        self.batches_done += 1

    def update_data(self, features, action, reward):
        """ This functions processes a single online training record """
        # Store for statistics
        global_stats.append(reward)

        # Update our online batch
        self.batch_x[self.batch_size] = features[0]
        self.batch_a[self.batch_size] = action
        self.batch_y[self.batch_size] = reward
        self.batch_size += 1

        # Check if batches are big enough to do a training session
        if self.batch_size == self.config.online_training_batch:
            # We have a full batch, process it
            self.add_batch_to_full()

            if self.batches_done % self.config.grid_search_interval == 0:
                # Rebuild model with grid search and all data
                self.model = do_grid_search(
                    arange_kronecker(preprocess(self.full_x), self.full_a, self.config.num_products),
                    self.full_y
                )
            else:
                # Partial fit on our batch data
                kronecker_product = arange_kronecker(preprocess(self.batch_x), self.batch_a, self.config.num_products)
                self.model.partial_fit(kronecker_product, self.batch_y)
            self.reset_batches()

    def act(self, observation, features):
        # Show progress
        if self.times_acted and self.times_acted % 10000 == 0:
            running_stats = global_stats[-10000:]
            ctr_running = sum(running_stats) / 10000
            ctr_cumulative = sum(global_stats) / len(global_stats)
            print(
                f'Times acted: {self.times_acted} | Running CTR: {ctr_running:.2%} ({len(running_stats)}) | Total CTR: {ctr_cumulative:.2%} ({len(global_stats)}) | Alpha: {self.model.get_params()["alpha"]}')

        self.times_acted += 1

        # Decide if we want to explore or exploit
        if random.random() < self.config.epsilon:
            # Explore: pick a random product
            action = random.randrange(self.config.num_products)
        else:
            # Exploit: take our best guess

            # We want a prediction for every action
            matrix = sparse.kron(sparse.eye(features.shape[1]), preprocess(features), format="csr")

            # Get prediction for every action
            predictions = self.model.predict_proba(matrix)
            predictions = predictions[:, self.positive_class_index]

            # Take best action
            action = np.argmax(predictions)
            self.action2count[action] += 1

        ps_all = np.zeros(self.config.num_products)
        ps_all[action] = 1.0

        return {
            **super().act(observation, features),
            **{"a": action, "ps": 1.0, "ps-a": ps_all},
        }


class MultinomialNBAgentFeaturesProvider(ViewsFeaturesProvider):
    def __init__(self, config):
        super().__init__(config)
        print(f"Number of products: {self.config.num_products}")

    def features(self, observation):
        base_features = super().features(observation)
        return base_features.reshape(1, self.config.num_products)


class MultinomialNBAgentModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        print("Building model with the following configuration:")
        for k, v in self.config.__dict__.items():
            print(f'{str(k).ljust(25)}: {v}')

        # Get data
        offline_features, offline_actions, offline_rewards, offline_pss = self.train_data()

        model = MultinomialNBAgentModel(self.config, offline_features, offline_actions, offline_rewards)
        model.add_batch_to_full()

        model.model = do_grid_search(
            arange_kronecker(preprocess(offline_features), offline_actions, self.config.num_products),
            offline_rewards
        )

        model.reset_batches()

        return (
            MultinomialNBAgentFeaturesProvider(self.config),
            model,
        )


def do_grid_search(X, y):
    """ Does an exhaustive grid search on MultinomialNB hyperparameters """
    mnb_param_grid = {
        # "alpha": [0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        "alpha": [10**(-i) for i in range(10)] + [.005, .075, .0125, .025, .05, .15, .3, .5, .75]
    }
    mnb_classifier = MultinomialNB()
    gcv = GridSearchCV(
        mnb_classifier,
        mnb_param_grid,
        cv=KFold(n_splits=10, shuffle=True, random_state=712),
        scoring='recall',
        n_jobs=-1,
        verbose=0)
    gcv.fit(X, y)

    # Print winning estimator
    """
    print(f"\nBest estimator:\n{gcv.best_estimator_}")
    print("Best params:")
    for k, v in gcv.best_params_.items():
        print(f"{str(k).ljust(25)}\tBest: {v}")
    """
    # Return the winning estimator
    return gcv.best_estimator_


class TestAgent(ModelBasedAgent):
    """
    LightGBM Gradient Boosting Agent.
    """

    def __init__(self, config=Configuration(test_agent_args)):
        self.previous_features = None
        self.previous_action = None
        super().__init__(config, MultinomialNBAgentModelBuilder(config))

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

global_stats = []
agent = build_agent_init("MultinomialNBAgent", TestAgent, {**test_agent_args})
