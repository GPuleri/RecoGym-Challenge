from multiprocessing import Process

from abc import ABC, abstractmethod

from recogym import build_agent_init, Configuration
from recogym.agents.abstract import (
    Agent,
    FeatureProvider,
    AbstractFeatureProvider,
    ViewsFeaturesProvider,
    Model,
    ModelBasedAgent,
    ModelBuilder,
)

import copy

import math

import pandas as pd
import numpy as np

import scipy
from scipy import sparse
from scipy.spatial.distance import cdist

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

test_agent_args = {
    "random_seed": np.random.randint(2 ** 31 - 1),
    "num_products": 10,
    'num_latent_factors': 20,
    "fallback_threshold": 0.00,
    "online_training": True,
    "online_training_batch": 1000,
}

################################################################################

class TestAgent(ModelBasedAgent):
    def __init__(self, config):
        super(TestAgent, self).__init__(
            config,
            # AE_ClostestProduct_ModelBuilder(config)
            AE_LogReg_ModelBuilder(config)
            # AE_CrossConfirmationMNB_ModelBuilder(config)
        )
        self.previous_features = None
        self.previous_action = None

    def act(self, observation, reward, done):
        """ We're overloading this method so we can do online training on the
        previous observation whenever we get a new one """

        # Build model first if not yet done
        if self.model is None:
            assert self.feature_provider is None
            assert self.model is None
            self.feature_provider, self.model = self.model_builder.build()
            print("[Test model]")

        # Now that we have the reward, train based on previous features and reward we got for our action
        if self.config.online_training \
            and self.previous_features is not None \
            and reward is not None:
            self.model.train_online(
                                    self.previous_features,
                                    self.previous_action,
                                    reward
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

################################################################################
# model_builders

# base class

class Basic_ModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(Basic_ModelBuilder, self).__init__(config)
        self.num_latent_factors = min(
                                        self.config.num_latent_factors,
                                        self.config.num_products - 1,
                                    )

    def get_user_views(self):
        d = pd.DataFrame().from_dict(self.data)

        user_ids = d['u'].unique()
        user_index_dict = { u_id: index for index, u_id in enumerate(user_ids) }

        # create a matrix of size (user_count, product_count)
        user_views = np.zeros(
                            (len(user_ids), self.config.num_products),
                            dtype="float32"
                            )
        # fill matrix: user_views[u,v] = #times user u viewed product v
        for row_index, row in d[d['z'] == 'organic'].iterrows():
            u = user_index_dict[row['u']]
            v = int(row['v'])
            user_views[u, v] += 1

        return user_views

    def build(self):
        raise NotImplemented

# sub classes

class AE_ClostestProduct_ModelBuilder(Basic_ModelBuilder):
    def __init__(self, config):
        super(AE_ClostestProduct_ModelBuilder, self).__init__(config)
        self.num_latent_factors = min(
                                        self.config.num_latent_factors,
                                        self.config.num_products - 1,
                                    )

    def build(self):
        user_views = self.get_user_views()

        pre_processor = ZNormalizer()

        # mapper = LatentMapperSVD(self.num_latent_factors, pre_processor, user_views)
        latent_mapper = LatentMapperAE(
                                        self.num_latent_factors,
                                        pre_processor,
                                        self.config.num_products
                                      )
        latent_mapper.train(user_views)
        characteristic_users = latent_mapper.apply(np.eye(self.config.num_products))

        return (
                LatentFeatureProvider(self.config, latent_mapper),
                ClosestProductModel(self.config, characteristic_users)
        )

class AE_LogReg_ModelBuilder(Basic_ModelBuilder):
    def __init__(self, config):
        super(AE_LogReg_ModelBuilder, self).__init__(config)

    def build(self):
        user_views = self.get_user_views()

        pre_processor = ZNormalizer()

        # Create & train a latent mapper
        # latent_mapper = LatentMapperSVD(self.num_latent_factors, pre_processor, user_views)
        latent_mapper = LatentMapperAE(
                                        self.num_latent_factors,
                                        pre_processor,
                                        self.config.num_products
                                      )
        latent_mapper.train(user_views)

        # Create & train a model
        model = LogRegModel(self.config)
        features, actions, clicks, _pss = self.train_data()
        processed_features = latent_mapper.apply(features)
        model.train(processed_features, actions, clicks, _pss)

        return (
            LatentFeatureProvider(self.config, latent_mapper),
            model
        )

class AE_CrossConfirmationMNB_ModelBuilder(Basic_ModelBuilder):
    def __init__(self, config):
        super(AE_CrossConfirmationMNB_ModelBuilder, self).__init__(config)

    def build(self):
        class CrossConfirmationMNBAgentFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super().__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        # user_views = self.get_user_views()
        #
        # pre_processor = None #ZNormalizer()
        #
        # # Create & train a latent mapper
        # latent_mapper = LatentMapperAE(
        #                                 self.num_latent_factors,
        #                                 pre_processor,
        #                                 self.config.num_products
        #                               )
        # latent_mapper.train(user_views)

        # Create & train a model
        model = CrossConfirmationMNBAgentModel(self.config)
        features, actions, clicks, pss = self.train_data()
        # features = latent_mapper.apply(features)
        model.train(features, actions, clicks, pss)

        return (
            CrossConfirmationMNBAgentFeaturesProvider(self.config),
            model
        )

################################################################################
# PreProcessors

# base class

class PreProcessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process_training_data(self, user_views):
        pass

    @abstractmethod
    def process_row(self, user_view):
        pass

    def divide_or_zero(self, a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

# subclasses

class ZNormalizer(PreProcessor):
    def __init__(self):
        self.means = None
        self.stds = None

    def process_training_data(self, user_views):
        # row_wise division by the sum of the row
        user_profiles = self.divide_or_zero(user_views, user_views.sum(axis=1).reshape((user_views.shape[0],1)))

        # compute mean and std for every column
        self.means = user_profiles.mean(axis=0)
        self.stds = user_profiles.std(axis=0)

        # z-normalization of user_views: X'[:,i] = (X[:,i] - μ_i) / σ_i
        normalized_profiles = self.divide_or_zero(user_profiles-self.means, self.stds)

        return normalized_profiles

    def process_row(self, user_view):
        user_view = self.divide_or_zero(user_view, user_view.sum())
        return self.divide_or_zero(user_view-self.means, self.stds)

class DivideBySum(PreProcessor):
    def __init__(self):
        pass

    def process_training_data(self, user_views):
        return self.divide_or_zero(user_profiles, user_profiles.sum(acis=0))

    def process_row(self):
        return self.divide_or_zero(user_view, user_view.sum())

################################################################################
# LatentMappers

# base class

class LatentMapper(ABC):
    def __init__(self, num_latent_factors):
        self.num_latent_factors = num_latent_factors

    @abstractmethod
    def apply(self, user_view):
        pass

# subclasses

class LatentMapperSVD(LatentMapper):
    """Latent mapper using singular value decomposition."""
    def __init__(self, num_latent_factors, pre_processor):
        super(LatentMapperSVD, self).__init__(num_latent_factors)
        self.pre_processor = pre_processor
        self.v = None

    def train(self, unprocessed_user_views):
        user_views = unprocessed_user_views
        if self.pre_processor:
            user_views = self.pre_processor.process_training_data(unprocessed_user_views)
        _, _, self.v = sparse.linalg.svds(user_views, self.num_latent_factors)

    def apply(self, unprocessed_user_view):
        user_view = unprocessed_user_view
        if self.pre_processor:
            user_view = self.pre_processor.process_row(unprocessed_user_view)
        return np.dot(user_view, self.v.T)

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AutoEncoder,self).__init__()

        self.encoder = nn.Linear(in_channels, out_channels)
        self.decoder = nn.Linear(out_channels, in_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def apply(self, x):
        return self.encoder(x)

class LatentMapperAE(LatentMapper):
    """Latent mapper using autoencoder."""
    def __init__(self, num_latent_factors, pre_processor, num_products):
        super(LatentMapperAE, self).__init__(num_latent_factors)
        self.num_latent_factors = num_latent_factors
        self.pre_processor = pre_processor

        self.num_products = num_products
        self.num_epochs = 5
        self.batch_size = 20
        self.learning_rate = 1e-3

        self.model = AutoEncoder(
                                in_channels = self.num_products,
                                out_channels = self.num_latent_factors,
                                )
        self.criterion = nn.MSELoss()
        # use other loss-function?
        # denoizing AutoEncoders (not useful because there is no noise)
        # and variation AutoEncoders
        self.optimizer = torch.optim.Adam(
                                        self.model.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=1e-5
                                        )

    def train(self, unprocessed_user_views):
        print("[Train Latent Mapper]")
        user_views = unprocessed_user_views
        if self.pre_processor:
            user_views = self.pre_processor.process_training_data(unprocessed_user_views)
        user_views = torch.from_numpy(user_views)

        n_epochs = 5000 # this should certainly be enough for convergence
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            output = self.model(user_views)
            loss = F.mse_loss(output, user_views)
            loss.backward()
            self.optimizer.step()
            # if epoch % 100 == 0:
            #     print(f"{epoch}: {loss.item()}")

        self.model.eval()

    def apply(self, unprocessed_user_view):
        user_view = unprocessed_user_view
        if self.pre_processor:
            user_view = self.pre_processor.process_row(unprocessed_user_view)
        user_view = torch.from_numpy(user_view).float()

        with torch.no_grad():
            r = self.model.apply(user_view)
            return r

################################################################################
# FeatureProviders

class LatentFeatureProvider(FeatureProvider):
    def __init__(self, config, latent_mapper):
        super(LatentFeatureProvider, self).__init__(config)
        self.latent_mapper = latent_mapper
        self.view_counts = None
        self.reset()

    def observe(self, observation):
        for s in observation.sessions():
            self.view_counts[s['v']] += 1

    def features(self, observation):
        return self.latent_mapper.apply(self.view_counts)

    def reset(self):
        self.view_counts = np.zeros(self.config.num_products)

################################################################################
# Models

# base class

class Basic_model(Model):
    def __init__(self, config):
        super(Basic_model, self).__init__(config)

    def arange_kronecker(self, features, actions, num_actions):
        """ compute kronecker product of each feature with one-hot encoded action """
        n = actions.size
        num_features = features.shape[-1]
        data = np.broadcast_to(features, (n, num_features)).ravel()
        ia = num_features * np.arange(n+1)
        ja = np.ravel(num_features*actions[:, np.newaxis] + np.arange(num_features))
        return sparse.csr_matrix(
                                (data, ja, ia),
                                shape=(n, num_features*num_actions)
                                )

    def train(self, features, actions, clicks, pss):
        pass

    # def act(self, observation, features):
    #     pass

    def train_online(self, features, action, reward):
        pass

    def reset_online(self):
        pass

# sub classes

class ClosestProductModel(Basic_model):
    def __init__(self, config, product_encodings, metric='cosine'):
        super(ClosestProductModel, self).__init__(config)
        self.product_encodings = product_encodings
        self.metric = metric

    def act(self, observation, features):
        dists = cdist(
            features.reshape(1, -1),
            self.product_encodings,
            self.metric,
        )
        return {
            **super().act(observation, features),
            'a': np.argmin(dists),
            'ps': 1,
        }

class LogRegModel(Basic_model):
    def __init__(self, config):
        super(LogRegModel, self).__init__(config)
        self.model = LogisticRegression(
                                        solver = 'lbfgs',
                                        warm_start=True,
                                        max_iter=10000
                                        )
        self.reset_count = -1
        self.reset_online()

    def train(self, features, actions, clicks, pss):
        print("[Train LogRegModel]")
        training_matrix = self.arange_kronecker(
                                                features,
                                                actions,
                                                self.config.num_products
                                                )
        self.model.fit(training_matrix, clicks)

    def act(self, observation, features):
        # We want a prediction for every action
        a = np.arange(self.config.num_products)
        inputs = self.arange_kronecker(features, a, self.config.num_products)

        # Get prediction for every action
        predictions = self.model.predict_proba(inputs)[:,1]

        return {
            **super().act(observation, features),
            **{
                'a': np.argmax(predictions),
                'ps': 1.0,
            },
        }

    def train_online(self, features, action, reward):
        # # don't use feature that didn't get rewarded
        # if not reward:
        #     return

        # Update our online batch
        new_result = self.arange_kronecker(
                                            features,
                                            np.array([action]),
                                            self.config.num_products
                                            )
        self.online_matrix[self.online_count] = new_result
        self.online_clicks[self.online_count] = int(reward)
        self.online_count += 1

        # Check if we want to do a training session
        if self.online_count == self.config.online_training_batch:
            self.model.fit(self.online_matrix.tocsr(), self.online_clicks)

            # Reset our online batch data
            self.reset_online()

    def reset_online(self):
        self.reset_count += 1
        B = self.config.online_training_batch   # batch size
        F = self.config.num_latent_factors      # feature size
        P = self.config.num_products            # product count
        self.online_count   = 0
        self.online_matrix  = sparse.dok_matrix((B, F*P), dtype=np.float32)
        self.online_clicks  = np.zeros(B)

class CrossConfirmationMNBAgentModel(Basic_model):
    def __init__(self, config):
        super(CrossConfirmationMNBAgentModel, self).__init__(config)
        # self.model = model
        # self.full_x = full_x
        # self.full_y = full_y
        self.model  = None
        self.full_x = None
        self.full_y = None
        self.online_matrix = None
        self.P = config.num_products

        self.times_acted = 0
        self.online_rewards = np.empty(self.config.online_training_batch)
        self.online_count = 0

    def train(self, features, actions, clicks, pss):
        print("[Train MNB Model]")
        # NxP vector where rows are users, columns are counts of organic views
        X = features
        y = clicks

        N = X.shape[0]  # Number of bandit feedback samples
        # A = actions  # Vector of length N - indicating the action that was taken

        # # Initialize a sparse DOK matrix and fill it up. This has the same result as a row by row kronecker product
        # # of the features by A (one-hot-encoded)
        # data = np.broadcast_to(features, (N, P)).ravel()
        # ia = P * np.arange(N + 1)
        # ja = np.ravel(P * actions[:, np.newaxis] + np.arange(P))
        # training_matrix = sparse.csr_matrix((data, ja, ia), shape=(N, P * P))
        training_matrix = self.arange_kronecker(
                                                features,
                                                actions,
                                                self.config.num_products
                                                )

        models = list()
        roc_auc = list()
        # precision = list()
        k_crosses = list()
        training_dataset = training_matrix.tocsr()
        for i in (3, 5, 10):
            for j in (0.01, 0.1, 0.3, 0.5, 1.0):
                general_model = MultinomialNB(alpha=j)
                scores = cross_validate(
                                        general_model,
                                        training_dataset,
                                        y,
                                        return_estimator=True,
                                        cv=i,
                                        scoring=('roc_auc', 'precision_micro')
                                       )
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
        self.model = np.take(models, indexes)
        self.full_x = training_matrix
        self.full_y = clicks  # Vector of length N - indicating whether a click occurred
        self.online_matrix = sparse.dok_matrix(
            (self.config.online_training_batch, self.P * self.P)
        )

    def act(self, observation, features):
        # Show progress
        self.times_acted += 1

        # We want a prediction for every action
        matrix = sparse.kron(sparse.eye(self.P), features, format="csr")

        # Get prediction for every action
        predictions_0 = self.model[0].predict_proba(matrix)[:, 1]
        predictions_1 = self.model[1].predict_proba(matrix)[:, 1]
        predictions_2 = self.model[2].predict_proba(matrix)[:, 1]

        action_0 = np.argmax(predictions_0)
        action_1 = np.argmax(predictions_1)
        action_2 = np.argmax(predictions_2)

        if np.sum(predictions_0) + np.sum(predictions_1)  \
                + np.sum(predictions_2) <= self.config.fallback_threshold:
            action = np.argmax(features)
        elif action_0 == action_1 == action_2:
            action = action_0
        elif action_0 == action_1 != action_2 \
                or action_0 == action_2 != action_1:
            if predictions_0[action_0] + predictions_1[action_1] \
                    >= predictions_2[action_2]:
                action = action_0
            else:
                action = action_2
        elif action_0 != action_1 == action_2:
            if predictions_0[action_1] + predictions_1[action_2] \
                    >= predictions_2[action_2]:
                action = action_1
            else:
                action = action_0
        else:
            action = np.argmax(features)
        predictions = np.array([predictions_0, predictions_1, predictions_2])
        predictions = np.mean(predictions, axis=0)
        ps_all = np.zeros(self.config.num_products)
        ps_all[action] = 1.0

        # Store for statistics
        # global_stats.append(np.max(predictions))

        return {
            **super().act(observation, features),
            **{"a": action, "ps": 1.0, "ps-a": ps_all},
        }

    def reset_online(self):
        self.online_matrix = sparse.dok_matrix((
                                            self.config.online_training_batch,
                                            self.P * self.P
                                            ))
        self.online_rewards = np.empty(self.config.online_training_batch)
        self.online_count = 0

    def train_online(self, features, action, reward):
        """ This method does the online training (in batches) based on
        the reward we got for the previous action """
        # don't use feature that didn't get rewarded
        if not reward:
            return

        # Update our online batch
        for j in range(self.P):
            self.online_matrix[(self.online_count, action * self.P + j)] = features[0, j]

        self.online_rewards[self.online_count] = reward
        self.online_count += 1

        # Check if we want to do a training session
        if self.online_count == self.config.online_training_batch:
            # Stack matrices
            self.full_x = sparse.vstack([self.full_x, self.online_matrix])
            self.full_y = np.concatenate([self.full_y, self.online_rewards])

            # Reached batch size, completely retrain our booster on cumulative data
            online_dataset = self.full_x.tocsr()
            online_label = self.full_y
            self.model[0] = self.model[0].partial_fit(online_dataset, online_label)
            self.model[1] = self.model[1].partial_fit(online_dataset, online_label)
            self.model[2] = self.model[2].partial_fit(online_dataset, online_label)

            # Reset our online batch data
            self.reset_online()

# only offline training
seed = 4473
# kwargsOffline = copy.deepcopy(test_agent_args)
# kwargsOffline["seed"] = seed
# kwargsOffline["online_training"] = False
# # agent = build_agent_init("AE_LogReg_Agent_Offline_seed"+str(seed), TestAgent, test_agent_args)
# print("START [Process AE_LogReg_Agent_Offline_seed {}]".format(seed))
# Process(
#     target=build_agent_init,
#     args=("AE_LogReg_Agent_Offline_seed"+str(seed), TestAgent, kwargsOffline)
#     ).start()

# # with online training
kwargsOnline["seed"] = seed
kwargsOnline["online_training"] = True
# agent = build_agent_init("AE_LogReg_Agent_Online_seed"+str(seed), TestAgent, test_agent_args)
print("START [Process AE_LogReg_Agent_Online_seed {}]".format(seed))
Process(
        target=build_agent_init,
        args=("AE_LogReg_Agent_Online_seed"+str(seed), TestAgent, kwargsOnline)
        ).start()
