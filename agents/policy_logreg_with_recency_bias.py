from tqdm import tqdm
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
    "num_latent_factors": 20,
    "recency_bias": 0.85,
    "fallback_threshold": 0.00,
    "online_training": False,
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
            # print("[Test model]")

        # Now that we have the reward, train based on previous features and
        #   reward we got for our action
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

class AE_LogReg_ModelBuilder(AbstractFeatureProvider):
    def __init__(self, config, is_sparse=False):
        super(AE_LogReg_ModelBuilder, self).__init__(config)

    def train_data(self):
        data = pd.DataFrame().from_dict(self.data)

        features = []
        actions = []
        pss = []
        deltas = []

        for user_id in tqdm(data['u'].unique(), desc='Train Data'):
            f = np.zeros(self.config.num_products)

            # for every user
            for _, user_datum in data[data['u'] == user_id].iterrows():
                if user_datum['z'] == 'organic':
                    f *= self.config.recency_bias
                    view = np.int16(user_datum['v']) #index of viewed product
                    f[view] += 1
                else: #bandit
                    action = np.int16(user_datum['a'])
                    delta = np.int16(user_datum['c'])
                    ps = user_datum['ps']
                    time = np.int16(user_datum['t'])-1

                    features.append(
                        sparse.coo_matrix(f)
                    )

                    # append
                    actions.append(action)
                    deltas.append(delta)
                    pss.append(ps)

        out_features = sparse.vstack(features, format='csr')
        return (
            out_features,
            np.array(actions, dtype=np.int16),
            np.array(deltas),
            np.array(pss)
        )

    def build(self):
        # print("\n\nphase1")
        features, actions, deltas, pss = self.train_data()
        pre_processor = ZNormalizer()

        # Create & train a latent mapper
        # latent_mapper = LatentMapperSVD(self.num_latent_factors, pre_processor)
        # latent_mapper = LatentMapperAE(
        #                                 self.num_latent_factors,
        #                                 pre_processor,
        #                                 self.config.num_products
        #                               )
        # latent_mapper.train(features)
        latent_mapper = None

        # Get data
        # print("\n\nphase2")

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

        # Create & train a model
        model = LogRegModel(self.config)
        if latent_mapper:
            processed_features = latent_mapper.apply(X)
        else:
            processed_features = X
        model.train(processed_features, A, y, pss)
        # print("\n\nphase3")

        return (
            LatentFeatureProvider(self.config, latent_mapper),
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
            return r.numpy()

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
            self.view_counts *= self.config.recency_bias
            self.view_counts[s['v']] += 1

    def features(self, observation):
        if self.latent_mapper:
            return self.latent_mapper.apply(self.view_counts)
        else:
            return self.view_counts

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

class LogRegModel(Basic_model):
    def __init__(self, config):
        super(LogRegModel, self).__init__(config)
        self.model = LogisticRegression(
                                        solver = 'lbfgs',
                                        multi_class = 'multinomial',
                                        max_iter=10000
                                        )
        self.reset_count = -1
        self.reset_online()

    def train(self, features, actions, clicks, pss):
        print("[Train LogRegModel]")
        self.model.fit(features, actions, sample_weight = 1 / pss)

    def act(self, observation, features):
        # X is a vector of organic counts
        features = features.reshape((1, len(features)))
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

    # def train_online(self, features, action, reward):
    #     # # don't use feature that didn't get rewarded
    #     # if not reward:
    #     #     return
    #
    #     # Update our online batch
    #     new_result = self.arange_kronecker(
    #                                         features,
    #                                         np.array([action]),
    #                                         self.config.num_products
    #                                         )
    #     self.online_matrix[self.online_count] = new_result
    #     self.online_clicks[self.online_count] = int(reward)
    #     self.online_count += 1
    #
    #     # Check if we want to do a training session
    #     if self.online_count == self.config.online_training_batch:
    #         self.model.fit(self.online_matrix.tocsr(), self.online_clicks)
    #
    #         # Reset our online batch data
    #         self.reset_online()

    # def reset_online(self):
    #     self.reset_count += 1
    #     B = self.config.online_training_batch   # batch size
    #     F = self.config.num_latent_factors      # feature size
    #     P = self.config.num_products            # product count
    #     self.online_count   = 0
    #     self.online_matrix  = sparse.dok_matrix((B, F*P), dtype=np.float32)
    #     self.online_clicks  = np.zeros(B)

# only offline training
agent = build_agent_init("AE_LogReg_Agent_Offline", TestAgent, test_agent_args)
