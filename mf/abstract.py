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

import math

import numpy as np
import pandas as pd

import scipy
from scipy import sparse
from scipy.spatial.distance import cdist

def divide_or_zero(a, b):
    """ Divide a by b, returning 0 when b was 0. """
    out = np.zeros_like(a)
    np.divide(a, b, out=out, where=b!=0)
    return out

class UserViewsMatrixProvider(ModelBuilder):
    """ Base model builder class """
    def __init__(self, config):
        super(UserViewsMatrixProvider, self).__init__(config)
        self.user_ids = []
        self.user_nums = {}
    
    def user_views(self):
        d = pd.DataFrame().from_dict(self.data)

        self.user_ids = d['u'].unique()
        self.user_nums = { u_id: num for num, u_id in enumerate(self.user_ids) }

        user_views = np.matrix(np.zeros((len(self.user_ids), self.config.num_products)))

        for _idx, organic_view in d[d['z'] == 'organic'].iterrows():
            u = self.user_nums[organic_view['u']]
            v = int(organic_view['v'])
            user_views[u, v] += 1
        
        return user_views

class LatentMapper:
    """ Maps a user view vector to a latent representation """
    def apply(self, data):
        raise NotImplementedError

class LatentUserFeatureProvider(FeatureProvider):
    """ Produces a feature vector by applying a LatentMapper to
        a product view vector """
    def __init__(self, config, mapper):
        super(LatentUserFeatureProvider, self).__init__(config)
        self.mapper = mapper
        self.reset()
    
    def observe(self, observation):
        for s in observation.sessions():
            self.view_counts[s['v']] += 1
        
    def features(self, _observation):
        return self.mapper.apply(self.view_counts)
    
    def reset(self):
        self.view_counts = np.zeros(self.config.num_products)


class ClosestProductModel(Model):
    """ Recommends the product that is closest to the provided features
    """
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

class CharacteristicUserModel(ClosestProductModel):
    """ Encodes a 'characteristic user' that is only interested in one product and uses this
        to represent a product. Then acts like the ClosestProductModel. """
    def __init__(self, config, mapper, metric='cosine'):
        characteristic_users = mapper.apply(np.eye(config.num_products))
        super(CharacteristicUserModel, self).__init__(config, characteristic_users, metric)