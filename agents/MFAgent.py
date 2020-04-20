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
    out = np.zeros_like(a)
    np.divide(a, b, out=out, where=b!=0)
    return out

class MFModelBuilder(ModelBuilder):
    def __init__(self, config):
        super(MFModelBuilder, self).__init__(config)
        self.num_latent_factors = min(
            self.config.num_latent_factors,
            self.config.num_products - 1,
        )

    def build(self):
        d = pd.DataFrame().from_dict(self.data)

        user_ids = d['u'].unique()
        user_nums = { u_id: num for num, u_id in enumerate(user_ids) }

        user_views = np.matrix(np.zeros((len(user_ids), self.config.num_products)))

        for _idx, organic_view in d[d['z'] == 'organic'].iterrows():
            u = user_nums[organic_view['u']]
            v = int(organic_view['v'])
            user_views[u, v] += 1

        user_profiles = divide_or_zero(user_views, user_views.sum(axis=1))

        means = user_profiles.mean(axis=0)
        stds = user_profiles.std(axis=0)

        normalized_profiles = divide_or_zero(user_profiles - means, stds)

        _, _, v = sparse.linalg.svds(normalized_profiles, self.num_latent_factors)

        mapper = LatentMapper(v, means, stds)

        characteristic_users = mapper.apply(np.eye(self.config.num_products))

        feature_provider = LatentUserFeatureProvider(self.config, mapper)
        model = ClosestProductModel(self.config, characteristic_users)
        return (feature_provider, model)


class LatentMapper:
    def __init__(self, v, means, stds):
        self.v = v
        self.means = means
        self.stds = stds
    
    def apply(self, data):
        return np.dot(
            divide_or_zero(data - self.means, self.stds),
            self.v.T,
        )

class LatentUserFeatureProvider(FeatureProvider):
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

class TestAgent(ModelBasedAgent):
    def __init__(self, config):
        super(TestAgent, self).__init__(
            config,
            MFModelBuilder(config)
        )

test_agent_args = {
    'num_products': 10,
    'num_latent_factors': 20,
    'random_seed': np.random.randint(2 ** 31 - 1),
}

agent = build_agent_init("MFAgent", TestAgent, test_agent_args)