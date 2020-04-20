import numpy as np
from numpy.random import choice
from numpy import linalg
from recogym.agents import Agent

import scipy

test_agent_args = {
    'random_seed': 20,
    'num_products': 10,
    'num_latent_factors': 5,
    'num_users': 1000,
    'recency_bias': 0.75,
    # "online_training": False,
    # "online_training_batch": 1000,
}


# Define an Agent class.
class TestAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        # organic_user_views should be of size (config.U, config.P), but for some reason number of test_users != config.U
        # so we add a new row every time we encounter a new user_id

        # print(self.config.__dict__.keys())

        self.rowCount = 0
        self._user_indices = {}
        self.organic_user_views = scipy.sparse.dok_matrix((self.rowCount, config.num_products), dtype="float")

        self.products_mat = None

    def train(self, observation, action, reward, done):
        """Train method learns from a tuple of data.
            this method can be called for offline or online learning"""
        # action, is a number between 0 and num_products - 1 that references the index of the product recommended.
        # reward is 0 if the user does not click on the recommended product and 1 if they do. Notice that when a user clicks on a product
        # done is a True/False flag indicating if the episode (aka user's timeline) is over.
        # info currently not used, so it is always an empty dictionary.
        if observation:
            # print("observation", len(observation.sessions()))
            for s in observation.sessions():
                # s = [t,u, z, v]
                #   t = timestep
                #   u = user id
                #   z = organic / bandit
                #   v = product id viewed
                u = self.get_index(s['u']) #problem :user increases sometimes with 2 instead of 1
                p = s['v']
                self.organic_user_views[u] *= self.config.recency_bias
                self.organic_user_views[u, p] += 1

    def act(self, observation, reward, done):
        """Act method returns an action based on current observation and past
            history"""

        if self.products_mat is None:
            # first time we act, compute svd and products_mat
            # size(u) = (#users, K)
            # size(s) = (K)
            # size(vt) = (K, #products)
            u, s, vt = scipy.sparse.linalg.svds(self.organic_user_views, self.config.num_latent_factors)
            self.products_mat = np.dot(vt.T, vt)

        # create a products vector for the observed user
        product_vec = np.zeros(self.config.num_products)
        for s in observation.sessions():
            product_vec *= self.config.recency_bias
            product_vec[s['v']] += 1

        # select best action
        prediction = np.dot(product_vec, self.products_mat)
        action = np.argmax(prediction)#get index of best prediction

        # assign a probability to each action
        # in this case, propose best match with 100% probability, all other 0%
        prob = np.zeros(self.config.num_products)
        prob[action] = 1.0

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': prob[action],
                'ps-a': prob,
            }
        }

    def get_index(self, user_id):
        """translate user id to row index"""
        if not user_id in self._user_indices:
            # if not encountered user_id:
            #   add a new row to the user-product matrix
            #   assign last row to user_id
            self._user_indices[user_id] = self.rowCount
            self.rowCount += 1
            self.organic_user_views.resize((self.rowCount, self.config.num_products))
        return self._user_indices[user_id]
