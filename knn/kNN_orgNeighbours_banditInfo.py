import numpy as np
from recogym import Configuration
from recogym.agents import Agent
from torch import Tensor, nn, optim
from scipy import sparse
import sys
from scipy import spatial
from sklearn.neighbors import NearestNeighbors

# Default Arguments ----------------------------------------------------------
test_agent_args = {
    'num_products': 10,
    'embed_dim': 5,
    'mini_batch_size': 32,
    'loss_function': nn.CrossEntropyLoss(),
    'optim_function': optim.RMSprop,
    'learning_rate': 0.01,
}


# Model ----------------------------------------------------------------------
class TestAgent(nn.Module, Agent):
    """
    Organic Matrix Factorisation (Square)

    The Agent that selects an Action from the model that performs
     Organic Events matrix factorisation.
    """

    def __init__(self, config = Configuration(test_agent_args)):
        nn.Module.__init__(self)
        Agent.__init__(self, config)

        self.product_embedding = nn.Embedding(
            self.config.num_products, self.config.embed_dim
        )
        self.output_layer = nn.Linear(
            self.config.embed_dim, self.config.num_products
        )

        # Initializing optimizer type.
        self.optimizer = self.config.optim_function(
            self.parameters(), lr = self.config.learning_rate
        )

        #Users 'u' index don't all arrive therefore we use an alternative indexing which does
        #go from 0 to nr_users without skipping a number
        self.id_to_indx=np.empty((1,0), dtype= int) #u_to_indx[session['u']] gives the place at which user nr 'u' arrived
        self.indx_to_id=np.empty((1,0), dtype = int ) #indx_to_u[b] gives the 'u' number for the b'th arrival
        self.did_click=np.empty((1,0), dtype= bool )
        self.last_arrival_id=-1 #u index of last arrival
        self.nr_arrivals=0 #counts nr_arrivals there have been.

        # Create matrices we need to perform kNN
        #sparse.csr_matrix for sparse matrices
        self.M_organic=np.empty((0,self.config.num_products))
        self.M_bandit_clicks=np.empty((0,self.config.num_products))
        self.M_bandit_attempts=np.empty((0,self.config.num_products))
        self.user_organic=np.zeros((self.config.num_products))
        self.weight_organic=0
        self.weight_bandit=1
        self.k=5

        self.last_product_viewed = None
        self.curr_step = 0
        self.train_data = []
        self.action = None

        self.knn_model = None
        self.M_organic_scaled = None

    def update_information(self, observation, action, reward, done):
        if observation is None:
            return
        for session in observation.current_sessions:
            user_id=session['u']
            if user_id != self.last_arrival_id:
                self.nr_arrivals+=1
                self.did_click=np.append(self.did_click, False)
                self.indx_to_id=np.append(self.indx_to_id, user_id)
                self.id_to_indx=np.append(self.id_to_indx, np.zeros((1,user_id-self.last_arrival_id), dtype=int))
                self.last_arrival_indx=self.nr_arrivals-1
                self.id_to_indx[-1]=self.last_arrival_indx
                self.last_arrival_id=user_id
            while self.last_arrival_indx>=self.M_organic.shape[0]:
                self.M_organic=np.vstack((self.M_organic, np.zeros((1,self.config.num_products))))
                self.M_bandit_clicks=np.vstack((self.M_bandit_clicks, np.zeros((1,self.config.num_products))))
                self.M_bandit_attempts=np.vstack((self.M_bandit_attempts, np.ones((1,self.config.num_products))))
            if session['z']=='pageview':
                self.M_organic[self.last_arrival_indx,session['v']] += 1
            if action:
                self.M_bandit_attempts[self.last_arrival_indx,action['a']]+=1
                if reward:
                    self.did_click[self.last_arrival_indx]=True
                    self.M_bandit_clicks[self.last_arrival_indx, action['a']]+=1


    def find_kNN_organic(self):
        if not self.knn_model:
            total_organic = np.sum(self.M_organic, 1)
            self.M_organic_scaled = self.M_organic / total_organic[:, None]
            #self.M_organic_scaled = np.unique(self.M_organic_scaled, axis=0)
            self.M_organic_scaled_only_clicks=self.M_organic_scaled[self.did_click,:];
            self.M_bandit_attempts_only_clicks=self.M_bandit_attempts[self.did_click,:]
            self.M_bandit_clicks_only_clicks=self.M_bandit_clicks[self.did_click,:]
            self.knn_model = NearestNeighbors(n_neighbors=self.k, algorithm='auto').fit(self.M_organic_scaled_only_clicks)

        distances, indices = self.knn_model.kneighbors([self.user_organic])
        return distances, indices

    def pick_best_action(self, distances, nearest_neighbours):
        compl_sigmoid_distances=1-1/(1+np.exp(-distances))
        total_organic=np.sum(self.M_organic_scaled_only_clicks, 1)
        scaled_organic_M=self.M_organic_scaled_only_clicks/total_organic[:,None]
        M_bandit_CTR_only_clicks=self.M_bandit_clicks_only_clicks/self.M_bandit_attempts_only_clicks
        score_items=np.zeros(self.config.num_products)
        for j, neighbour in enumerate(nearest_neighbours):
            score_items+=self.weight_organic*compl_sigmoid_distances[j]*self.M_organic_scaled_only_clicks[neighbour,:]
            score_items+=self.weight_bandit*compl_sigmoid_distances[j]*M_bandit_CTR_only_clicks[neighbour,:]
        score_items=score_items/np.sum(score_items)
        return score_items

    def act(self, observation, reward, done):
        if observation is not None and len(observation.current_sessions) > 0:
            self.user_organic=np.zeros((self.config.num_products))
            for session in observation.current_sessions:
                self.user_organic[session['v']]+=1
        self.user_organic=self.user_organic/np.sum(self.user_organic)
            #self.update_information(observation,None,None,done)
            #self.last_arrival_id=observation.current_sessions[-1]['u']
            #self.last_arrival_indx=self.id_to_indx[observation.current_sessions[-1]['u']]
        distances, nearest_neighbours=self.find_kNN_organic()
        #score = np.sum(self.M_organic_scaled[nearest_neighbours[0], :], axis=0)
        score=self.pick_best_action(distances[0], nearest_neighbours[0])
        action = score.argmax().item()
        all_ps = np.zeros(self.config.num_products)
        all_ps[action] = 1.0
        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': score[action],
                'ps-a': all_ps,
            },
        }

        

    def train(self, observation, action, reward, done = False):
        """Method to deal with the """
        # Increment step.
        self.update_information(observation, action, reward, done)