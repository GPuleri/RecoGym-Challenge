import numpy as np
from recogym import Configuration
from recogym.agents import Agent
from torch import Tensor, nn, optim
from scipy import sparse
import sys
from scipy import spatial

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
        self.last_arrival_id=-1 #u index of last arrival
        self.nr_arrivals=0 #counts nr_arrivals there have been.

        # Create matrices we need to perform kNN
        #sparse.csr_matrix for sparse matrices
        self.M_organic=np.empty((0,self.config.num_products))
        self.M_bandit_clicks=np.empty((0,self.config.num_products))
        self.M_bandit_attempts=np.empty((0,self.config.num_products))
        self.user_organic=np.zeros((self.config.num_products))
        self.weight_organic=0.7
        self.weight_bandit=0.3
        self.k=3

        self.last_product_viewed = None
        self.curr_step = 0
        self.train_data = []
        self.action = None

    def update_information(self, observation, action, reward, done):
        if observation is None:
            return
        for session in observation.current_sessions:
            user_id=session['u']
            if user_id != self.last_arrival_id:
                self.nr_arrivals+=1
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
                    self.M_bandit_clicks[self.last_arrival_indx, action['a']]+=1

    def find_kNN_organic(self):
        #user_indx=self.last_arrival_indx
        total_organic=np.sum(self.M_organic, 1)
        M_organic_scaled=self.M_organic/total_organic[:,None]
        tree = spatial.KDTree(M_organic_scaled)
        #An issue is that the user itself is one of its neighbours, but maybe this is also good?
        nearest_neighbours=tree.query(self.user_organic,self.k)
        return nearest_neighbours

    def pick_best_action(self, distances, nearest_neighbours):
        compl_sigmoid_distances=1-1/(1+np.exp(-distances))
        total_organic=np.sum(self.M_organic, 1)
        scaled_organic_M=self.M_organic/total_organic[:,None]
        M_bandit_CTR=self.M_bandit_clicks/self.M_bandit_attempts
        score_items=np.zeros(self.config.num_products)
        for j, neighbour in enumerate(nearest_neighbours):
            score_items+=self.weight_organic*compl_sigmoid_distances[j]*scaled_organic_M[neighbour,:]

            #Add to items for Bandit information
            score_items+=self.weight_bandit*compl_sigmoid_distances[j]*M_bandit_CTR[neighbour,:]
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
        score_items=self.pick_best_action(distances, nearest_neighbours)

        action = score_items.argmax().item()
        all_ps = np.zeros(self.config.num_products)
        all_ps[action] = 1.0
        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': score_items[action],
                'ps-a': all_ps,
            },
        }

        

    def train(self, observation, action, reward, done = False):
        """Method to deal with the """
        # Increment step.
        self.update_information(observation, action, reward, done)