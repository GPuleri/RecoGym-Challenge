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
    'k' : 6
}


# Model ----------------------------------------------------------------------
class TestAgent(nn.Module, Agent):
    """
    this agent perform k-Nearest Neighbours based on the organic information.
    The Bandit information on the neighbours is then leveraged to find
    the best action.
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
        self.did_click=np.empty((1,0), dtype= bool ) #did_click is true for the users which have clicked any product
        self.last_arrival_id=-1 #u index of last arrival
        self.nr_arrivals=0 #counts nr_arrivals there have been in total.

        # Create matrices we need to perform kNN
        # For all matrices the row corresponds to the index of the arrival and the
        # column corresponds to the item.
        self.M_organic=np.empty((0,self.config.num_products)) #total nr of organic views
        self.M_organic_scaled = None #A scaled version of M_organic
        self.M_bandit_clicks=np.empty((0,self.config.num_products)) #total nr of clicks
        self.M_bandit_attempts=np.empty((0,self.config.num_products)) #total nr of times we suggested an item

        # We could use the organic and bandit data of our neighbours. We associate a weight to each.
        self.weight_organic=0
        self.weight_bandit=1

        # The information about the user we are currently serving
        self.user_organic=np.zeros((self.config.num_products))
        self.user_attempts=np.zeros_like(self.user_organic)
        self.user_clicks=np.zeros_like(self.user_organic)
        
        # We use self.k number of neighbours. This can be given but we overwrite it later on.
        self.k=20

        # The model we train, currently we train it once after training, we could still do online batch training
        self.knn_model = None

        #Number of arrivals since last update for online learning
        self.num_arrivals_since_update=0;
        self.first_time=True;

        self.nr_attempts=0
        self.nr_clicks=0
        self.timer=0
        g=open('clicks.txt','a+')
        g.write("];\n c=[")
        h=open('attempts.txt','a+')
        h.write("]; \n c=[")

    def update_information(self, observation, action, reward, done):
        #### This function is used to update M_organic, M_bandit_clicks and M_bandit_attempts based on a session
        #### This function is used in the training stage.
        if observation is None:
            return
        for session in observation.current_sessions:
            user_id=session['u']
            if user_id != self.last_arrival_id:
                #When a new user arrives, we update our indx -> id and id -> indx arrays
                self.nr_arrivals+=1
                self.did_click=np.append(self.did_click, False)
                self.indx_to_id=np.append(self.indx_to_id, user_id)
                self.id_to_indx=np.append(self.id_to_indx, np.zeros((1,user_id-self.last_arrival_id), dtype=int))
                self.last_arrival_indx=self.nr_arrivals-1
                self.id_to_indx[-1]=self.last_arrival_indx
                self.last_arrival_id=user_id
            while self.last_arrival_indx>=self.M_organic.shape[0]:
                #Add new rows to M_organic, M_bandit_cliks and M_bandit_attempts until these matrices are sufficiently large
                self.M_organic=np.vstack((self.M_organic, np.zeros((1,self.config.num_products))))
                self.M_bandit_clicks=np.vstack((self.M_bandit_clicks, np.zeros((1,self.config.num_products))))
                self.M_bandit_attempts=np.vstack((self.M_bandit_attempts, np.zeros((1,self.config.num_products))))
            #Update organic information/bandit information.
            if session['z']=='pageview':
                self.M_organic[self.last_arrival_indx,session['v']] += 1
            if action:
                self.M_bandit_attempts[self.last_arrival_indx,action['a']]+=1
                if reward:
                    self.did_click[self.last_arrival_indx]=True
                    self.M_bandit_clicks[self.last_arrival_indx, action['a']]+=1

    def div(self,a,b):
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

    def update_model(self):
        ### This part is called only once, when training is done to define our kNN model.
        # We set k equal to nr of users which clicked on something divided by 10.
        self.k=int(np.floor(sum(self.did_click)/10))
        total_organic = np.sum(self.M_organic, 1)
        #Use scaled version of organic information to find neighbours
        self.M_organic_scaled = self.M_organic / total_organic[:, None]

        #Only use those users which have actually clicked something
        self.M_organic_scaled_only_clicks=self.M_organic_scaled[self.did_click,:];
        self.M_bandit_attempts_only_clicks=self.M_bandit_attempts[self.did_click,:]
        self.M_bandit_clicks_only_clicks=self.M_bandit_clicks[self.did_click,:]
        self.M_bandit_CTR_only_clicks=self.div(self.M_bandit_clicks_only_clicks,self.M_bandit_attempts_only_clicks)
        #Compute the total CTR for all items, not being used now.
        self.total_CTR=np.sum(self.M_bandit_CTR_only_clicks,0)
        
        #Define our kNN model.
        self.knn_model = NearestNeighbors(n_neighbors=self.k, algorithm='auto').fit(self.M_organic_scaled_only_clicks)

    def find_kNN_organic(self):
        if not self.knn_model:
            self.update_model();

        #This part is called every time : obtain neighbours & distances
        distances, neighbours = self.knn_model.kneighbors([self.user_organic_scaled])
        return distances, neighbours

    def pick_best_action(self, distances, nearest_neighbours):
        #### Input :
        # distances : a vector consisting of the distance to all k nearest neighbours
        # nearest_neighbours : the indices of the nearest neighbours in the set of users which have clicked something
        
        max_dist=np.max(distances)
        compl_distances=max_dist-distances #we use 1-distance to weigh the neighbours, this could be altered to f(compl_distances) for some f
        score_items=np.zeros(self.config.num_products)
        for j, neighbour in enumerate(nearest_neighbours):
            #add score absed on organic information of neighbours
            score_items+=self.weight_organic*compl_distances[j]*self.M_organic_scaled_only_clicks[neighbour,:]
            #add score based on bandit information of neighbours
            score_items+=self.weight_bandit*max(compl_distances[j],0)*self.M_bandit_CTR_only_clicks[neighbour,:]
            #add score based on bandit information of user itself
        score_items+=max_dist*np.divide(self.user_clicks, self.user_attempts, out=np.zeros_like(self.user_clicks), where=self.user_attempts!=0)
        score_items=score_items/np.sum(score_items) #rescale scores to sum to one.
        return score_items

    def act(self, observation, reward, done):
        self.timer+=1
        self.nr_attempts+=1
        if reward:
            self.nr_clicks+=1
        if self.timer==100:
            g=open('clicks.txt','a+')
            g.write(str(self.nr_clicks))
            g.write(',')
            h=open('attempts.txt','a+')
            h.write(str(self.nr_attempts))
            h.write(',')
            self.timer=0
        if reward:
            self.user_clicks[self.previous_action]+=1
        if observation is not None and len(observation.current_sessions) > 0:
            #When it is a new user : set his organic and bandit information to zero
            if (self.last_arrival_id!=observation.current_sessions[0]['u']):
                if np.sum(self.user_clicks)>0:
                    self.M_organic=np.vstack((self.M_organic, self.user_organic))
                    self.M_bandit_clicks=np.vstack((self.M_bandit_clicks, self.user_clicks))
                    self.M_bandit_attempts=np.vstack((self.M_bandit_attempts, self.user_attempts))
                    self.did_click=np.append(self.did_click,[True])
                    if np.sum(self.did_click)>400:
                        for j in range(len(self.did_click)):
                            if self.did_click[j]:
                                self.did_click[j]=False
                                break
                    self.num_arrivals_since_update+=1
                    if self.num_arrivals_since_update>50:
                        self.update_model()
                        self.num_arrivals_since_update=0
                self.user_organic=np.zeros((self.config.num_products))
                self.user_clicks=np.zeros_like(self.user_organic)
                self.user_attempts=np.zeros_like(self.user_organic)
            self.last_arrival_id=observation.current_sessions[0]['u']
            # Update organic information based on the current session
            for session in observation.current_sessions:
                self.user_organic[session['v']]+=1
        #Obtain a scaled version of the organic information
        self.user_organic_scaled=self.user_organic/np.sum(self.user_organic)
        #find the kNN
        distances, nearest_neighbours=self.find_kNN_organic()
        score=self.pick_best_action(distances[0], nearest_neighbours[0])
        #As action we simply pick the one with the highest score
        action = score.argmax().item()

        #Update the user's bandit information
        if np.random.uniform()<0.01:
            action=np.random.randint(0,self.config.num_products-1)
        self.user_attempts[action]+=1
        self.previous_action=action

        #Execute the chosen action
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
        ### Update our training data.
        self.update_information(observation, action, reward, done)