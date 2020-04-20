import numpy as np
from recogym import build_agent_init, Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)

test_agent_args = {
    'num_products': 10,
    'random_seed': np.random.randint(2 ** 31 - 1),
}

class PersonalOrganicModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(PersonalOrganicModelBuilder, self).__init__(config)

    def build(self):
        class PersonalOrganicFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super(PersonalOrganicFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class PersonalOrganicModel(Model):
            def __init__(self, config):
                super(PersonalOrganicModel, self).__init__(config)

            def act(self, observation, features):
                # Choose the item the user has already seen most often organically
                features = features.ravel()
                action = np.argmax(features)
                ps_all = np.zeros_like(features)
                ps_all[action] = 1.0
                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': 1.0,
                        'ps-a': ps_all,
                    },
                }
        
        return (
            PersonalOrganicFeaturesProvider(self.config),
            PersonalOrganicModel(self.config)
        )

class TestAgent(ModelBasedAgent):
    """
    Agent that performs the action with the user has organically seen most often already
    """
    def __init__(self, config = Configuration(test_agent_args)):
        super(TestAgent, self).__init__(
            config,
            PersonalOrganicModelBuilder(config)
        )

agent = build_agent_init('PersonalOrganicAgent', TestAgent, {**test_agent_args})
