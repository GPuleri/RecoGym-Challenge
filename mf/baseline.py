from mf.abstract import *

class IdentityMapper:
    def apply(self, data):
        return data


class MyModelBuilder(ModelBuilder):
    def __init__(self, config):
        super(MyModelBuilder, self).__init__(config)

    def build(self):
        mapper = IdentityMapper()

        return (
            LatentUserFeatureProvider(self.config, mapper),
            CharacteristicUserModel(self.config, mapper),
        )

class TestAgent(ModelBasedAgent):
    def __init__(self, config):
        super(TestAgent, self).__init__(
            config,
            MyModelBuilder(config)
        )

test_agent_args = {
    'num_products': 10,
    'random_seed': np.random.randint(2 ** 31 - 1),
}

agent = build_agent_init("MFAgent", TestAgent, test_agent_args)