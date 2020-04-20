from mf.abstract import *

class MyLatentMapper:
    def __init__(self, v, means, stds):
        self.v = v
        self.means = means
        self.stds = stds
    
    def apply(self, data):
        scaled = divide_or_zero(data, data.sum())
        return np.dot(
            divide_or_zero(scaled - self.means, self.stds),
            self.v.T,
        )


class MyModelBuilder(UserViewsMatrixProvider):
    def __init__(self, config):
        super(MyModelBuilder, self).__init__(config)
        self.num_latent_factors = min(
            self.config.num_latent_factors,
            self.config.num_products - 1,
        )

    def build(self):
        user_views = self.user_views()
        user_profiles = divide_or_zero(user_views, user_views.sum(axis=1))

        means = user_profiles.mean(axis=0)
        stds = user_profiles.std(axis=0)

        normalized_profiles = divide_or_zero(user_profiles - means, stds)
        _, _, v = sparse.linalg.svds(normalized_profiles, self.num_latent_factors)

        mapper = MyLatentMapper(v, means, stds)

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
    'num_latent_factors': 20,
    'random_seed': np.random.randint(2 ** 31 - 1),
}

agent = build_agent_init("MFAgent", TestAgent, test_agent_args)