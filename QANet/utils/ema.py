# Order:
# After every training step
# 1. __call__
# Just before evaluation
# 2. assign
# Evaluate and then
# 3. resume

class EMA():

    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, model):
        # Given model, it will simply clone all trainable parameters and place
        # them in a dict as dict[param_name] = param_value. Dict is called
        # shadow
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        # calling ema(model, n) will do the following. For every parameter, if is trainable
        # it will compute new_average and assign the result to the shadow
        decay = min(self.mu, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        # using assign will place the original params in a dic and assign them
        # their current shadow values
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        # using resume will re-assign the original values to the trainable params
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]