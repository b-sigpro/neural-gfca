import torch
from torch.distributions import RelaxedBernoulli, SigmoidTransform
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli


class ApproxBernoulli(RelaxedBernoulli):
    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        base_dist = LogitRelaxedBernoulli(temperature, probs, logits, validate_args=validate_args)
        super(RelaxedBernoulli, self).__init__(base_dist, SigmoidTransform(), validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):  # noqa
        x = super().rsample(sample_shape)
        return x - x.detach() + (x > 0.5).to(x.dtype)
