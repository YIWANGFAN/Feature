import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


pi = torch.from_numpy(np.array(np.pi))


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output
    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.liner = nn.Linear(512,512)
    def forward(self, inputs):
        output = self.liner(inputs)
        return output
class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 bias=True):
        super(MaskedLinear, self).__init__()
        # self.normaliz = torch.nn.BatchNorm1d(in_features,eps=1e-05,momentum=0.1,affine=True)
        self.linear = nn.Linear(in_features, out_features)

        self.register_buffer('mask', mask)

    def forward(self, inputs):
        # output = self.normaliz(inputs)
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        return output


nn.MaskedLinear = MaskedLinear
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Linear = Linear()
Linear = Linear.to(device)

class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 act='relu',
                 pre_exp_tanh=False):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'PReLU': nn.PReLU, 'LeakyReLU': nn.LeakyReLU, 'sigmoid': nn.Sigmoid,
                       'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            512, 512, 512, mask_type='input')
        hidden_mask = get_mask(512, 512, 512)
        output_mask = get_mask(
            512, 512 * 2, 512, mask_type='output')

        self.lin = nn.Linear(4096,512)
        self.joiner = nn.MaskedLinear(512, 512, input_mask)

        self.trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(512, 512,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(512, 512 * 2,
                                                   output_mask))

    def forward(self, inputs, mode='direct'):
        h = self.joiner(inputs)
        m, a = self.trunk(h).chunk(2, 1)
        u = (inputs - m) * torch.exp(-a)
        return u, -a.sum(-1, keepdim=True)


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def set_class_mean(self, class_mean=None, class_mean_grad=True):
        self.class_mean = torch.nn.Parameter(class_mean, requires_grad=class_mean_grad)
        return self.class_mean

    def forward(self, inputs):
        inputs = Linear(inputs)
        logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)
        for module in self._modules.values():
            inputs, logdet = module(inputs)
            logdets += logdet
        return inputs, logdets

    def dnf_Gaussian_log_likelihood(self, inputs, mean_j, v_c):
        # print(inputs.shape)
        # forward pass
        u, logdet = self(inputs)
        class_var = torch.ones(u.shape[1], device=u.device) * (v_c ** 0.5)

        log_det_sigma = torch.log(class_var + 1e-10).sum(-1, keepdim=True)
        log_probs = -0.5 * ((torch.pow((u - mean_j), 2) / (class_var + 1e-10) + torch.log(2 * pi)).sum(-1,
                                                                                                       keepdim=True) + log_det_sigma)

        return u, -(log_probs + logdet).mean(), log_probs, logdet




