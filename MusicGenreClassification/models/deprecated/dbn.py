import torch
import torch.nn.functional as F


class GBRBM:
    def __init__(self, n_visible, n_hidden, device=None,
                 lr=1e-3, momentum=0.5, weight_decay=2e-4):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.device = device
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.W = torch.randn(n_visible, n_hidden, device=device).mul_(0.1)
        self.visible_biases = torch.zeros(n_visible, device=device)
        self.sigmas = torch.zeros(n_visible, device=device)
        self.hidden_biases = torch.zeros(n_hidden, device=device)
        
        self.buffer = {}
        self.buffer['weights_momentum'] = torch.zeros(n_visible, n_hidden, device=device)
        self.buffer['visible_bias_momentum'] = torch.zeros(n_visible, device=device)
        self.buffer['hidden_bias_momentum'] = torch.zeros(n_hidden, device=device)

    def v2h(self, v):
        v = v / self.sigmas.pow(2)
        pos_hid_probs = F.linear(v, self.W.t(), self.hidden_biases)
        return torch.sigmoid(pos_hid_probs)
        
    def h2v(self, h):
        v = F.linear(h, self.W, self.visible_biases)
        return v
    
    def cd_k(self, v0, k=1):
        h0 = self.v2h(v0)
        hk = h0

        for _ in range(k):
            hk = torch.bernoulli(hk)
            vk = self.h2v(hk)
            hk = self.v2h(vk)

        return v0, h0, vk, hk

    def train(self, v0, k=1, lr=None):
        if lr is None:
            lr = self.lr
        v0, h0, vk, hk = self.cd_k(v0, k)
        
        sq_sigmas = self.sigmas.pow(2)
        vis0 = v0.sum() / sq_sigmas
        hid0 = h0.sum()
        w0 = torch.mm(v0.t(), h0) / self.n_visible
        w0 = w0 / sq_sigmas
        sigma0 = (v0 - self.visible_biases).pow_(2)
        
        

        # Gradients
        self.buffer['weights_momentum'] *= self.momentum
        self.buffer['weights_momentum'] += lr * (
            w0 - torch.mm(prob_vk.t(), prob_hk) - self.W * self.weight_decay)
        
        self.buffer['visible_bias_momentum'] *= self.momentum
        self.buffer['visible_bias_momentum'] += lr * torch.sum(prob_v0 - prob_vk, 0)
        
        self.buffer['hidden_bias_momentum'] *= self.momentum
        self.buffer['hidden_bias_momentum'] += lr * torch.sum(prob_h0 - prob_hk, 0)

        # Update
        self.W += self.buffer['weights_momentum']
        self.a += self.buffer['visible_bias_momentum']
        self.b += self.buffer['hidden_bias_momentum']

        # Reconstruction error
        return F.mse_loss(prob_vk, prob_v0, reduction='sum') / x.size()[0]

class DBN():
    def __init__(self, n_visible, n_hiddens, device=None, **global_kwargs):
        assert isinstance(n_hiddens, list)
        self.rbms = []

        for idx in range(len(n_hiddens)):
            if idx == 0:
                n_vis = n_visible
            else:
                n_vis = n_hiddens[idx-1]
            n_hid = n_hiddens[idx]
            kwargs = global_kwargs
            rbm = GBRBM(n_vis, n_hid, device=device, **kwargs)
            self.rbms.append(rbm)

    def reconstruct(self, v, ith_layer):
        assert 0 < ith_layer <= len(self.rbms)
        act = v
        for idx in range(ith_layer):
            _, act = self.rbms[idx].v2h(act)
        for idx in range(ith_layer-1, -1, -1):
            _, act = self.rbms[idx].h2v(act)
        # Reconstruction error
        loss = F.mse_loss(act, v, reduction='sum') / v.size()[0]
        return act, loss

    def train(self, v, ith_layer, k=1, epoch=None):
        activation_h = v
        for idx in range(ith_layer):
            _, activation_h = self.rbms[idx].v2h(activation_h)
        lr = None
        rbm = self.rbms[ith_layer]
        if epoch is not None:
            lr = rbm.lr / (1 + epoch * 0.01)
        return rbm.train(activation_h, k=k, lr=lr)