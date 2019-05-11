import torch
import torch.nn.functional as F


class ProbMaxPooling:
    def __init__(self, kernel_size, device=None):
        self.kernel_size = kernel_size
        self.device = device
        
    def sample(self, x):
        """Probabilistic Max Pooling.
        
        Identical to original implementation of CRBM.
        See also: https://github.com/honglaklee/convDBN/blob/master/sample_multrand.m
        """
        # 1 x channels x 1 x origin_length
        _, num_channels, _, num_frames = x.size()
        ks = self.kernel_size
        length = num_channels * num_frames // ks

        # extract local blocks
        x[x > 20] = 20
        x[x < -20] = -20
        x = x.exp_()
        x = x.squeeze_()
        x = x.reshape(length, ks)
        x = torch.cat([x, torch.ones(length, 1, device=self.device)], dim=1)
        # x = (x - x.max(1, keepdim=True)[0].expand_as(x))
        probs = x.div_(x.sum(1, keepdim=True))

        # sample
        cumsum = probs.cumsum(1)
        randoms = torch.rand(length, 1, device=self.device)
        temp = (cumsum > randoms).type(torch.FloatTensor)
        diff = temp[..., 1:] - temp[..., :-1]
        state = torch.zeros_like(cumsum)
        state[..., 0] = 1 - diff.sum(1)
        state[..., 1:] = diff

        # convert back to original sized matrix
        hidden_state = state[..., :-1].reshape(1, num_channels, 1, num_frames)
        hidden_probs = probs[..., :-1].reshape(1, num_channels, 1, num_frames)

        return hidden_probs, hidden_state


class CRBM:
    def __init__(self, in_channels, out_channels, kernel_size, num_features, device=None,
                 lr=1e-3, momentum=0.5, l1reg=0, l2reg=1e-2, sigma=0.2, sparsity=0.003,
                 pooling_kernel_size=2):
        self.kernel_size = kernel_size
        self.num_features = num_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l1reg = l1reg
        self.l2reg= l2reg
        self.sigma = sigma
        self.sparsity = sparsity
        self.pooling_kernel_size = pooling_kernel_size
        self.lr = lr
        self.momentum = momentum
        self.device = device

        self.W = torch.randn(out_channels, in_channels,
                             num_features, kernel_size, device=device).mul_(0.01)
        self.vbias = torch.zeros(in_channels, device=device)
        self.hbias = torch.ones(out_channels, device=device).mul_(-0.1)
        self.prob_max_pool = ProbMaxPooling(kernel_size=pooling_kernel_size, device=device)
        
        self.buffer = {}
        self.buffer['weights_momentum'] = torch.zeros_like(self.W, device=device)
        self.buffer['visible_bias_momentum'] = torch.zeros_like(self.vbias, device=device)
        self.buffer['hidden_bias_momentum'] = torch.zeros_like(self.hbias, device=device)

    def v2h(self, v):
        hidden_probs = F.conv2d(v, self.W, self.hbias).mul_(1 / self.sigma**2)
        return self.prob_max_pool.sample(hidden_probs)
        
    def h2v(self, h):
        # Important: The flip action for kernel weights is necessary in order to be identical to original
        # implementation. In Matlab conv2(), it flips the kernel first before convolution, See also: 
        # https://www.mathworks.com/matlabcentral/answers/74274-why-do-we-need-to-flip-the-kernel-in-2d-convolution
        # In its original implementation, the W for inference phase is flipped first before conv2(),
        # and in conv2(), it is flipped again, so we get result like [1, 2, 3] -> [3, 2, 1] -> [1, 2, 3]
        # But for reconstruction phase, it is flipped in conv2() only, so we get result like [1, 2, 3] -> [3, 2, 1]
        W = self.W.transpose(0, 1).flip((2, 3))
        neg_data = F.conv_transpose2d(h, self.W)
        return neg_data
    
    def cd_k(self, x, k=1):
        prob_v0 = prob_vk = x
        # positive phase
        prob_h0, act_hk = self.v2h(prob_v0)

        # Reconstruction error
        loss = F.mse_loss(self.h2v(prob_h0), prob_v0, reduction='sum') / x.size()[0]
        
        # negative phase, CD-K
        for _ in range(k):
            prob_vk = self.h2v(act_hk)
            prob_hk, act_hk = self.v2h(prob_vk)

        return prob_v0, prob_h0, prob_vk, prob_hk, loss

    def train(self, x, k=1, lr=None):
        if lr is None:
            lr = self.lr
        prob_v0, prob_h0, prob_vk, prob_hk, loss = self.cd_k(x, k)
        
        N = x.size(0)
        hidden_size = prob_h0.size(2) * prob_h0.size(3)
        visible_size = prob_v0.size(2) * prob_v0.size(3)

        # Update(with momentum)
        delta_weights = torch.zeros_like(self.W, device=self.device)
        
        for i in range(N):
            for c in range(self.in_channels):
                delta_weights[:, c] += (
                    F.conv2d(prob_v0[i, c][None, None, ...], prob_h0[i][:, None, ...]) - 
                    F.conv2d(prob_vk[i, c][None, None, ...], prob_hk[i][:, None, ...]))[0]

        # L1 & L2 reg
        delta_weights /= N * hidden_size
        delta_weights -= self.l2reg * self.W + self.l1reg * ((self.W > 0).type(self.W.type()) * 2 - 1)
        self.buffer['weights_momentum'] *= self.momentum
        self.buffer['weights_momentum'] += lr * delta_weights 

        # visible bias
        delta_vbias = (prob_v0 - prob_vk).sum((0, 2, 3)) / (N * visible_size)
        self.buffer['visible_bias_momentum'] *= self.momentum
        self.buffer['visible_bias_momentum'] += lr * delta_vbias
        
        # hidden bias with sparsity reg
        delta_hbias = (prob_h0 - prob_hk).sum((0, 2, 3)) / (N * hidden_size)
        delta_hbias += 5 * (self.sparsity - prob_h0.sum((0, 2, 3)) / (N * hidden_size))
        self.buffer['hidden_bias_momentum'] *= self.momentum
        self.buffer['hidden_bias_momentum'] += lr * delta_hbias
        
        # Update
        self.W += self.buffer['weights_momentum']
        self.vbias += self.buffer['visible_bias_momentum']
        self.hbias += self.buffer['hidden_bias_momentum']
        
        return loss


class CDBN:
    def __init__(self, layers, device=None, **global_kwargs):
        """Initialization for CDBN.
        
        Parameters
        ----------
        layers: list
            A list of tuples for CRBM, e.g. (in_channels, out_channels, kernel_size, **kwargs)
        """
        self.crbms = []
        self.device = device

        for layer in layers:
            in_channels, out_channels, kernel_size, *layer_kwargs = layer
            kwargs = global_kwargs
            if layer_kwargs:
                kwargs.update(layer_kwargs[0])
            crbm = CRBM(in_channels, out_channels, kernel_size,
                        device=device, **kwargs)
            self.crbms.append(crbm)
    
    def reconstruct(self, v, ith_layer):
        assert 0 < ith_layer <= len(self.crbms)
        act = v
        for idx in range(ith_layer):
            _, act = self.crbms[idx].v2h(act)
        for idx in range(ith_layer-1, -1, -1):
            act = self.crbms[idx].h2v(act)
        # Reconstruction error
        loss = F.mse_loss(act, v, reduction='sum') / v.size()[0]
        return act, loss

    def train(self, v, ith_layer, k=1, epoch=None):
        activation_h = v
        for idx in range(ith_layer):
            _, activation_h = self.crbms[idx].v2h(activation_h)
        crbm = self.crbms[ith_layer]
        lr = None
        if epoch is not None:
            lr = crbm.lr / (1 + epoch * 0.01)
        return crbm.train(activation_h, k=k, lr=lr)