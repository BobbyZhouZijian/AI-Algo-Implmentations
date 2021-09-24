import torch
from torch.optim import Optimizer


class LARS(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-6, weight_decay=0):
        default = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
        )
        super(LARS, self).__init__(params, default)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p.data)

                momentum = state['momentum']
                state['step'] += 1

                step_size, beta, weight_decay = group['lr'], group['beta'], group['weight_decay']

                sgd_step = grad
                if weight_decay != 0:
                    sgd_step.add_(p.data, alpha=weight_decay)

                momentum.mul_(beta).add_(grad + sgd_step, alpha=1 - beta)

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                momentum_norm = momentum.pow(2).sum().sqrt()

                if momentum_norm == 0 or weight_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / momentum_norm

                state['weight_norm'] = weight_norm
                state['momentum_norm'] = momentum_norm
                state['trust_ratio'] = trust_ratio

                p.data.add_(momentum, alpha=-trust_ratio * step_size)

        return loss
