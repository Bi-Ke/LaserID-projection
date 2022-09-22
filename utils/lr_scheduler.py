"""
    Function: learning rate schedulers.

    Copy from https://github.com/CoinCheung/BiSeNet

    Adding: a.WarmupCosineAnnealingLR (combine Warmup, CosineAnnealingLR)

            b.WarmupCosineAnnealingWarmRestartsLR (combine Warmup, CosineAnnealingWarmRestarts)

            c.WarmupCosineAnnealingWarmRestartsPolyLrScheduler (combine Warmup, CosineAnnealingWarmRestarts, and PolyLr)
            Here, WarmupCosineAnnealingWarmRestartsPolyLrScheduler is used to avoid getting a large learning rate
            during the last iterations.

    Date: October 27, 2021.
    Updated: September 1, 2022.
"""

import math
from bisect import bisect_right
import torch


class WarmupLrScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            ratio = self.get_main_ratio()
        return ratio

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


class WarmupPolyLrScheduler(WarmupLrScheduler):

    def __init__(
            self,
            optimizer,
            power,
            max_iter,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.power = power
        self.max_iter = max_iter
        super(WarmupPolyLrScheduler, self).__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter
        ratio = (1 - alpha) ** self.power
        return ratio


class WarmupExpLrScheduler(WarmupLrScheduler):
    def __init__(
            self,
            optimizer,
            gamma,
            interval=1,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.gamma = gamma
        self.interval = interval
        super(WarmupExpLrScheduler, self).__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        ratio = self.gamma ** (real_iter // self.interval)
        return ratio


class WarmupCosineLrScheduler(WarmupLrScheduler):
    """Note that this is different from the corresponding implementation in PyTorch."""
    def __init__(
            self,
            optimizer,
            max_iter,
            eta_ratio=0,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.eta_ratio = eta_ratio
        self.max_iter = max_iter
        super(WarmupCosineLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        # real_iter = self.last_epoch - self.warmup_iter + 1
        real_max_iter = self.max_iter - self.warmup_iter
        return self.eta_ratio + (1 - self.eta_ratio) * (1 + math.cos(math.pi * self.last_epoch / real_max_iter)) / 2


class WarmupStepLrScheduler(WarmupLrScheduler):
    def __init__(
            self,
            optimizer,
            milestones: list,
            gamma=0.1,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.milestones = milestones
        self.gamma = gamma
        super(WarmupStepLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        ratio = self.gamma ** bisect_right(self.milestones, real_iter)
        return ratio


class WarmupCosineAnnealingLR(WarmupLrScheduler):
    """combine Warmup, CosineAnnealingLR"""
    def __init__(self,
                 optimizer,
                 max_iter,  # not epochs, is iteration.
                 T_max,  # the number of cosine periods.
                 eta_ratio=0,  # the minimum ratio multiplied by the learning rate is 0.0
                 warmup_iter=500,
                 warmup_ratio=5e-4,
                 warmup='exp',
                 last_epoch=-1):
        super(WarmupCosineAnnealingLR, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)
        self.max_iter = max_iter
        self.T_max = T_max
        self.eta_ratio = eta_ratio

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter

        num_iters_per_period = real_max_iter // self.T_max
        ratio = self.eta_ratio + 0.5 * (1.0 - self.eta_ratio) * (1 + math.cos(real_iter / num_iters_per_period * math.pi))
        return ratio


class WarmupCosineAnnealingWarmRestartsLR(WarmupLrScheduler):
    """combine Warmup, CosineAnnealingWarmRestartsLR"""
    def __init__(self,
                 optimizer,
                 max_iter,  # not epochs, is iteration.
                 T_0=1000,  # the number of iterations for the first cosine period.
                 T_mult=1,  # A factor increases T_i after the following cosine periods.
                 eta_ratio=0,  # the minimum ratio multiplied by the learning rate is 0.0
                 warmup_iter=500,
                 warmup_ratio=5e-4,
                 warmup='exp',
                 last_epoch=-1):
        super(WarmupCosineAnnealingWarmRestartsLR, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)
        self.max_iter = max_iter
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_ratio = eta_ratio

    def _get_intervals(self, real_max_iter):
        intervals = []
        n = 0
        factor = 2
        # 2*T_0 >= real_max_iter
        if factor * self.T_0 >= real_max_iter:
            intervals.append(real_max_iter)
            return intervals, n

        while True:
            if factor * self.T_0 < real_max_iter:
                intervals.append([factor * self.T_0 - self.T_0, factor * self.T_0])  # down & up.
                n = n + 1
                factor += self.T_mult ** n
            else:
                intervals.append([factor * self.T_0 - self.T_0, factor * self.T_0])  # down & up.
                return intervals, n

    def cosine_down(self, relative_real_iter, T_i):
        ratio = self.eta_ratio + 0.5 * (1.0 - self.eta_ratio) * (1 + math.cos(relative_real_iter / T_i * math.pi))
        return ratio

    def warm_up(self, relative_real_iter):
        alpha = relative_real_iter / self.T_0
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.eta_ratio) * alpha
            return ratio
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
            return ratio

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        intervals, n = self._get_intervals(real_max_iter)
        if n == 0:
            T_i = intervals[0]
            ratio = self.eta_ratio + 0.5 * (1.0 - self.eta_ratio) * (1 + math.cos(real_iter / T_i * math.pi))
            return ratio
        else:
            for idx, interval in enumerate(intervals):
                if (idx == 0) & (real_iter < interval[0]):  # Down during the first interval.
                    T_i = interval[0]
                    return self.cosine_down(relative_real_iter=real_iter, T_i=T_i)

                elif (idx == 0) & (interval[0] <= real_iter) & (real_iter < interval[1]):  # Up during the first interval.
                    return self.warm_up(relative_real_iter=real_iter-interval[0])

                elif (idx != 0) & (real_iter < interval[0]):  # Down in other intervals.
                    T_i = interval[0] - intervals[idx-1][1]
                    return self.cosine_down(relative_real_iter=real_iter-intervals[idx-1][1], T_i=T_i)

                elif (idx != 0) & (interval[0] <= real_iter < interval[1]):
                    return self.warm_up(relative_real_iter=real_iter-interval[0])  # Down in other intervals.


class WarmupCosineAnnealingWarmRestartsPolyLrScheduler(WarmupLrScheduler):
    """combine Warmup, CosineAnnealingWarmRestarts, and PolyLrScheduler"""
    def __init__(self,
                 optimizer,
                 max_iter,  # not epochs, is iteration.
                 inter_iter,  # [0, inter_iter), using CosineAnnealingWarmRestarts; [inter_iter, -1], using PolyLr.
                 power,
                 T_0=1000,  # the number of iterations for the first cosine period.
                 T_mult=1,  # A factor increases T_i after the following cosine periods.
                 eta_ratio=0,  # the minimum ratio multiplied by the learning rate is 0.0
                 warmup_iter=500,
                 warmup_ratio=5e-4,
                 warmup='exp',
                 last_epoch=-1):
        super(WarmupCosineAnnealingWarmRestartsPolyLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)
        self.max_iter = max_iter
        if inter_iter is None:
            self.inter_iter = max_iter // 2
        else:
            self.inter_iter = inter_iter
            assert inter_iter < max_iter, "inter_iter must be smaller than max_iter."
        self.power = power
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_ratio = eta_ratio

    def _get_intervals(self, real_inter_iter):
        intervals = []
        n = 0
        factor = 2
        # 2*T_0 >= real_max_iter
        if factor * self.T_0 >= real_inter_iter:
            intervals.append(real_inter_iter)
            return intervals, n

        while True:
            if factor * self.T_0 < real_inter_iter:
                intervals.append([factor * self.T_0 - self.T_0, factor * self.T_0])  # down & up.
                n = n + 1
                factor += self.T_mult ** n
            else:
                intervals.append([factor * self.T_0 - self.T_0, factor * self.T_0])  # down & up.
                return intervals, n

    def cosine_down(self, relative_real_iter, T_i):
        ratio = self.eta_ratio + 0.5 * (1.0 - self.eta_ratio) * (1 + math.cos(relative_real_iter / T_i * math.pi))
        return ratio

    def warm_up(self, relative_real_iter):
        alpha = relative_real_iter / self.T_0
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.eta_ratio) * alpha
            return ratio
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
            return ratio

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_inter_iter = self.inter_iter - self.warmup_iter

        if real_iter < real_inter_iter:  # WarmupCosineAnnealingWarmRestarts
            intervals, n = self._get_intervals(real_inter_iter)
            if n == 0:
                T_i = intervals[0]
                self.ratio = self.eta_ratio + 0.5 * (1.0 - self.eta_ratio) * (1 + math.cos(real_iter / T_i * math.pi))
                return self.ratio
            else:
                for idx, interval in enumerate(intervals):
                    if (idx == 0) & (real_iter < interval[0]):  # Down during the first interval.
                        T_i = interval[0]
                        self.ratio = self.cosine_down(relative_real_iter=real_iter, T_i=T_i)
                        return self.ratio

                    elif (idx == 0) & (interval[0] <= real_iter) & (real_iter < interval[1]):  # Up during the first interval.
                        self.ratio = self.warm_up(relative_real_iter=real_iter-interval[0])
                        return self.ratio

                    elif (idx != 0) & (real_iter < interval[0]):  # Down in other intervals.
                        T_i = interval[0] - intervals[idx-1][1]
                        self.ratio = self.cosine_down(relative_real_iter=real_iter-intervals[idx-1][1], T_i=T_i)
                        return self.ratio

                    elif (idx != 0) & (interval[0] <= real_iter < interval[1]):
                        self.ratio = self.warm_up(relative_real_iter=real_iter-interval[0])  # Down in other intervals.
                        return self.ratio
        else:  # PolyLr
            real_iter = real_iter - real_inter_iter
            real_max_iter = self.max_iter - self.inter_iter
            alpha = real_iter / real_max_iter
            poly_ratio = self.ratio * (1 - alpha) ** self.power
            return poly_ratio


if __name__ == "__main__":
    model = torch.nn.Conv2d(3, 16, 3, 1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2)

    # max_iter = 145200  # 119600
    # max_iter = 290100  # 119600
    # max_iter = 145050
    max_iter = 193400
    # lr_scheduler = WarmupPolyLrScheduler(optim, 0.9, max_iter, 1000, 0.1, 'linear', -1)
    # lr_scheduler = WarmupPolyLrScheduler(optim, 0.9, max_iter, 1000, 0.1, 'exp', -1)
    # lr_scheduler = WarmupCosineLrScheduler(optim, max_iter, eta_ratio=0, warmup_iter=1000, warmup_ratio=5e-4,
    #                                        warmup='exp', last_epoch=-1,)
    # lr_scheduler = WarmupCosineAnnealingLR(optim, max_iter, T_max=5, eta_ratio=0, warmup_iter=1000, warmup_ratio=5e-4,
    #                                        warmup='exp', last_epoch=-1,)
    # lr_scheduler = WarmupCosineAnnealingWarmRestartsLR(optim, max_iter, T_0=1000, T_mult=2,  eta_ratio=0,
    #                                                    warmup_iter=1000, warmup_ratio=5e-4, warmup='exp', last_epoch=-1)
    lr_scheduler = WarmupCosineAnnealingWarmRestartsPolyLrScheduler(optim, max_iter, inter_iter=40000, power=0.9,
                                                                    T_0=1000, T_mult=2,  eta_ratio=0, warmup_iter=1000,
                                                                    warmup_ratio=5e-4, warmup='exp', last_epoch=-1)
    lrs = []
    for _ in range(max_iter):
        lr = lr_scheduler.get_lr()[0]
        print(lr)
        lrs.append(lr)
        lr_scheduler.step()

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    lrs = np.array(lrs)
    n_lrs = len(lrs)
    plt.plot(np.arange(n_lrs), lrs)
    plt.grid()
    plt.show()

    # print(lr_scheduler.state_dict())
    # new_lr_scheduler = WarmupCosineLrScheduler(optim, max_iter, eta_ratio=0, warmup_iter=1000, warmup_ratio=5e-4, warmup='exp', last_epoch=-1,)
    # new_lr_scheduler.load_state_dict(lr_scheduler.state_dict())
    # print(new_lr_scheduler.state_dict())

