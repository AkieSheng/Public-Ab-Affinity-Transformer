# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/8/20 20:47
@author: LiFan Chen
@Filename: lookahead.py
@Software: PyCharm
"""
from collections import defaultdict
from itertools import chain
from torch.optim.optimizer import Optimizer
import torch
import warnings


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        # 初始化外部优化器
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state

        # 初始化defaults属性，包含外部优化器的默认参数
        self.defaults = self.optimizer.defaults.copy()  # 继承外部优化器的默认设置
        
        # 初始化_optimizer_step_pre_hooks和_optimizer_step_post_hooks
        self._optimizer_step_pre_hooks = {}  # 初始化前置钩子
        self._optimizer_step_post_hooks = {}  # 初始化后置钩子
        
        # 为每个param_group增加"counter"属性
        for group in self.param_groups:
            group["counter"] = 0


    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)
