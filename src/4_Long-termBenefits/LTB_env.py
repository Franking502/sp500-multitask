import numpy as np
import gymnasium as gym
from gymnasium import spaces

##股票交易环境：过去天数的特征、决定仓位（0、0.5/1）
class SP500Env(gym.Env):
    """股票交易环境：根据过去 window 天的特征，决定仓位（0/0.5/1）"""

    def __init__(self, df, cols, window=10, cost=0.001):
        super().__init__()
        self.X = df[cols].values.astype(np.float32)   # 特征
        self.ret = df["ret"].values.astype(np.float32)  # 每日收益
        self.window = window
        self.actions = (0.0, 0.5, 1.0)  # 仓位：空仓 / 半仓 / 满仓
        self.cost = cost
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(window * len(cols) + 1,), dtype=np.float32)

    def _obs(self):
        # 观测 = 过去 window 天特征 + 当前仓位
        past = self.X[self.t - self.window: self.t].flatten()
        return np.append(past, self.pos).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.pos = self.window, 0.0
        return self._obs(), {}

    def step(self, action):
        target = self.actions[int(action)]
        # 当日收益 = 目标仓位 * 当日收益 - 换仓成本
        r = target * self.ret[self.t] - abs(target - self.pos) * self.cost
        self.pos = target
        self.t += 1
        done = self.t >= len(self.ret) - 1
        obs = self._obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {"ret": r}
        return obs, r, done, False, info