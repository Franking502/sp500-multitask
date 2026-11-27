import numpy as np
from stable_baselines3 import PPO
from LTB_env import SP500Env


def rollout(model, df, cols):
    """用训练好的模型在指定数据集上跑一遍交易环境，返回每日净收益数组"""
    env = SP500Env(df, cols)
    reset_out = env.reset()
    obs = reset_out[0]
    rets, done = [], False
    while not done:
        action_out = model.predict(obs, deterministic=True)
        act = action_out[0]
        step_out = env.step(act)
        obs = step_out[0]
        # Gymnasium 返回5个值：(obs, reward, terminated, truncated, info)
        if len(step_out) == 5:
            terminated = step_out[2]
            truncated = step_out[3]
            done = terminated or truncated
            info = step_out[4]
        else:
            done = step_out[2]
            info = step_out[3]

        rets.append(info["ret"])

    return np.array(rets)

def sharpe(rets):
    return (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)

def train_ppo(train_df, val_df, cols, save_path):
    """
    训练 PPO：每 10000 步在验证集评估一次，保留验证集 Sharpe 最优的模型。
    """
    model = PPO(
        "MlpPolicy",
        env=SP500Env(train_df, cols),
        learning_rate=3e-4,  # PPO 经典的初始学习率
        n_steps=2048,  # 每次收集 2048 步的经验再做一次更新
        batch_size=64,  # 每次从收集的经验中抽 64 条训练
        n_epochs=10,  # 同批数据重复训练 10 次
        gamma=0.99,  # 折扣因子（金融数据通常设高一点，0.99）
        gae_lambda=0.95,  # GAE 优势估计的平滑参数
        clip_range=0.2,  # 核心！限制策略更新的幅度
        policy_kwargs=dict(net_arch=[128, 128]),  # 网络结构和 DQN 保持一致
        verbose=0,  # 打印训练日志
    )

    # PPO 训练步数，设在 2万-10万
    steps = list(range(20000, 100001, 20000))  # [20000, 40000, 60000, 80000, 100000]
    best_shp = -np.inf
    val_curve = []

    for step in steps:
        # PPO 的 learn 不需要 reset_num_timesteps=False，默认就是累计的
        model.learn(total_timesteps=20000, reset_num_timesteps=False)

        val_shp = sharpe(rollout(model, val_df, cols))
        val_curve.append(val_shp)

        if val_shp > best_shp:
            best_shp = val_shp
            model.save(save_path)

    # 返回最优模型
    return PPO.load(save_path), steps, val_curve