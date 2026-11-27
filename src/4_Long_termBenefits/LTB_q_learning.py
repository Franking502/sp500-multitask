import numpy as np

##基于离散状态的表格型 Q-Learning 交易回测
##m目的：用简单的市场状态做决策，并通过验证集夏普比率防过拟合

## Close > ma_20 判断状态趋势，用 vol_21_ann > vol_med 判断是否处于高波动，提取二值状态和每日收益 ret
def _states(df, vol_med):
    """把数据转成 (趋势, 高波动) 两个状态 + 每日收益"""
    trend = (df["Close"] > df["ma_20"]).astype(int).values
    high_vol = (df["vol_21_ann"] > vol_med).astype(int).values
    return trend, high_vol, df["ret"].values

##固定一张 Q 表跑回测，每步按最优动作计算“持仓收益减换仓手续费（cost）”，并更新仓位 pos
def _evaluate(q, df, vol_med, cost):
    """给定 Q 表，跑一遍回测，返回每日收益"""
    trend, high_vol, rets = _states(df, vol_med)
    pos, daily = 0, []
    for t in range(len(df)):
        act = int(np.argmax(q[trend[t], high_vol[t]]))
        daily.append(act * rets[t] - abs(act - pos) * cost)
        pos = act
    return np.array(daily)

##按 252 个交易日计算年化夏普比率 （一年有 365 天，其中每周 7 天里只有 5 个工作日）
def _sharpe(rets):
    return (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)

##用训练集算波动率中位数防信息泄露；训练 epochs 轮，每步用 10% 探索、90% 贪心选动作，
# 按学习率 0.05、折扣因子 0.95 更新 Q 表；每轮用验证集夏普比率挑出最优 Q 表 best_q，在测试集上回测。
def run_tabular_q(train_df, val_df, test_df, cost=0.001, epochs=30):
    """训练 Q 表，用验证集挑最好的，最后在测试集回测"""
    vol_med = train_df["vol_21_ann"].median()
    ##把训练集 train_df 按 vol_med 转换成三个一维数组，并分别赋值给 t_tr、v_tr、r_tr
    t_tr, v_tr, r_tr = _states(train_df, vol_med)
    #t_tr训练集每天的趋势状态数组
    #v_tr训练集每天的高波动状态数组
    #r_tr训练集每天的收益率数组
    q = np.zeros((2, 2, 2))
    best_q = q.copy()  ##保存对应的解
    best_shp = -np.inf   ##初始化为负无穷
    for _ in range(epochs):
        pos = 0  ##空仓
        for t in range(len(train_df) - 1): #保证更新 Q 值时，t + 1 的“下一状态”一定存在、不会报错。
            ##循环到 len(train_df) - 2（即 range(len(train_df) - 1)
            s0, s1 = t_tr[t], v_tr[t]
            # 10% 概率随机探索，否则选最优
            if np.random.rand() < 0.1:
                act = np.random.choice([0, 1])
            else:
                act = int(np.argmax(q[s0, s1]))  ##其余 90% 概率选择当前状态 (s0, s1) 下 Q 值最大的动作。

            reward = act * r_tr[t] - abs(act - pos) * cost  ##（持仓收益减换仓手续费）
            ns0, ns1 = t_tr[t + 1], v_tr[t + 1]
            # Q-learning 更新公式-更新当前仓位 pos = act
            q[s0, s1, act] += 0.05 * (reward + 0.95 * np.max(q[ns0, ns1]) - q[s0, s1, act])
            pos = act
       ##按验证集夏普比率早停/选优：每轮训练集结束后，用当前 q 在 val_df 上回测并算年化夏普比率；
        # 若优于历史最高 best_shp，用 .copy() 保存这张 Q 表，缓解单一指数上的过拟合。
        val_shp = _sharpe(_evaluate(q, val_df, vol_med, cost))
        if val_shp > best_shp:
            best_shp, best_q = val_shp, q.copy()
    ##测试集输出：全部轮次结束后，用 best_q 在未参与训练与选优的 test_df 上回测，返回测试集每日收益数组。
    return _evaluate(best_q, test_df, vol_med, cost)



