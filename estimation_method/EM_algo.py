#%%
import numpy as np
from functools import partial
import sys
sys.path.append('..')
import Make_Data 
#from Make_Data import L2P, L3P

# 数値計算で発散させないように小さなεをおいておく
epsilon =  0.0001
# 受験者パラメータの取りうる範囲を定義する。
X_k = np.linspace(-4, 4, 41)
# 受験者パラメータの分布を定義する。ここでは、scipyの正規分布を使う。
from scipy.stats import norm
g_k = norm.pdf(X_k)
# E stepの関数
def get_exp_params(irm_ij, g_k, P3_ik):
    Lg_jk = np.exp(irm_ij.T.dot(np.log(P3_ik)) + (1 - irm_ij).T.dot(np.log(1 - P3_ik)))* g_k
    n_Lg_jk = Lg_jk / Lg_jk.sum(axis=1)[:, np.newaxis]
    f_k = n_Lg_jk.sum(axis=0)
    r_ik = irm_ij.dot(n_Lg_jk)
    return f_k, r_ik
# M step用のスコア関数
def score_(param, f_k, r_k, X_k):
    a, b, c = param
    P3_k = partial(Make_Data.L3P, a, b, c)(X_k)
    P2_k = partial(Make_Data.L2P, a, b)(X_k)
    R_k = r_k / P3_k - f_k
    v = [
        ((X_k - b) * R_k * P2_k).sum(),
        - a * (R_k * P2_k).sum(),
        R_k.sum() / (1 - c)
    ]
    return np.linalg.norm(v)

from scipy.optimize import minimize

# 2pl用に書き直す必要あり
def EM_algo(irm_ij, item_params_):
    # minimize用の制約条件を定義する。
    cons_ = {
        'type': 'ineq',
        'fun': lambda x:[
            x[0] - 0.1,
            2 - x[0],
            x[1] - (-2),
            2 - x[1],
            x[2] - c_min,
            c_max - x[2],
        ]
    }
    # 初期parameter生成用のparameterを用意する。
    a_min, a_max = 0.1, 8.0
    b_min, b_max = -4.0, 4.0
    c_min, c_max = epsilon, 0.6

    num_items, num_users = irm_ij.shape

    # 推定実行用のoarameter
    # EM algorithmの繰り返し終了条件
    delta = 0.001
    # EM algorithmの繰り返し最大回数
    max_num_of_itr = 1000

    # 数値安定のために何度か計算して、安定したものの中の中央値を採用する
    p_data = []
    for n_try in range(10):
        """
        # 推定の初期値を定義する。 ← 別関数で定義?
        item_params_ = np.array(
            [np.random.uniform(a_min, a_max, num_items),
            np.random.uniform(b_min, b_max, num_items),
            np.random.uniform(c_min, c_max, num_items)]
        ).T
        """
        prev_item_params_ = item_params_
        for itr in range(max_num_of_itr):
            # E step : exp paramの計算 
            P3_ik = np.array([partial(Make_Data.L3P, *ip)(X_k) for ip in item_params_])
            f_k, r_ik = get_exp_params(irm_ij, g_k, P3_ik)
            ip_n = []
            # 各問題ごとに最適化問題をとく
            for item_id in range(num_items):
                target = partial(score_, f_k=f_k, r_k=r_ik[item_id], X_k=X_k)
                result = minimize(target, x0=item_params_[item_id], constraints=cons_, method="slsqp")         
                ip_n.append(list(result.x))

            item_params_ = np.array(ip_n)
            # 前回との平均差分が一定値を下回ったら計算終了
            mean_diff = abs(prev_item_params_ - item_params_).sum() / item_params_.size
            if mean_diff < delta:
                break
            prev_item_params_ = item_params_

        p_data.append(item_params_)

    p_data_ = np.array(p_data)
    result_ = []
    for idx in range(p_data_.shape[1]):
        t_ = np.array(p_data)[:, idx, :]
        # 計算結果で極端なものを排除
        filter_1 = t_[:, 1] < b_max - epsilon 
        filter_2 = t_[:, 1] > b_min + epsilon
        # 残った中のmedianを計算結果とする。
        result_.append(np.median(t_[filter_1 & filter_2], axis=0))

    result = np.array(result_)

    return result
# %%
