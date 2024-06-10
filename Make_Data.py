#%%
import numpy as np
from functools import partial
from scipy.stats import truncnorm
import pandas as pd
np.random.seed(314)
# 数値計算で発散させないように小さなεをおいておく
epsilon =  0.0001
# 3 parameter logistic model の定義
def L3P(a, b, c, x):
    return c + (1 - epsilon - c) / (1 + np.exp(-  a * (x - b)))

# 2 parameter logistic model の定義。処理の統一のためにcも引数に取ることとする。
def L2P(a, b, x):
    return (1 - epsilon) / (1 + np.exp(-  a * (x - b)))


def make_data(num_items, num_users):

    # model parameterの定義
    # aは正の実数, bは実数, cは0より大きく1未満であれば良い

    a_min = 0.1#0.7
    a_max = 2.0#4

    b_min = -2
    b_max = 2

    c_min = 0
    c_max = .4

    theta_min = -2
    theta_max = 2

    # 何問、何人にするか、下なら10問4000人
    #num_items = 100
    #num_users = 5000

    rng = np.random.default_rng()

    theta_mean = 0
    theta_std = 1.0 # 標準偏差
    # 標準化する。 (scipy.stats.truncnorm のドキュメントの Notes 参照)
    theta_min, theta_max= (theta_min - theta_mean) / theta_std, (theta_max - theta_mean) / theta_std
    # 確率変数から観測値を10000個得る
    para_theta = truncnorm.rvs(theta_min, theta_max, loc=theta_mean, scale=theta_std, size=num_users)

    b_mean = 0
    b_std = np.sqrt(1.0) # 標準偏差
    # 標準化する。 (scipy.stats.truncnorm のドキュメントの Notes 参照)
    b_min, b_max = (b_min - b_mean) / b_std, (b_max - b_mean) / b_std
    para_b = truncnorm.rvs(b_min, b_max, loc=b_mean, scale=b_std, size=num_items)
    #para_x = np.random.normal(0, 1, size=1*N)#正規分布からとってきた
    
    # aの真値
    a_mean = -0.5
    a_std = 0.2
    # 対数正規分布から(data, datafilterdのは全部これ)
    #para_a = rng.lognormal(a_mean, a_std, size=num_items) # 対数正規分布から
    para_a = np.random.uniform(a_min, a_max, num_items)
    # 問題parameterの生成
    """ # 一様分布から
    item_params = np.array(
        [np.random.uniform(a_min, a_max, num_items),
        np.random.uniform(b_min, b_max, num_items),
        np.random.uniform(c_min, c_max, num_items)]
    ).T
    """
    # a: 対数正規分布, b: 正規分布から
    item_params = np.array(
        [para_a,
         para_b]
    ).T

    # 受験者parameterの生成
    #user_params = np.random.normal(size=num_users)

    # 項目反応行列の作成、 要素は1(正答)か0(誤答)
    # i行j列は問iに受験者jがどう反応したか
    ir_matrix_ij = np.vectorize(int)(
        np.array(
            [partial(L2P, *ip)(para_theta) > np.random.uniform(0, 1, num_users) for ip in item_params]
        )
    )
    
    filter_a = ir_matrix_ij.sum(axis=1) / num_users < 0.9
    filter_b = ir_matrix_ij.sum(axis=1) / num_users > 0.1
    filter_c = np.corrcoef(np.vstack((ir_matrix_ij, ir_matrix_ij.sum(axis=0))))[num_items][:-1] >= 0.3
    filter_total = filter_a & filter_b & filter_c
    #print(filter_total)
    # 項目反応行列を再定義する
    irm_ij = ir_matrix_ij[filter_total]
    para_a = para_a[filter_total]
    para_b = para_b[filter_total]
    
    return irm_ij, para_a, para_b, para_theta
# %% シミュレーションによるデータ
def simdata():
    df = pd.read_table('https://raw.githubusercontent.com/trycycle/pymc_irt/main/dataset/item_response.tsv', sep='\t', header=0, index_col=0)
    df.head()
    sim_data = df.to_numpy()
    print(sim_data)

    item_df = pd.read_table('https://raw.githubusercontent.com/trycycle/pymc_irt/main/dataset/item_params.tsv', sep='\t', header=0)
    item_df.index = [f'Q{qid}' for qid in range(1, 21)]
    # 設問の真の難易度
    b_real = item_df.b.values

    a_real = item_df.a.values

    # 回答者の能力パラメータ
    participant_df = pd.read_table('https://raw.githubusercontent.com/trycycle/pymc_irt/main/dataset/participant_params.tsv', sep='\t', header=0)
    theta_real = participant_df.theta.values
    
    return sim_data.T, a_real, b_real, theta_real
# %%
# 通過率の極端な項目、合計点との相関性が低すぎる項目を除く
def make_data_filterd(num_items, num_users):

    # model parameterの定義
    # aは正の実数, bは実数, cは0より大きく1未満であれば良い
    a_min = 0.1#0.7
    a_max = 2.0#4
    b_min = -2.0
    b_max = 2.0
    
    rng = np.random.default_rng()

    para_theta = np.random.randn(num_users)

    para_b = np.random.randn(num_items)
    #para_x = np.random.normal(0, 1, size=1*N)#正規分布からとってきた
    
    # aの真値
    a_mean = -0.5 # -0.75
    a_std = 0.2
    para_a = rng.lognormal(a_mean, a_std, size=num_items) # 対数正規分布から
    # 問題parameterの生成
    # 一様分布から
    #para_a = np.random.uniform(a_min, a_max, num_items)
    """
    item_params = np.array(
        [np.random.uniform(a_min, a_max, num_items),
        np.random.uniform(b_min, b_max, num_items)]
    ).T
    """
    # a: 対数正規分布, b: 正規分布から
    item_params = np.array(
        [para_a,
        para_b]
    ).T
    
    # 受験者parameterの生成
    #user_params = np.random.normal(size=num_users)

    # 項目反応行列の作成、 要素は1(正答)か0(誤答)
    # i行j列は問iに受験者jがどう反応したか
    ir_matrix_ij = np.vectorize(int)(
        np.array(
            [partial(L2P, *ip)(para_theta) > np.random.uniform(0, 1, num_users) for ip in item_params]
        )
    )

    # Check if conditions are met
    filter_a = ir_matrix_ij.sum(axis=1) / num_users < 0.9
    filter_b = ir_matrix_ij.sum(axis=1) / num_users > 0.1
    filter_c = np.corrcoef(np.vstack((ir_matrix_ij, ir_matrix_ij.sum(axis=0))))[num_items][:-1] >= 0.3
    filter_total = filter_a & filter_b & filter_c
    #print(filter_total)
    # 項目反応行列を再定義する
    irm_ij = ir_matrix_ij[filter_total]
    para_a = para_a[filter_total]
    para_b = para_b[filter_total]
    
    return irm_ij, para_a, para_b, para_theta
#%%
from scipy.stats import bernoulli, pearsonr

def generate_binary_item_responses(num_items, num_subjects):
    item_responses = np.zeros((num_items, num_subjects), dtype=int)

    for i in range(num_items):
        while True:
            # Generate random item parameters
            pass_rate = np.random.uniform(0.1, 0.9)
            score_correlation = np.random.uniform(0.3, 1.0)
            
            # Generate random scores
            scores = np.random.binomial(1, pass_rate, size=num_subjects)
            
            # Check correlation with scores
            corr, _ = pearsonr(scores, np.arange(num_subjects))
            
            # Check if conditions are met
            if pass_rate >= 0.1 and pass_rate <= 0.9 and corr >= score_correlation:
                item_responses[i] = scores
                break

    return item_responses

# Example usage

# %%
