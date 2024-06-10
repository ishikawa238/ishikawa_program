#%%
import arviz as az
import pandas as pd
import numpy as np
import pymc3 as pm
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse # 楕円を描く
#import seaborn as sns
#colors = sns.color_palette().as_hex()

# %% pymc3を用いたmcmc法による推定
def mcmc(df):
    # データ、パラメタのDataFrame化
    # tidy data化
    response_df = pd.melt(df.reset_index(),
                        id_vars='participant', var_name='question', value_name='response')
    #print(response_df.head())
    # participants[participant_idx] is equal to response_df['participant']
    participant_idx, participants = pd.factorize(response_df['participant'])

    # questions[question_idx] is equal to response_df['question']
    question_idx, questions = pd.factorize(response_df['question'])
    
    # モデル化

    coords = {'participant': participants, 'question': questions}
    model = pm.Model(coords=coords)

    with model:  
        # 個人の能力（回答者ごとに割り当て）
        theta = pm.Normal('theta', mu=0, sigma=1, dims='participant')
        # 問題の識別力（設問ごとに割り当て）変えなきゃいけないかも
        a = pm.HalfNormal('a', sigma=1, dims='question')
        # 問題の難易度（設問ごとに割り当て）
        b = pm.Normal('b', mu=0, sigma=1, dims='question')
        
        # 2パラメータロジスティックモデル（2PLM）
        p = 1 / (1 + pm.math.exp(-1 * a[question_idx] * (theta[participant_idx] - b[question_idx])))

        y_obs = pm.Bernoulli('y_obs', p=p, observed=response_df.response)
    
    with model:
        # デフォルトだとサンプリング数が1000なので，多めに
        # デフォルトはギブスサンプリング?
        #step = pm.Metropolis() # メトロポリス・ヘイスティングス
        #step = pm.HamiltonianMC() # ハミルトニアンモンテカルロ
        idata = pm.sample(draws=1000, progressbar=True, return_inferencedata=False)
        #idata = pm.sample(draws=1000, step = pm.HamiltonianMC(), progressbar=True, return_inferencedata=False)
        #print(pm.summary(idata))
        
    
    summary = az.summary(idata, round_to=3)

    a_predicted = summary['a[0]':'a['+str(df.shape[1]-1)+']']['mean'].values
    b_predicted = summary['b[0]':'b['+str(df.shape[1]-1)+']']['mean'].values
    theta_predicted = summary['theta[0]':'theta['+str(df.shape[0]-1)+']']['mean'].values

    est_params = np.array(
        [a_predicted,
         b_predicted,
        ]
    ).T

    return est_params, theta_predicted, idata
# %%
# %% pymc3を用いたmcmc法による推定
def hmc(df):
    # データ、パラメタのDataFrame化
    # tidy data化
    response_df = pd.melt(df.reset_index(),
                        id_vars='participant', var_name='question', value_name='response')
    #print(response_df.head())
    # participants[participant_idx] is equal to response_df['participant']
    participant_idx, participants = pd.factorize(response_df['participant'])

    # questions[question_idx] is equal to response_df['question']
    question_idx, questions = pd.factorize(response_df['question'])
    
    # モデル化

    coords = {'participant': participants, 'question': questions}
    model = pm.Model(coords=coords)

    with model:  
        # 個人の能力（回答者ごとに割り当て）
        theta = pm.Normal('theta', mu=0, sigma=1, dims='participant')
        # 問題の識別力（設問ごとに割り当て）変えなきゃいけないかも
        a = pm.HalfNormal('a', sigma=1, dims='question')
        # 問題の難易度（設問ごとに割り当て）
        b = pm.Normal('b', mu=0, sigma=1, dims='question')
        
        # 2パラメータロジスティックモデル（2PLM）
        p = 1 / (1 + pm.math.exp(-1 * a[question_idx] * (theta[participant_idx] - b[question_idx])))

        y_obs = pm.Bernoulli('y_obs', p=p, observed=response_df.response)
    
    with model:
        # デフォルトだとサンプリング数が1000なので，多めに
        # デフォルトはギブスサンプリング?
        #step = pm.Metropolis() # メトロポリス・ヘイスティングス
        #step = pm.HamiltonianMC() # ハミルトニアンモンテカルロ
        idata = pm.sample(draws=1000, step = pm.HamiltonianMC(), progressbar=True, return_inferencedata=False)
        #idata = pm.sample(draws=1000, step = pm.HamiltonianMC(), progressbar=True, return_inferencedata=False)
        #print(pm.summary(idata))
        
    
    summary = az.summary(idata, round_to=3)

    a_predicted = summary['a[0]':'a['+str(df.shape[1]-1)+']']['mean'].values
    b_predicted = summary['b[0]':'b['+str(df.shape[1]-1)+']']['mean'].values
    theta_predicted = summary['theta[0]':'theta['+str(df.shape[0]-1)+']']['mean'].values

    est_params = np.array(
        [a_predicted,
         b_predicted,
        ]
    ).T

    return est_params, theta_predicted, idata

#%%
def ADVI(df):
    # データ、パラメタのDataFrame化
    # tidy data化
    response_df = pd.melt(df.reset_index(),
                        id_vars='participant', var_name='question', value_name='response')
    #print(response_df.head())
    # participants[participant_idx] is equal to response_df['participant']
    participant_idx, participants = pd.factorize(response_df['participant'])

    # questions[question_idx] is equal to response_df['question']
    question_idx, questions = pd.factorize(response_df['question'])
    
    # モデル化

    coords = {'participant': participants, 'question': questions}
    model = pm.Model(coords=coords)

    with model:  
        # 個人の能力（回答者ごとに割り当て）
        theta = pm.Normal('theta', mu=0, sigma=1, dims='participant')
        # 問題の識別力（設問ごとに割り当て）変えなきゃいけないかも
        a = pm.HalfNormal('a', sigma=1, dims='question')
        # 問題の難易度（設問ごとに割り当て）
        b = pm.Normal('b', mu=0, sigma=1, dims='question')
        
        # 2パラメータロジスティックモデル（2PLM）
        p = 1 / (1 + pm.math.exp(-1 * a[question_idx] * (theta[participant_idx] - b[question_idx])))

        y_obs = pm.Bernoulli('y_obs', p=p, observed=response_df.response)
    
    with model:
        # ADVIによる推論
        # 10000回
        idata = pm.fit(n=10000, obj_optimizer=pm.adagrad(learning_rate=1e-1))
        #idata = pm.sample(draws=1000, step = pm.HamiltonianMC(), progressbar=True, return_inferencedata=False)
        #print(pm.summary(idata))
        
    #fig = plt.figure(figsize=(7, 4))
    #ax = fig.subplots(1,1)

    #ax.plot(idata.hist)
    #ax.set_xlabel('iteration')
    #ax.set_ylabel('ELBO')
    #print(idata)
    #summary = az.summary(idata, round_to=3)
    #print(idata.hist)

    sample_post_advi = idata.sample(1000)
    #print(sample_post_advi.varnames)
    a_predicted = sample_post_advi['a'].mean(axis=0)
    b_predicted = sample_post_advi['b'].mean(axis=0)
    theta_predicted = sample_post_advi['theta'].mean(axis=0)

    est_params = np.array(
        [a_predicted,
         b_predicted,
        ]
    ).T
    
    return est_params, theta_predicted, idata