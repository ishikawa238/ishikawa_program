# %%
import numpy as np
import math
import arviz as az
import pandas as pd
from scipy import stats
np.random.seed(314)
#%%
# function eta
def eta(x):
  import numpy as np
  I = x.shape[0]
  J = x.shape[1]
  eta = np.zeros([I, J])
  for i in range(I):
    for j in range(J):
      #eta_1[i][j] = eta(old_eps[i][j])
      y = (np.exp(x[i][j])) / (np.exp(x[i][j])+1) - 1/2
      eta[i][j] = y/(2*x[i][j])
  #for i in range(I):
  #  for j in range(J):
      if np.abs(eta[i][j]) < 0.01:
        eta[i][j] = 0.125
  return eta

#%% ミルズ比を出す
def milz(x):
  # X>xの場合、元の正規分布のパラメタmean, sd
  norm = stats.norm(loc=0,scale=1).pdf(x)
  cdf_norm =stats.norm(loc=0,scale=1).cdf(x)
  R = (1-cdf_norm) / (norm+0.0001)
  for j in range(len(R)):
    if R[j] < 0.001:
      R[j] = 0.001
  return R

# q(a)の期待値を計算
def culc_Ea(mean_a, sd_a):
   R = milz(-mean_a/sd_a)
   #print("R", R)
   Ea = mean_a + (sd_a/R)
   Ea2 = mean_a**2 + (2*mean_a*sd_a/R) - (sd_a**2/R)*(mean_a/sd_a) + sd_a**2
   return Ea, Ea2

#%%
# aを切断正規分布に近似
def vb_new(old_A, old_B, old_eps, Y):
  N = Y.shape[0] # 受検者の人数
  J = Y.shape[1] # 項目数
  converged = 0
  old_eta = eta(old_eps)
  #print("1")
  t = 0
  sig_x = np.ones(N)
  mu_x = np.zeros(N)
  sd_a = np.zeros(J)
  mu_a = old_A #np.zeros(J)
  sig_b = np.ones(J)
  mu_b = old_B #np.zeros(J)
  for j in range(J):
    sd_a[j] = 0.05
    #mu_a[j] = 0.5

  while(converged==0):

    old_A = mu_a
    old_B = mu_b
    old_X = mu_x

    # aの期待値を先に用意(old_Aの期待値)
    Ea, Ea2 = culc_Ea(mu_a, sd_a)
    print("Ea, Ea2\n", Ea, Ea2)
    """
    # 他の分布の計算においては↓これでいいのかも?
    Ea = mu_a
    Ea2 = mu_a**2 + sd_a**2
    print("Ea, Ea2 その2\n", Ea, Ea2)
    """
    new_eps = np.zeros([N, J])
    # 変分分布q_xの更新(平均と分散の更新)
    for i in range(N):
      sigma_sum = 0
      mu_sum = 0
      sigma_sum = np.sum(old_eta[i] * Ea2)
      mu_sum = np.sum((Y[i] - 0.5) * Ea - 2 * old_eta[i] * mu_b * Ea2)
      
      # θの事前分布をN(0, 1)としている
      sigma_i_hat = 1 / (1 + 2 * sigma_sum) 
      mu_i_hat = sigma_i_hat * mu_sum
      sig_x[i] = sigma_i_hat
      mu_x[i] = mu_i_hat

      # M-step
      # εijの更新
      eps_tmp = Ea2*(sig_x[i]**2+mu_x[i]**2-2*mu_x[i]*mu_b+mu_b**2+sig_b**2)
      new_eps[i] = np.sqrt(eps_tmp)

    new_eta = eta(new_eps)
    print("new_eps", new_eps)
    print("mu_x", mu_x)
    print("nea_eta", new_eta)
    # qa, qbの更新(こっちさき？)
    
    # Bの更新
    sum_b1 = np.sum(new_eta*Ea2, axis=0)
    sum_b2 = np.sum((-Y+0.5)*Ea + 2*np.vstack([mu_x]*J).T*new_eta[i][j]*Ea2, axis=0)

    # Aの更新
    sum_a1 = np.sum(new_eta*(np.vstack([mu_x**2]*J).T + np.vstack([sig_x**2]*J).T - 2*np.vstack([mu_x]*J).T*mu_b + mu_b**2 + sig_b**2), axis=0)
    #sum_a2 = np.sum(Y*np.vstack([mu_x]*J).T + (mu_b-np.vstack([mu_x]*J).T)/2, axis=0)
    sum_a2 = np.sum((Y-0.5)*(np.vstack([mu_x]*J).T-mu_b), axis=0)

    # a, bの事前分布は a ~ 切断正規分布だけどパラメタ的にはN(0.5, 0.1), b ~ N(0, 1)
    sig_b = 1/(1 + 2*sum_b1)
    mu_b = sig_b*sum_b2

    # 元の正規分布のパラメタ？
    sig_a_pre = 1/(1/0.1 + 2*sum_a1)
    mu_a_pre = sig_a_pre*(0.5/0.1 + sum_a2)

    #切断正規分布に直したパラメタ
    mu_a, sd_a = culc_Ea(mu_a_pre, np.sqrt(sig_a_pre))


    print("mu_a\n", mu_a_pre, mu_a)
    print("sig_a\n", np.sqrt(sig_a_pre), sd_a)
    print("mu_b\n", mu_b)   
    # 収束判定
    M = np.sqrt(np.sum((mu_a-old_A)**2)+np.sqrt(np.sum((mu_b-old_B)**2)))
    MA = np.sqrt(np.sum((mu_a-old_A)**2))
    MB = np.sqrt(np.sum((mu_b-old_B)**2))
    if M < 0.0001:
      converged = 1
    print("M\n", M)
    print("MA\n", MA)
    print("MB\n", MB)
    print("")
    # 値の更新
    #old_A = new_A #np.array(new_A).T
    #old_B = new_B
    old_eps = new_eps 
    old_eta = new_eta

    # 最大反復回数まで収束しなかったら打ち切り
    t += 1
    if t > 1000:
      converged = 1
      print("1000回打ち切り")

  return mu_a, sd_a, mu_b, sig_b, mu_x, sig_x 

#%% aを切断正規分布に設定
def vb_new_base(old_A, old_B, old_eps, Y):
  N = Y.shape[0] # 受検者の人数
  J = Y.shape[1] # 項目数
  converged = 0
  old_eta = eta(old_eps)
  #print("1")
  t = 0
  sig_x = np.ones(N)
  mu_x = np.zeros(N)
  sd_a = np.zeros(J)
  mu_a = old_A #np.zeros(J)
  sig_b = np.ones(J)
  mu_b = old_B #np.zeros(J)
  # 元の正規分布のパラメタ？
  sig_a_pre = np.zeros(J)
  mu_a_pre = np.zeros(J)
  for j in range(J):
    sd_a[j] = 0.05
    #mu_a[j] = 0.5

  Ea, Ea2 = culc_Ea(mu_a, sd_a)

  while(converged==0):

    old_A = mu_a
    old_B = mu_b
    old_X = mu_x

    # aの期待値を先に用意(old_Aの期待値)
    #print("milz\n", milz(-mu_a/sd_a))
    #Ea, Ea2 = culc_Ea(mu_a, sd_a)
    """
    # 他の分布の計算においては↓これでいいのかも?
    Ea = mu_a
    Ea2 = mu_a**2 + sd_a**2
    print("Ea, Ea2 その2\n", Ea, Ea2)
    """

    # 変分分布q_xの更新(平均と分散の更新)
    # ↓最初だから初期化してok
    sig_x = np.ones(N)
    mu_x = np.zeros(N)
    for i in range(N):
      sigma_sum = 0
      mu_sum = 0
      sigma_sum = np.sum(old_eta[i] * Ea2)
      mu_sum = np.sum((Y[i] - 0.5) * Ea + 2 * old_eta[i] * mu_b * Ea2)
      
      # θの事前分布をN(0, 1)としている
      sigma_i_hat = 1 / (1 + 2 * sigma_sum) 
      mu_i_hat = sigma_i_hat * mu_sum
      sig_x[i] = sigma_i_hat
      mu_x[i] = mu_i_hat

    #print("mu_x", mu_x)

    # qaの更新(こっちさき？)
    sig_a_pre = np.zeros(J)
    mu_a_pre = np.zeros(J)
    for j in range(J): 
        sum_a1 = 0
        sum_a2 = 0
        for i in range(N):
            # Aの更新
            sum_a1 += old_eta[i][j]*(mu_x[i]**2 + sig_x[i] - 2*mu_x[i]*mu_b[j] + mu_b[j]**2 + sig_b[j])
            sum_a2 += (Y[i][j]-0.5)*(mu_x[i] - mu_b[j])

        sig_a_hat = 1/(1/0.1 + 2*sum_a1)
        mu_a_hat = sig_a_hat*(0.5/0.1+sum_a2)

        mu_a_pre[j] = mu_a_hat
        sig_a_pre[j] = sig_a_hat
    
    mu_a = mu_a_pre
    sd_a = np.sqrt(sig_a_pre)
    #print("mu_a\n", mu_a_pre, mu_a)
    #print("sig_a\n", np.sqrt(sig_a_pre), sd_a)
    # a更新したので期待値も更新
    #Ea, Ea2 = culc_Ea(mu_a, sd_a)
    Ea = mu_a
    Ea2 = mu_a**2 + sd_a**2
    #print("Ea, Ea2\n", Ea, Ea2)

    # qbの更新
    new_sig_b = np.ones(J)
    new_mu_b = np.zeros(J)
    for j in range(J): 
        sum_b1 = 0
        sum_b2 = 0
        for i in range(N):
            # Bの更新
            sum_b1 += old_eta[i][j]*Ea2[j]
            sum_b2 += (-Y[i][j]+0.5)*Ea[j] + 2*mu_x[i]*old_eta[i][j]*Ea2[j]

        sig_b_hat = 1/(1 + 2*sum_b1)
        mu_b_hat = sig_b_hat*(sum_b2)


        #if mu_b_hat > 2.5:
        #   mu_b_hat = 2.5

        #if mu_b_hat < -2.5:
        #   mu_b_hat = -2.5

        new_mu_b[j] = mu_b_hat
        new_sig_b[j] = sig_b_hat


    #切断正規分布に直したパラメタ 
    mu_b = new_mu_b
    sig_b = new_sig_b
    #print("mu_b\n", mu_b)  

    # Mstep ここで発散している ←　bがでかすぎるのが原因臭い
    new_eps = np.zeros([N, J])
    for i in range(N):
      # M-step
      # εijの更新
      eps_tmp = Ea2*(sig_x[i]+mu_x[i]**2-2*mu_x[i]*mu_b+mu_b**2+sig_b)
      for j in range(J):
        if eps_tmp[j] < 0:
           #print("it is not plus")
           eps_tmp = abs(eps_tmp)
      new_eps[i] = np.sqrt(eps_tmp)

    new_eta = eta(new_eps)
    #print("new_eps", new_eps)
    #print("nea_eta", new_eta)

    # 収束判定
    M = np.sqrt(np.sum((mu_a-old_A)**2)+np.sqrt(np.sum((mu_b-old_B)**2)))
    MA = np.sqrt(np.sum((mu_a-old_A)**2))
    MB = np.sqrt(np.sum((mu_b-old_B)**2))
    #Meps = np.sqrt(np.sum((new_eps-old_eps)**2))
    if M < 0.0001:
      converged = 1
    #print("M\n", M)
    #print("MA\n", MA)
    #print("MB\n", MB)
    #print("Meps\n", Meps)
    #print("")
    # 値の更新
    #old_A = new_A #np.array(new_A).T
    #old_B = new_B
    old_eps = new_eps 
    old_eta = new_eta

    # 最大反復回数まで収束しなかったら打ち切り
    t += 1
    if t > 10000:
      converged = 1
      print("10000回打ち切り")

  #mu_a, sd_a = culc_Ea(mu_a_pre, np.sqrt(sig_a_pre))
  #print("eps\n", old_eps)
  #print("aの分布のパラメタ確認")
  #print(mu_a-mu_a_pre)
  #print(sd_a-np.sqrt(sig_a_pre))
  return mu_a, sd_a, mu_b, sig_b, mu_x, sig_x 

#%% aをガンマ分布で近似
def vb_new_gamma(old_A, old_B, old_eps, Y):
  N = Y.shape[0] # 受検者の人数
  J = Y.shape[1] # 項目数
  converged = 0
  old_eta = eta(old_eps)
  #print("1")
  t = 0
  sig_x = np.ones(N)
  mu_x = np.zeros(N)
  sd_a = np.zeros(J)
  mu_a = old_A #np.zeros(J)
  sig_b = np.ones(J)
  mu_b = old_B #np.zeros(J)
  # 事前ガンマ分布のパラメータ
  k = 100
  lamda_pre = 1
  lamda = np.ones(J)

  old_eps2 = np.full((N, J), 0.5)

  for j in range(J):
    lamda[j] = 200
    #mu_a[j] = 0.5

  while(converged==0):

    old_A = mu_a
    old_B = mu_b
    old_X = mu_x
    old_lamda = lamda
    # 変分分布q_xの更新(平均と分散の更新)
    # ↓最初だから初期化してok
    sig_x = np.ones(N)
    mu_x = np.zeros(N)
    for i in range(N):
      sigma_sum = 0
      mu_sum = 0
      sigma_sum = np.sum(old_eta[i] * (k / old_lamda**2))
      mu_sum = np.sum((Y[i] - 0.5) * (k / old_lamda) + 2 * old_eta[i] * mu_b * (k / old_lamda**2))
      
      # θの事前分布をN(0, 1)としている
      sigma_i_hat = 1 / (1 + 2 * sigma_sum) 
      mu_i_hat = sigma_i_hat * mu_sum
      sig_x[i] = sigma_i_hat
      mu_x[i] = mu_i_hat

    #print("mu_x", mu_x)
    mu_a = np.zeros(J)
    lamda = np.zeros(J)
    print("old_eps2\n", old_eps2)
    # qaの更新(こっちさき？)
    for j in range(J): 
        sum_a1 = 0
        sum_a2 = 0
        for i in range(N):
            # Aの更新
            sum_a1 += (mu_x[i] - mu_b[j])*(Y[i][j]-(1-np.tanh(old_eps[i][j]/2))/2)
            # 別の近似ver
            sum_a2 += (mu_x[i] - mu_b[j])*(Y[i][j]-old_eps2[i][j])

        #if sum_a1 < 0:
           #print("sum_a1 is not plus")
           #sum_a1 = abs(sum_a1)
        
        #print("sum_a\n", sum_a1, sum_a2)
        lamda[j] = abs(lamda_pre - sum_a2)
        mu_a_hat = k / lamda[j]
        mu_a[j] = mu_a_hat

    print("mu_a\n", mu_a)
    #print("sig_a\n", np.sqrt(sig_a_pre), sd_a)
    #print("Ea, Ea2\n", Ea, Ea2)

    # qbの更新
    new_sig_b = np.ones(J)
    new_mu_b = np.zeros(J)
    for j in range(J): 
        sum_b1 = 0
        sum_b2 = 0
        for i in range(N):
            # Bの更新
            sum_b1 += old_eta[i][j]*(k / lamda[j]**2)
            sum_b2 += (-Y[i][j]+0.5)*(k / lamda[j]) + 2*mu_x[i]*old_eta[i][j]*(k / lamda[j]**2)

        sig_b_hat = 1/(1 + 2*sum_b1)
        mu_b_hat = sig_b_hat*(sum_b2)

        if mu_b_hat > 2.5:
           mu_b_hat = 2.5

        if mu_b_hat < -2.5:
           mu_b_hat = -2.5

        new_mu_b[j] = mu_b_hat
        new_sig_b[j] = sig_b_hat


    #切断正規分布に直したパラメタ 
    mu_b = new_mu_b
    sig_b = new_sig_b
    #print("mu_b\n", mu_b)  

    # Mstep ここで発散している ←　bがでかすぎるのが原因臭い
    new_eps = np.zeros([N, J])
    new_eps2 = np.zeros([N, J])
    for i in range(N):
      # M-step
      # εijの更新
      eps_tmp = (k / lamda**2)*(sig_x[i]+mu_x[i]**2-2*mu_x[i]*mu_b+mu_b**2+sig_b)

      # ε'の更新
      exp2 = np.exp((k/lamda)*(mu_x[i]-mu_b))
      new_eps2[i] = exp2 / (1+exp2)

      for j in range(J):
        if eps_tmp[j] < 0:
           #print("it is not plus")
           eps_tmp = abs(eps_tmp)

      new_eps[i] = np.sqrt(eps_tmp)

      


    new_eta = eta(new_eps)
    #print("new_eps", new_eps)
    #print("nea_eta", new_eta)

    # 収束判定
    M = np.sqrt(np.sum((mu_a-old_A)**2)+np.sqrt(np.sum((mu_b-old_B)**2)))
    MA = np.sqrt(np.sum((mu_a-old_A)**2))
    MB = np.sqrt(np.sum((mu_b-old_B)**2))
    Meps = np.sqrt(np.sum((new_eps-old_eps)**2))
    if Meps < 0.0001:
      converged = 1
    print("M\n", M)
    print("MA\n", MA)
    print("MB\n", MB)
    #print("Meps\n", Meps)
    #print("")
    # 値の更新
    #old_A = new_A #np.array(new_A).T
    #old_B = new_B
    old_eps = new_eps 
    old_eta = new_eta
    print(new_eps2-old_eps2)
    old_eps2 = new_eps2

    # 最大反復回数まで収束しなかったら打ち切り
    t += 1
    if t > 1000:
      converged = 1
      print("1000回打ち切り")

  #print("aの分布のパラメタ確認")
  #print(mu_a-mu_a_pre)
  #print(sd_a-np.sqrt(sig_a_pre))
  return mu_a, mu_b, sig_b, mu_x, sig_x 

#%% θとbを変分事後分布で推定、他はMステップで
# aが正しく推定できていない
def vb_new_xb(old_A, old_B, old_eps, Y, real_params, real_theta):
  N = Y.shape[0] # 受検者の人数
  J = Y.shape[1] # 項目数
  converged = 0
  old_eta = eta(old_eps)
  #print("1")
  t = 0
  sig_x = np.ones(N)
  mu_x = np.zeros(N)
  sig_b = np.ones(J)
  mu_a = np.zeros(J)
  mu_b = old_B #np.zeros(J)
  # 元の正規分布のパラメタ？
  sig_a_pre = np.zeros(J)
  mu_a_pre = np.zeros(J)
  Bias = np.zeros(3)
  RMSE = np.zeros(3)
  RMSE_a = []
  RMSE_b = []
  RMSE_theta = []
  Bias_a = []
  Bias_b = []
  Bias_theta = []

  while(converged==0):

    #old_A = mu_a
    old_B = mu_b
    sig_x = np.ones(N)
    mu_x = np.zeros(N)
    new_eps = np.zeros([N, J])
    #old_x = mu_x
    # 変分分布q_xの更新(平均と分散の更新)
    for i in range(N):
      sigma_sum = 0
      mu_sum = 0
      sigma_sum = np.sum(old_eta[i] * old_A**2)
      mu_sum = np.sum((Y[i] - 0.5) * old_A + 2 * old_eta[i] * mu_b * old_A**2)
      
      # θの事前分布をN(0, 1)としている
      sigma_i_hat = 1 / (1 + 2 * sigma_sum) 
      mu_i_hat = sigma_i_hat * mu_sum
      sig_x[i] = sigma_i_hat
      mu_x[i] = mu_i_hat

      # M-step
      # εijの更新
    for i in range(N):
      eps_tmp = old_A**2 * (sig_x[i]+mu_x[i]**2-2*mu_x[i]*mu_b+mu_b**2+sig_b)
      new_eps[i] = np.sqrt(eps_tmp)

    #Mx = np.sqrt(np.sum((new_mu_x-old_x)**2))
    #sig_x = new_sig_x
    new_eta = eta(new_eps)
    #rint("new_eps", new_eps)
    #print("mu_x", mu_x)
    #print("nea_eta", new_eta)
    # qbの更新(こっちさき？)
    new_sig_b = np.ones(J)
    new_mu_b = np.zeros(J)
    for j in range(J): 
        sum_b1 = 0
        sum_b2 = 0
        for i in range(N):
            # Bの更新
            sum_b1 += new_eta[i][j] * old_A[j]**2
            sum_b2 += (-Y[i][j]+0.5)*old_A[j] + 2*mu_x[i]*new_eta[i][j]*old_A[j]**2

        sig_b_hat = 1/(1 + 2*sum_b1)
        mu_b_hat = sig_b_hat*(sum_b2)

        new_mu_b[j] = mu_b_hat
        new_sig_b[j] = sig_b_hat

    mu_b = new_mu_b
    sig_b = new_sig_b

    # aの更新
    new_A = np.zeros(J)
    for j in range(J): 
        sum_a1 = 0
        sum_a2 = 0
        for i in range(N):        
            # Aの更新
            sum_a1 += 2*new_eta[i][j]*(mu_x[i]**2 + sig_x[i] - 2*mu_x[i]*mu_b[j] + mu_b[j]**2 + sig_b[j])
            sum_a2 += (Y[i][j]-0.5)*(mu_x[i] - mu_b[j])

        new_A[j] = sum_a2 / sum_a1

    #print("mu_a\n", new_A)
    #print("mu_b, sig_b\n", mu_b, sig_b)   
    # 収束判定
    M = np.sqrt(np.sum((new_A-old_A)**2)+np.sqrt(np.sum((mu_b-old_B)**2)))
    MA = np.sqrt(np.sum((new_A-old_A)**2))
    MB = np.sqrt(np.sum((mu_b-old_B)**2))
    Me = np.sqrt(np.sum((new_eps-old_eps)**2))
    #print("かくにん\n",mu_a,old_A)
    if M < 0.0001:
      converged = 1
    #print("M\n", M)
    #print("反復回数%d, MA ",t, MA)
    #print("\n")
    #print("(MB, Me, Mx)", MB, Me)
    #print("MB\n", MB)
    #print("")
    # 値の更新
    old_A = new_A #np.array(new_A).T
    #old_B = new_B
    old_eps = new_eps 
    old_eta = new_eta
    #mu_x = new_mu_x

    # 最大反復回数まで収束しなかったら打ち切り
    t += 1
    if t > 10000:
      converged = 1
      print("10000回打ち切り")

    # 反復ごとのRMSEを出したい
    # とりあえず１回の推定で導出する体で
    RMSE[0] = np.sqrt(sum(np.power(real_params.T[0] - new_A,  2)) / J )
    RMSE[1] = np.sqrt(sum(np.power(real_params.T[1] - mu_b,  2)) / J )
    RMSE[2] = np.sqrt(sum(np.power(real_theta - mu_x,  2)) / N )
    Bias[0] = sum(real_params.T[0] - new_A) / J
    Bias[1] = sum(real_params.T[1] - mu_b) / J   
    Bias[2] = sum(real_theta - mu_x) / N   

    RMSE_a.append(RMSE[0])
    RMSE_b.append(RMSE[1])
    RMSE_theta.append(RMSE[2])
    Bias_a.append(Bias[0])
    Bias_b.append(Bias[1])
    Bias_theta.append(Bias[2])

  prog_accuracy_item = np.array(
    [RMSE_a,
    RMSE_b,
    Bias_a,
    Bias_b]
  ).T
  prog_accuracy_user = np.array(
     [RMSE_theta,
      Bias_theta]
  ).T
  #print(t)
  return old_A, mu_b, sig_b, mu_x, sig_x, prog_accuracy_item, prog_accuracy_user 

#%% プログラムを高速に推定
def vb_new_xb_fast(old_A, old_B, old_eps, Y):
  N = Y.shape[0] # 受検者の人数
  J = Y.shape[1] # 項目数
  converged = 0
  old_eta = eta(old_eps)
  #print("1")
  t = 0
  sig_x = np.ones(N)
  mu_x = np.zeros(N)
  sig_b = np.ones(J)
  mu_a = np.zeros(J)
  mu_b = old_B #np.zeros(J)
  # 元の正規分布のパラメタ？
  sig_a_pre = np.zeros(J)
  mu_a_pre = np.zeros(J)
  Bias = np.zeros(3)
  RMSE = np.zeros(3)
  RMSE_a = []
  RMSE_b = []
  RMSE_theta = []
  Bias_a = []
  Bias_b = []
  Bias_theta = []

  while(converged==0):

    old_B = mu_b
    # 変分分布q_xの更新(平均と分散の更新)
    # θの事前分布をN(0, 1)としている
    sig_x = 1 / (1 + 2*np.sum(old_eta*np.array([old_A**2]*N), axis=1) )
    mu_x = sig_x * (np.sum((2*old_eta*old_A*mu_b + Y - 0.5) * old_A, axis=1))

    # M-step
    # εijの更新
    eps_tmp = old_A**2 * (np.array([sig_x+mu_x**2]*J).T - 2*np.array([mu_x]*J).T*mu_b+np.array([mu_b**2+sig_b]*N))
    new_eps = np.sqrt(eps_tmp)
    new_eta = eta(new_eps)

    # qbの更新(こっちさき？)
    sig_b = 1 / (2*np.sum(new_eta*np.array([old_A**2]*N), axis=0) + 1)
    mu_b = sig_b * (np.sum(2*np.multiply(new_eta.T, mu_x).T * old_A - Y + 0.5, axis=0) * old_A)

    # aの更新
    #new_A = np.zeros(J)
    mat = np.sum(new_eta * ( np.array([sig_b + mu_b**2]*N) - np.multiply((np.array([2 * mu_b]*N)).T, mu_x).T + np.array([sig_x+mu_x**2]*J).T), axis=0)
    vec = np.sum((np.multiply((Y - 0.5).T, mu_x).T + np.multiply((0.5 - Y), mu_b)), axis=0)
    new_A = vec / (2 * mat)
 
    # 収束判定
    M = np.sqrt(np.sum((new_A-old_A)**2)+np.sqrt(np.sum((mu_b-old_B)**2)))
    MA = np.sqrt(np.sum((new_A-old_A)**2))
    MB = np.sqrt(np.sum((mu_b-old_B)**2))
    Me = np.sqrt(np.sum((new_eps-old_eps)**2))
    #print("かくにん\n",mu_a,old_A)
    if M < 0.0001:
      converged = 1
    # 値の更新
    old_A = new_A #np.array(new_A).T
    #old_B = new_B
    old_eps = new_eps 
    old_eta = new_eta
    #mu_x = new_mu_x

    # 最大反復回数まで収束しなかったら打ち切り
    t += 1
    if t > 10000:
      converged = 1
      print("10000回打ち切り")

    # 反復ごとのRMSEを出したい
    # とりあえず１回の推定で導出する体で
  #   RMSE[0] = np.sqrt(sum(np.power(real_params.T[0] - new_A,  2)) / J )
  #   RMSE[1] = np.sqrt(sum(np.power(real_params.T[1] - mu_b,  2)) / J )
  #   RMSE[2] = np.sqrt(sum(np.power(real_theta - mu_x,  2)) / N )
  #   Bias[0] = sum(real_params.T[0] - new_A) / J
  #   Bias[1] = sum(real_params.T[1] - mu_b) / J   
  #   Bias[2] = sum(real_theta - mu_x) / N   

  #   RMSE_a.append(RMSE[0])
  #   RMSE_b.append(RMSE[1])
  #   RMSE_theta.append(RMSE[2])
  #   Bias_a.append(Bias[0])
  #   Bias_b.append(Bias[1])
  #   Bias_theta.append(Bias[2])

  # prog_accuracy_item = np.array(
  #   [RMSE_a,
  #   RMSE_b,
  #   Bias_a,
  #   Bias_b]
  # ).T
  # prog_accuracy_user = np.array(
  #    [RMSE_theta,
  #     Bias_theta]
  # ).T
  
  #print(t)
  return old_A, mu_b, sig_b, mu_x, sig_x, t #, prog_accuracy_item, prog_accuracy_user 


#%%  θとbを変分事後分布で推定+θとbの事前分布のパラメタも更新(経験ベイズ)
def vb_new_xb_ebl(old_A, old_B, old_eps, Y):
  N = Y.shape[0] # 受検者の人数
  J = Y.shape[1] # 項目数
  converged = 0
  old_eta = eta(old_eps)
  #print("1")
  t = 0
  sig_x = np.ones(N)
  mu_x = np.zeros(N)
  sig_b = np.ones(J)
  mu_a = np.zeros(J)
  mu_b = old_B #np.zeros(J)
  # 元の正規分布のパラメタ？
  sig_x_pre = np.ones(N)
  mu_x_pre = np.zeros(N)
  sig_b_pre = np.ones(J)
  mu_b_pre = np.zeros(J)

  while(converged==0):

    #old_A = mu_a
    old_B = mu_b
    sig_x = np.ones(N)
    mu_x = np.zeros(N)
    new_eps = np.zeros([N, J])
    #old_x = mu_x
    # 変分分布q_xの更新(平均と分散の更新)
    for i in range(N):
      sigma_sum = 0
      mu_sum = 0
      sigma_sum = np.sum(old_eta[i] * old_A**2)
      mu_sum = np.sum((Y[i] - 0.5) * old_A + 2 * old_eta[i] * mu_b * old_A**2)
      
      # θの事前分布をN(0, 1)としている
      sigma_i_hat = 1 / (1/sig_x_pre[i] + 2 * sigma_sum) 
      mu_i_hat = sigma_i_hat * (mu_x_pre[i]/sig_x_pre[i] + mu_sum)
      sig_x[i] = sigma_i_hat
      mu_x[i] = mu_i_hat

      # M-step
      # εijの更新
    for i in range(N):
      eps_tmp = old_A**2 * (sig_x[i]+mu_x[i]**2-2*mu_x[i]*mu_b+mu_b**2+sig_b)
      new_eps[i] = np.sqrt(eps_tmp)

    new_eta = eta(new_eps)

    # qbの更新(こっちさき？)
    new_sig_b = np.ones(J)
    new_mu_b = np.zeros(J)
    for j in range(J): 
        sum_b1 = 0
        sum_b2 = 0
        for i in range(N):
            # Bの更新
            sum_b1 += new_eta[i][j] * old_A[j]**2
            sum_b2 += (-Y[i][j]+0.5)*old_A[j] + 2*mu_x[i]*new_eta[i][j]*old_A[j]**2

        sig_b_hat = 1/(1/sig_b_pre[j] + 2*sum_b1)
        mu_b_hat = sig_b_hat*(mu_b_pre[j]/sig_b_pre[j] + sum_b2)

        new_mu_b[j] = mu_b_hat
        new_sig_b[j] = sig_b_hat

    mu_b = new_mu_b
    sig_b = new_sig_b

    # aの更新
    new_A = np.zeros(J)
    for j in range(J): 
        sum_a1 = 0
        sum_a2 = 0
        for i in range(N):        
            # Aの更新
            sum_a1 += 2*new_eta[i][j]*(mu_x[i]**2 + sig_x[i] - 2*mu_x[i]*mu_b[j] + mu_b[j]**2 + sig_b[j])
            sum_a2 += (Y[i][j]-0.5)*(mu_x[i] - mu_b[j])

        new_A[j] = sum_a2 / sum_a1
  
    # 収束判定
    M = np.sqrt(np.sum((new_A-old_A)**2)+np.sqrt(np.sum((mu_b-old_B)**2)))
    MA = np.sqrt(np.sum((new_A-old_A)**2))
    MB = np.sqrt(np.sum((mu_b-old_B)**2))
    Me = np.sqrt(np.sum((new_eps-old_eps)**2))
    #print("M\n",M)
    #print("MA\n",MA)
    #print("MB\n",MB)
    if M < 0.0001:
      converged = 1
    # 値の更新
    old_A = new_A #np.array(new_A).T
    old_eps = new_eps 
    old_eta = new_eta
    # 経験ベイズ推定により更新
    #print(sig_x-sig_x_pre)
    sig_x_pre = sig_x
    mu_x_pre = mu_x
    sig_b_pre = sig_b
    mu_b_pre = mu_b
    # 最大反復回数まで収束しなかったら打ち切り
    t += 1
    if t > 10000:
      converged = 1
      print("10000回打ち切り")
  print(t)
  return old_A, mu_b, sig_b, mu_x, sig_x 
