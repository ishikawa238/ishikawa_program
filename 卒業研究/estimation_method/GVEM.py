# %%
# GVEM
import numpy as np
import math
import arviz as az
import pandas as pd

np.random.seed(314)
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
#%%
def H(x):
  return -x*np.log(x) - (1-x)*np.log(1-x)

def GVEM_1(old_A, old_B, old_eps, Y):
  N = Y.shape[0] # 受検者の人数
  J = Y.shape[1] # 項目数
  converged = 0
  old_eta = eta(old_eps)
  #print("1")
  t = 0
  while(converged==0):

    sig_i = np.zeros([1, N])
    mu_i = np.zeros([1, N])
    new_eps = np.zeros([N, J])
    # 変分分布qの更新(平均と分散の更新)
    for i in range(N):
      sigma_sum = 0
      mu_sum = 0
      sum_tmp = 0
      # E-step 
      for j in range(J):
        # 普通の2plver
        sigma_sum = sigma_sum + old_eta[i][j]*old_A[0][j]*old_A[0][j]
        mu_sum = mu_sum + (2*old_eta[i][j]*old_B[0][j]*old_A[0][j] + Y[i][j] - 0.5)*old_A[0][j]
      
      sigma_i_hat = 1 / (1 + 2*sigma_sum) # diag(k)で分散共分散を作成?
      #sigma_i_hat = 1+2*sigma_sum
      mu_i_hat = sigma_i_hat * mu_sum
      sig_i[0][i] = sigma_i_hat
      mu_i[0][i] = mu_i_hat


      # 本当に正規分布に近似して大丈夫そうか確かめてみる
      #tmp = (mu_sum**2 - 2*sum_tmp*(1 + 2*sigma_sum)) / ((1 + 2*sigma_sum)**2)
      #print(tmp)

      # ここからM-step?
      for j in range(J):
        eps_tmp = (old_B[0][j]**2-2*old_B[0][j]*mu_i_hat + sigma_i_hat + mu_i_hat**2)
        new_eps[i][j] = np.sqrt(eps_tmp)* old_A[0][j]
        #print(new_eps[i][j])
    print("mu_i", mu_i)
    print("sig_i", sig_i)
    print("new_eps", new_eps)
    # Bの更新
    #print("new_etaの計算")
    #print(new_eps)
    new_eta = eta(new_eps)
    #print("2")
    print("new_eta", new_eta)
    new_B = np.zeros([1,J])

    for j in range(J):
      num = 0
      num_eta = 0
      for i in range(N):
        num = num + (1/2 - Y[i][j] + 2*new_eta[i][j]*old_A[0][j]*mu_i[0][i])
        #
        num_eta = num_eta + 2*new_eta[i][j]*old_A[0][j]

      #print(num)
      #print(2*num_eta)
      #print(num / 2*num_eta)
      new_B[0][j] = num / num_eta #(np.sum(2*new_eta, axis=0))
    #print("---")
    print("new_B", new_B)
    # Aの更新
    #print(sig_i)
    new_A = np.zeros([1, J])
    for j in range(J):
      mat = 0
      vec = 0
      for i in range(N):
        mat = mat + new_eta[i][j]*(old_B[0][j]**2-2*old_B[0][j]*mu_i[0][i]+sig_i[0][i]+mu_i[0][i]**2)
        vec = vec + ((Y[i][j] - 0.5)*mu_i[0][i] + (0.5 - Y[i][j])*new_B[0][j])
        #print(vec)
      new_A[0][j] = vec / (2*mat)
      #new_A[j][0] = vec / (2*mat)
    print("new_A", new_A)
    # 収束判定

    M = np.sqrt(np.sum((new_A-old_A)**2)+np.sqrt(np.sum((new_B-old_B)**2)))
    MA = np.sqrt(np.sum((new_A-old_A)**2))
    MB = np.sqrt(np.sum((new_B-old_B)**2))
    if M < 0.0001:
      converged = 1
    print(M)
    print(MA)
    print(MB)
    print("")
    # 値の更新
    old_A = new_A #np.array(new_A).T
    old_B = new_B
    old_eps = new_eps 
    old_eta = new_eta
    #print(old_eta)
    #print(mu_i)
    #print(M)
    break
    t += 1
    if t > 1000:
      converged = 1
      print("1000回打ち切り")

  #print(b_est)

  return new_A, new_B, mu_i, sig_i

# %%
# 計算量削減版
def GVEM_vec_2pl(old_A, old_B, old_eps, Y, real_params, real_theta):
  N = Y.shape[0] # 受検者の人数
  J = Y.shape[1] # 項目数
  converged = 0
  old_eps2 = np.full((N, J), 0.5)
  old_eta = eta(old_eps)
  #print("1")
  t = 0
  S = 100
  Bias = np.zeros(3)
  RMSE = np.zeros(3)
  RMSE_a = []
  RMSE_b = []
  RMSE_theta = []
  Bias_a = []
  Bias_b = []
  Bias_theta = []

  while(converged==0):

    sig_i = np.zeros(N)
    mu_i = np.zeros(N)
    new_eps = np.zeros([N, J])
    new_eps2 = np.zeros([N, J])
    # 変分分布qの更新(平均と分散の更新)
    #print("old_eps2\n",old_eps2)
    for i in range(N):
      sigma_sum = 0
      mu_sum = 0
      sum_tmp = 0
      # E-step 
      sigma_sum = np.sum(old_eta[i] * old_A ** 2)
      mu_sum = np.sum((2 * old_eta[i] * old_B * old_A + Y[i] - 0.5) * old_A)

      # 新しい近似の方
      #mu_sum2 = np.sum((Y[i]-old_eps2[i])*old_A)

      #for s in range(S):
      #  sigma_sum = np.sum(old_eta[i] * old_A ** 2)
      #  mu_sum = np.sum((2 * old_eta[i] * (old_B - eps_s) * old_A + Y[i] - 0.5) * old_A)
      #print("比較\n", mu_sum, mu_sum2)
      sigma_i_hat = 1 / (1 + 2 * sigma_sum)
      mu_i_hat = sigma_i_hat * mu_sum
      sig_i[i] = sigma_i_hat
      mu_i[i] = mu_i_hat
      #print("mu_i\n", mu_sum / (1 + 2 * sigma_sum), mu_i[i])
      #print("")
      # M-step
      # εijの更新
      eps_tmp = old_B ** 2 - 2 * old_B * mu_i_hat + sigma_i_hat + mu_i_hat ** 2
      new_eps[i] = np.sqrt(eps_tmp) * old_A

      # eps2の更新（最小化）
      #new_eps2[i] = np.exp(old_A*(mu_i_hat-old_B)) / (1+np.exp(old_A*(mu_i_hat-old_B)))

    # Bの更新
    new_eta = eta(new_eps)
    num = np.sum((0.5 - Y + 2 * np.multiply((np.multiply(new_eta, old_A)).T, mu_i).T), axis=0)
    num_eta = np.sum(2 * np.multiply(new_eta, old_A), axis=0)
    new_B = num / num_eta

    # Aの更新
    mat = np.sum(new_eta * ( np.array([old_B**2]*N) - np.multiply((np.array([2 * old_B]*N)).T, mu_i).T + np.array([sig_i+mu_i**2]*J).T), axis=0)
    vec = np.sum((np.multiply((Y - 0.5).T, mu_i).T + np.multiply((0.5 - Y), new_B)), axis=0)
    new_A = vec / (2 * mat)
    
    # 収束判定
    M = np.sqrt(np.sum((new_A-old_A)**2)+np.sqrt(np.sum((new_B-old_B)**2)))
    MA = np.sqrt(np.sum((new_A-old_A)**2))
    MB = np.sqrt(np.sum((new_B-old_B)**2))
    if M < 0.0001:
      converged = 1
    #print(M)
    #print(MA)
    #print(MB)
    #print("")
    # 値の更新
    old_A = new_A #np.array(new_A).T
    old_B = new_B
    old_eps = new_eps 
    old_eta = new_eta
    old_eps2 = new_eps2
    # 最大反復回数まで収束しなかったら打ち切り
    t += 1
    if t > 10000:
      converged = 1
      print("10000回打ち切り")
    # 反復ごとのRMSEを出したい
    # とりあえず１回の推定で導出する体で
    RMSE[0] = np.sqrt(sum(np.power(real_params.T[0] - new_A,  2)) / J )
    RMSE[1] = np.sqrt(sum(np.power(real_params.T[1] - new_B,  2)) / J )
    RMSE[2] = np.sqrt(sum(np.power(real_theta - mu_i,  2)) / N )
    Bias[0] = sum(real_params.T[0] - new_A) / J
    Bias[1] = sum(real_params.T[1] - new_B) / J   
    Bias[2] = sum(real_theta - mu_i) / N   

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
  #print(old_eps2)
  return new_A, new_B, mu_i, sig_i, new_eps, prog_accuracy_item, prog_accuracy_user

#%% 最高速ver
def GVEM_vec_2pl_fast(old_A, old_B, old_eps, Y):
  N = Y.shape[0] # 受検者の人数
  J = Y.shape[1] # 項目数
  converged = 0
  old_eta = eta(old_eps)
  #print("1")
  t = 0
  Bias = np.zeros(3)
  RMSE = np.zeros(3)
  RMSE_a = []
  RMSE_b = []
  RMSE_theta = []
  Bias_a = []
  Bias_b = []
  Bias_theta = []

  while(converged==0):

    sig_i = np.zeros(N)
    mu_i = np.zeros(N)
    new_eps = np.zeros([N, J])
    new_eps2 = np.zeros([N, J])
    # 変分分布qの更新(平均と分散の更新)
    #print("old_eps2\n",old_eps2)
    sigma_i_hat = 1 / (1 + 2*np.sum(old_eta*np.array([old_A**2]*N), axis=1) )
    mu_i_hat = sigma_i_hat * (np.sum((2*old_eta*old_A*old_B + Y - 0.5) * old_A, axis=1))
    mu_i = mu_i_hat
    sig_i = sigma_i_hat

    # M-step
    # εijの更新
    eps_tmp = old_A**2 * (np.array([sig_i+mu_i**2]*J).T - 2*np.array([mu_i]*J).T*old_B+np.array([old_B**2]*N))
    new_eps = np.sqrt(eps_tmp)
    # eps2の更新（最小化）
    #new_eps2[i] = np.exp(old_A*(mu_i_hat-old_B)) / (1+np.exp(old_A*(mu_i_hat-old_B)))

    # Bの更新
    new_eta = eta(new_eps)
    num = np.sum((0.5 - Y + 2 * np.multiply((np.multiply(new_eta, old_A)).T, mu_i).T), axis=0)
    num_eta = np.sum(2 * np.multiply(new_eta, old_A), axis=0)
    new_B = num / num_eta

    # Aの更新
    mat = np.sum(new_eta * ( np.array([old_B**2]*N) - np.multiply((np.array([2 * old_B]*N)).T, mu_i).T + np.array([sig_i+mu_i**2]*J).T), axis=0)
    vec = np.sum((np.multiply((Y - 0.5).T, mu_i).T + np.multiply((0.5 - Y), new_B)), axis=0)
    new_A = vec / (2 * mat)
    
    # 収束判定
    M = np.sqrt(np.sum((new_A-old_A)**2)+np.sqrt(np.sum((new_B-old_B)**2)))
    MA = np.sqrt(np.sum((new_A-old_A)**2))
    MB = np.sqrt(np.sum((new_B-old_B)**2))
    if M < 0.0001:
      converged = 1
    #print(M)
    #print(MA)
    #print(MB)
    #print("")
    # 値の更新
    old_A = new_A #np.array(new_A).T
    old_B = new_B
    old_eps = new_eps 
    old_eta = new_eta
    old_eps2 = new_eps2
    # 最大反復回数まで収束しなかったら打ち切り
    t += 1
    if t > 10000:
      converged = 1
      print("10000回打ち切り")
    # 反復ごとのRMSEを出したい
    # とりあえず１回の推定で導出する体で
  #   RMSE[0] = np.sqrt(sum(np.power(real_params.T[0] - new_A,  2)) / J )
  #   RMSE[1] = np.sqrt(sum(np.power(real_params.T[1] - new_B,  2)) / J )
  #   RMSE[2] = np.sqrt(sum(np.power(real_theta - mu_i,  2)) / N )
  #   Bias[0] = sum(real_params.T[0] - new_A) / J
  #   Bias[1] = sum(real_params.T[1] - new_B) / J   
  #   Bias[2] = sum(real_theta - mu_i) / N   

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
  #print(old_eps2)
  return new_A, new_B, mu_i, sig_i, new_eps, t #prog_accuracy_item, prog_accuracy_user, t

#%%
def GVEM_tajigen_1(old_A, old_B, old_eps, Y):
  N = Y.shape[0] # 受検者の人数
  J = Y.shape[1] # 項目数
  converged = 0
  old_eta = eta(old_eps)
  t = 0
  while(converged==0):

    sig_i = np.zeros([1, N])
    mu_i = np.zeros([1, N])
    new_eps = np.zeros([N, J])

    for i in range(N):
      sigma_sum = 0
      mu_sum = 0
      sum_tmp = 0
      # E-step 
      for j in range(J):
        # 多次元の1次元ver
        """
        sigma_sum = sigma_sum + old_eta[i][j]*old_A[0][j]*old_A[0][j]
        mu_sum = mu_sum + (2*old_eta[i][j]*old_B[0][j] + Y[i][j] - 0.5)*old_A[0][j]
        sum_tmp += old_eta[i][j]*old_B[0][j]*old_B[0][j]
        """
        # 普通の2plver
        sigma_sum = sigma_sum + old_eta[i][j]*old_A[0][j]*old_A[0][j]
        mu_sum = mu_sum + (2*old_eta[i][j]*old_B[0][j] + Y[i][j] - 0.5)*old_A[0][j]
      
      sigma_i_hat = 1 / (1 + 2*sigma_sum) # diag(k)で分散共分散を作成?
      #sigma_i_hat = 1+2*sigma_sum
      mu_i_hat = sigma_i_hat * mu_sum
      sig_i[0][i] = sigma_i_hat
      mu_i[0][i] = mu_i_hat


      # 本当に正規分布に近似して大丈夫そうか確かめてみる
      #tmp = (mu_sum**2 - 2*sum_tmp*(1 + 2*sigma_sum)) / ((1 + 2*sigma_sum)**2)
      #print(tmp)

      # ここからM-step?
      for j in range(J):
        eps_tmp = old_B[0][j]**2-2*old_B[0][j]*mu_i_hat + old_A[0][j] *(sigma_i_hat + mu_i_hat * mu_i_hat) * old_A[0][j]
        new_eps[i][j] = np.sqrt(eps_tmp)
        #print(new_eps[i][j])

    # Bの更新
    new_eta = eta(new_eps)
    new_B = np.zeros([1,J])

    for j in range(J):
      num = 0
      num_eta = 0
      for i in range(N):
        num = num + (1/2 - Y[i][j] + 2*new_eta[i][j]*old_A[0][j]*mu_i[0][i])
        #
        num_eta = num_eta + 2*new_eta[i][j]

      #print(num)
      #print(2*num_eta)
      #print(num / 2*num_eta)
      new_B[0][j] = num / num_eta #(np.sum(2*new_eta, axis=0))
    #print("---")
    #print(new_B)
    # Aの更新
    #print(sig_i)
    new_A = np.zeros([1, J])
    for j in range(J):
      mat = 0
      vec = 0
      for i in range(N):
        mat = mat + new_eta[i][j]*sig_i[0][i]+new_eta[i][j]*(mu_i[0][i]**2)
        vec = vec + (Y[i][j] - 1/2 + 2*new_B[0][j]*new_eta[i][j])*mu_i[0][i]
        #print(vec)
      new_A[0][j] = vec / (2*mat)
      #new_A[j][0] = vec / (2*mat)

    # 収束判定

    M = np.sqrt(np.sum((new_A-old_A)**2)+np.sqrt(np.sum((new_B-old_B)**2)))
    MA = np.sqrt(np.sum((new_A-old_A)**2))
    MB = np.sqrt(np.sum((new_B-old_B)**2))
    if M < 0.0001:
      converged = 1
    #print(M)
    #print(MA)
    #print(MB)
    #print("--")
    # 値の更新
    old_A = new_A #np.array(new_A).T
    old_B = new_B
    old_eps = new_eps 
    old_eta = new_eta
    #print(old_eta)
    #print(mu_i)
    #print(M)
    t += 1
    if t > 1000:
      converged = 1
      print("1000回打ち切り")

  #print(b_est)

  return new_A, new_B, mu_i, sig_i

#%% 
# Eステップ変更版、尤度関数を正規分布に近似
def Gauss_like_EM(old_A, old_B, old_eps, Y):
  N = Y.shape[0] # 受検者の人数
  J = Y.shape[1] # 項目数
  converged = 0
  old_eta = eta(old_eps)
  #print("1")
  t = 0
  while(converged==0):

    sig_i = np.zeros(N)
    mu_i = np.zeros(N)
    sig_i2 = np.zeros(N)
    mu_i2 = np.zeros(N)
    new_eps = np.zeros([N, J])
    # 変分分布qの更新(平均と分散の更新)
    for i in range(N):
      sigma_sum = 0
      mu_sum = 0
      sum_tmp = 0
      # E-step 
      sigma_sum = np.sum(old_eta[i] * old_A ** 2)
      mu_sum = np.sum(Y[i]*old_A + 2 * old_eta[i] * old_A**2 * old_B - old_A/2)

      # かくにん用
      sigma_sum2 = np.sum(old_eta[i] * old_A ** 2)
      mu_sum2 = np.sum((2 * old_eta[i] * old_B * old_A + Y[i] - 0.5) * old_A)
      
      sigma_i_hat = 1 / (1 + 2 * sigma_sum)
      mu_i_hat = sigma_i_hat * mu_sum
      sig_i[i] = sigma_i_hat
      mu_i[i] = mu_i_hat

      sig_i2[i] = 1 / (1+2*sigma_sum2)
      mu_i2[i] = mu_sum2*sig_i2[i]

      # M-step
      # εijの更新
      eps_tmp = old_B ** 2 - 2 * old_B * mu_i_hat + sigma_i_hat + mu_i_hat ** 2
      new_eps[i] = np.sqrt(eps_tmp) * old_A
    
    print("mu_i\n", mu_i)
    print("sig_i\n", sig_i)

    print("mu2\n", mu_i2)
    print("sig_i2\n", sig_i2)
    # Bの更新
    new_eta = eta(new_eps)
    num = np.sum((0.5 - Y + 2 * np.multiply((np.multiply(new_eta, old_A)).T, mu_i).T), axis=0)
    num_eta = np.sum(2 * np.multiply(new_eta, old_A), axis=0)
    new_B = num / num_eta

    # Aの更新
    mat = np.sum(new_eta * ( np.array([old_B**2]*N) - np.multiply((np.array([2 * old_B]*N)).T, mu_i).T + np.array([sig_i+mu_i**2]*J).T), axis=0)
    vec = np.sum((np.multiply((Y - 0.5).T, mu_i).T + np.multiply((0.5 - Y), new_B)), axis=0)
    new_A = vec / (2 * mat)
    
    # 収束判定
    M = np.sqrt(np.sum((new_A-old_A)**2)+np.sqrt(np.sum((new_B-old_B)**2)))
    MA = np.sqrt(np.sum((new_A-old_A)**2))
    MB = np.sqrt(np.sum((new_B-old_B)**2))
    if M < 0.0001:
      converged = 1
    #print(M)
    #print(MA)
    #print(MB)
    #print("")
    # 値の更新
    old_A = new_A #np.array(new_A).T
    old_B = new_B
    old_eps = new_eps 
    old_eta = new_eta

    # 最大反復回数まで収束しなかったら打ち切り
    t += 1
    if t > 1000:
      converged = 1
      print("1000回打ち切り")

  return new_A, new_B, mu_i, sig_i