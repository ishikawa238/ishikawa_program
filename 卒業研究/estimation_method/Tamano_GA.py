# %%
import numpy as np
import math
import arviz as az
import pandas as pd
from scipy import stats
import random
np.random.seed(314)

#%% # function eta
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
#%% armijo条件による勾配降下法
class GradientDescent:
    def __init__(self, fun, der, xi=0.3, tau=0.9, tol=1e-2, ite_max=100):
        self.fun = fun         # 目的関数
        self.der = der         # 関数の勾配
        self.xi  = xi          # Armijo条件の定数
        self.tau = tau         # 方向微係数の学習率
        self.tol = tol         # 勾配ベクトルのL2ノルムがこの値より小さくなると計算を停止
        self.path = None       # 解の点列
        self.ite_max = ite_max # 最大反復回数

    def minimize(self, x, pre_p_params, item_params, Yi, S):
        path = [x]
        
        for i in range(self.ite_max):
            grad = self.der(x, pre_p_params, item_params, Yi, S)
            #print("grad\n", x, grad)
            if np.linalg.norm(grad, ord=2)<self.tol:
                break
            else:
                """
                beta = 1
                count_ar = 0
                while self.fun(x - beta*grad, pre_p_params, item_params, Yi, S) > (self.fun(x, pre_p_params, item_params, Yi, S) - self.xi*beta*np.dot(grad, grad)):
                #while self.fun(x - beta*grad) > (self.fun(x) - self.xi*beta*np.dot(grad, grad)):
                    # Armijo条件を満たすまでループする
                    beta = self.tau*beta
                    count_ar += 1
                
                #print("count_ar\n", count_ar)
                x = x - beta * grad
                """
                x = x - 0.001 * grad
                path.append(x)
        
        self.opt_x = x                # 最適解
        self.opt_result = self.fun(x, pre_p_params, item_params, Yi, S) #, pre_p_params, item_params, Yi, S) # 関数の最小値
        self.path = np.array(path)    # 探索解の推移

    def minimize_mean(self, x, sd_q, pre_p_params, item_params, Yi, S):
        path = [x]
        
        for i in range(self.ite_max):
            grad = self.der(x, sd_q, pre_p_params, item_params, Yi, S)
            print("grad\n", x, grad)
            #if np.linalg.norm(grad, ord=2)<self.tol: # 行列ver
            if grad < self.tol:
                break
            else:
                """
                beta = 1
                while self.fun(x - beta*grad, sd_q, pre_p_params, item_params, Yi, S) > (self.fun(x, sd_q, pre_p_params, item_params, Yi, S) - self.xi*beta*grad**2):
                    # Armijo条件を満たすまでループする
                    beta = self.tau*beta
                    
                x = x - beta * grad
                """
                x = x - 0.001 * grad
                path.append(x)
        
        self.opt_x = x                # 最適解
        self.opt_result = self.fun(x, sd_q, pre_p_params, item_params, Yi, S) # 関数の最小値
        self.path = np.array(path)    # 探索解の推移

    def minimize_sd(self, x, mean_q, pre_p_params, item_params, Yi, S):
        path = [x]
        
        for i in range(self.ite_max):
            grad = self.der(x, mean_q, pre_p_params, item_params, Yi, S)
            
            #if np.linalg.norm(grad, ord=2)<self.tol: # 行列ver
            if grad < self.tol:
                break
            else:
                beta = 1
                
                while self.fun(x - beta*grad, mean_q, pre_p_params, item_params, Yi, S) > (self.fun(x, mean_q, pre_p_params, item_params, Yi, S) - self.xi*beta*grad**2):
                    # Armijo条件を満たすまでループする
                    beta = self.tau*beta
                    
                x = x - beta * grad
                path.append(x)
        
        self.opt_x = x                # 最適解
        self.opt_result = self.fun(x, mean_q, pre_p_params, item_params, Yi, S) # 関数の最小値
        self.path = np.array(path)    # 探索解の推移

#%% 今回用の勾配降下法
def Gradient_armijo(f_mean, f_mean_der, f_sd, f_sd_der, init_params, pre_p_params, item_params, Yi, S):
   mean_q = init_params[0]
   sd_q = init_params[1]

   path_mean = [mean_q]
   path_sd = [sd_q]
   
   ite_max = 2000   # 最大反復回数
   tol = 1e-6       # 収束条件
   tau = 0.9        # 方向微係数の学習率
   xi = 0.3         # Armijo条件の定数

   for i in range(ite_max):
      grad_mean = f_mean_der(mean_q, sd_q, pre_p_params, item_params, Yi, S)
      grad_sd = f_sd_der(mean_q, sd_q, pre_p_params, item_params, Yi, S)
      print("grad\n", grad_mean, grad_sd)
      if grad_mean < tol and grad_sd < tol:
         break
      else:
         # Armijo条件を満たすalfaを導出
         # 平均パラメタの更新
         alfa_mean = 1
         while f_mean(mean_q-alfa_mean*grad_mean, sd_q, pre_p_params, item_params, Yi, S) > (f_mean(mean_q, sd_q, pre_p_params, item_params, Yi, S) - xi*alfa_mean*grad_mean**2):
            print(f_mean(mean_q-alfa_mean*grad_mean, sd_q, pre_p_params, item_params, Yi, S) - f_mean(mean_q, sd_q, pre_p_params, item_params, Yi, S) - xi*alfa_mean*grad_mean**2)
            alfa_mean = tau*alfa_mean

         mean_q = mean_q - alfa_mean*grad_mean
         print("mean_q\n", mean_q)
         path_mean.append(mean_q)

         # 標準偏差パラメタを更新
         alfa_sd = 1
         while f_sd(mean_q, sd_q-alfa_sd*grad_sd, pre_p_params, item_params, Yi, S) > (f_mean(mean_q, sd_q-alfa_sd*grad_sd, pre_p_params, item_params, Yi, S) - xi*alfa_sd*grad_sd**2):
            print(f_sd(mean_q, sd_q-alfa_sd*grad_sd, pre_p_params, item_params, Yi, S) - f_mean(mean_q, sd_q-alfa_sd*grad_sd, pre_p_params, item_params, Yi, S) - xi*alfa_sd*grad_sd**2)
            alfa_sd = tau*alfa_sd

         sd_q = sd_q - alfa_sd*grad_sd
         path_sd.append(sd_q)

         print("sd_q\n", sd_q)

   return mean_q, sd_q


#%% 関数の定義
"""
def f_mean(x, sd_q, pre_p_params, item_params, Yi, S):
    mean_p = pre_p_params[0]
    sd_p = pre_p_params[1]
    para_a = item_params.T[0]
    para_b = item_params.T[1]
    sum = 0
    for s in range(S):
       eps = np.random.normal(0, 1)
       theta_s = x + eps*sd_q
       for j in range(len(para_a)):
          exp = np.exp(-para_a[j]*(theta_s-para_b[j]))
          sum += (Yi[j] + exp/(1+exp))*para_a[j]
    
    fm = (x-mean_p)/sd_p**2 - sum/S
    return fm

def f_mean_der(x, sd_q, pre_p_params, item_params, Yi, S):
    mean_p = pre_p_params[0]
    sd_p = pre_p_params[1]
    para_a = item_params.T[0]
    para_b = item_params.T[1]
    sum = 0
    for s in range(S):
       eps = np.random.normal(0, 1)
       theta_s = x + eps*sd_q
       for j in range(len(para_a)):
          exp = np.exp(-para_a[j]*(theta_s-para_b[j]))
          sum += (exp/(1+exp)**2)*((-1)*para_a[j]**2)
    
    fm_der = 1/sd_p**2 - sum/S
    return fm_der

def f_sd(mean_q, x, pre_p_params, item_params, Yi, S):
    mean_p = pre_p_params[0]
    sd_p = pre_p_params[1]
    para_a = item_params.T[0]
    para_b = item_params.T[1]
    sum = 0
    for s in range(S):
       eps = np.random.normal(0, 1)
       theta_s = mean_q + eps*x
       for j in range(len(para_a)):
          exp = np.exp(-para_a[j]*(theta_s-para_b[j]))
          sum += (Yi[j] + exp/(1+exp))*para_a[j]*eps
    
    fs = -1/x + x/sd_p**2 - sum/S
    return fs

def f_sd_der(mean_q, x, pre_p_params, item_params, Yi, S):
    mean_p = pre_p_params[0]
    sd_p = pre_p_params[1]
    para_a = item_params.T[0]
    para_b = item_params.T[1]
    sum = 0
    for s in range(S):
       eps = np.random.normal(0, 1)
       theta_s = mean_q + eps*x
       for j in range(len(para_a)):
          exp = np.exp(-para_a[j]*(theta_s-para_b[j]))
          sum += (Yi[j] + exp/(1+exp)**2)* para_a[j]**2 * eps**2 * (-1)
    
    fs_der = 1/x**2 + 1/sd_p**2 - sum/S
    return fs_der
"""
#%% 関数改良版
def f(x, pre_p_params, item_params, Yi, S):
    mean_p = pre_p_params[0]
    sd_p = pre_p_params[1]
    para_a = item_params.T[0]
    para_b = item_params.T[1]
    #x = [x1, x2]
    if x[1] < 0:
       x[1] = 0.01

    kl = np.log(sd_p/x[1]) + (x[1]**2 + (x[0]-mean_p)**2)/(2*sd_p**2) - 1/2
    
    sum = 0
    for s in range(S):
       eps = np.random.normal(0, 1)
       theta_s = x[0] + eps*x[1]
       for j in range(len(para_a)):
          exp = np.exp(para_a[j]*(theta_s-para_b[j]))
          #sum += (Yi[j] * np.log(1/(1+exp)) + (1-Yi[j])*np.log(exp/(1+exp)))
          sum += Yi[j]*para_a[j]*(theta_s-para_b[j]) - np.log(1+exp)
    
    f_KL = kl - sum/S
    return f_KL

def f_der(x, pre_p_params, item_params, Yi, S):
    mean_p = pre_p_params[0]
    sd_p = pre_p_params[1]
    para_a = item_params.T[0]
    para_b = item_params.T[1]
    sum_mean = 0
    sum_sd = 0
    if x[1] < 0:
       x[1] = 0.01
    
    for s in range(S):
       eps = np.random.normal(0, 1)
       theta_s = x[0] + eps*x[1]
       sum1 = 1
       sum2 = 1
       for j in range(len(para_a)):
          exp = np.exp(para_a[j]*(theta_s-para_b[j]))
          # 尤度変形してから微分
          sum_mean += (Yi[j] - exp/(1+exp))*para_a[j]
          sum_sd += (Yi[j] - exp/(1+exp))*para_a[j]*eps


    fm = (x[0]-mean_p)/sd_p**2 - sum_mean/S
    fs = -1/x[1] + x[1]/sd_p**2 - sum_sd/S

    return np.array([fm, fs])

#%% 分散を既知とした場合
class GradientDescent2:
    def __init__(self, fun, der, xi=0.001, tau=0.7, tol=1e-3, ite_max=2000):
        self.fun = fun         # 目的関数
        self.der = der         # 関数の勾配
        self.xi  = xi          # Armijo条件の定数
        self.tau = tau         # 方向微係数の学習率
        self.tol = tol         # 勾配ベクトルのL2ノルムがこの値より小さくなると計算を停止
        self.path = None       # 解の点列
        self.ite_max = ite_max # 最大反復回数

    def minimize(self, x, pre_p_params, item_params, Yi, S):
        path = [x]
        count = 0
        for i in range(self.ite_max):
            #print(item_params)
            grad = self.der(x, pre_p_params, item_params, Yi, S)
            print("grad\n", x, grad)
            #if np.linalg.norm(grad, ord=2)<self.tol:
            if count == 100:
                break
            #count += 1
            #print(np.linalg.norm(grad, ord=2))
            #if np.linalg.norm(grad, ord=2)<self.tol:
            if abs(grad) < self.tol:
                print(grad, self.tol)
                break
            else:
                #beta = 1
                count_ar = 0
                # エラーでないように最初にある程度小さくする？
                #while x[1] - beta*grad[1] < 0:
                    #beta = self.tau*beta

                beta = self.xi
                """
                #while self.fun(x - beta*grad, pre_p_params, item_params, Yi, S) > (self.fun(x, pre_p_params, item_params, Yi, S) - self.xi*beta*grad[0]**2):
                while self.fun(x - beta*grad, pre_p_params, item_params, Yi, S) > (self.fun(x, pre_p_params, item_params, Yi, S) - self.xi*beta*grad**2):
                #while self.fun(x - beta*grad) > (self.fun(x) - self.xi*beta*np.dot(grad, grad)):
                    # Armijo条件を満たすまでループする
                    beta = self.tau*beta
                    count_ar += 1
                print("count_ar\n", count_ar)
                """
                x = x - beta * grad

                path.append(x)
        
        self.opt_x = x                # 最適解
        self.opt_result = self.fun(x, pre_p_params, item_params, Yi, S) # 関数の最小値
        self.path = np.array(path)    # 探索解の推移

def f_one(x, pre_p_params, item_params, Yi, S):
    mean_p = pre_p_params[0]
    sd_p = pre_p_params[1]
    para_a = item_params.T[0]
    para_b = item_params.T[1]
    sd_q = 0.3
    #x = [x1, x2]
    #x[1] = 1
    #if x[1] < 0:
    #    x[1] = 0.01

    kl = np.log(sd_p/sd_q) + (sd_q**2 + (x-mean_p)**2)/(2*sd_p**2) - 1/2
    
    sum = 0
    for s in range(S):
       eps = np.random.normal(0, 1)
       theta_s = x + eps*sd_q
       for j in range(len(para_a)):
          exp = np.exp(para_a[j]*(theta_s-para_b[j]))
          #sum += (Yi[j] * np.log(1/(1+exp)) + (1-Yi[j])*np.log(exp/(1+exp)))
          sum += Yi[j]*para_a[j]*(theta_s-para_b[j]) - np.log(1+exp)
    
    f_KL = kl - sum/S
    return f_KL

def f_der_one(x, pre_p_params, item_params, Yi, S):
    mean_p = pre_p_params[0]
    sd_p = pre_p_params[1]
    para_a = item_params.T[0]
    para_b = item_params.T[1]
    sum_mean = 0
    sum_sd = 0
    #if x[1] <0:
    #    x[1] = 0.01

    #x[1] = 1
    sd_q = 0.3
    for s in range(S):
       eps = np.random.normal(0, 1)
       theta_s = x + eps*sd_q
       sum1 = 1
       sum2 = 1
       for j in range(len(para_a)):
          exp = np.exp(para_a[j]*(theta_s-para_b[j]))
          # 尤度変形してから微分
          sum_mean += (Yi[j] - exp/(1+exp))*para_a[j]
          sum_sd += (Yi[j] - exp/(1+exp))*para_a[j]*eps


    fm = (x-mean_p)/sd_p**2 - sum_mean/S
    fs = -1/sd_q + sd_q/sd_p**2 - sum_sd/S

    return fm
    #return np.array([fm, fs])

#%% 推定アルゴリズム(GVEMの能力値推定を変更ver)
def Tamano_GA_EM(old_A, old_B, old_eps, Y):
  S = 100
  N = Y.shape[0] # 受検者の人数
  J = Y.shape[1] # 項目数
  converged = 0
  old_eta = eta(old_eps)
  #print("1")
  t = 0
  while(converged==0):

    sig_i = np.zeros(N)
    mu_i = np.zeros(N)
    new_eps = np.zeros([N, J])
    # 変分分布qの更新(平均と分散の更新)
    for i in range(N):
      print("i\n", i)

      # E-step 

      # ↓↓変えたとこ
      # 最適化で更新値を導出
      #gd_mean = GradientDescent(f_mean, f_mean_der)
      #gd_sd = GradientDescent(f_sd, f_sd_der)
      gd = GradientDescent(f, f_der)
      gd2 = GradientDescent2(f_one, f_der_one)
      init_params = [0, 1] # 求める平均、標準偏差の初期値
      pre_p_params = [0, 1] # 事前分布の平均、標準偏差 ←前回の更新値を入れる?
      item_params = np.array(
        [old_A,
        old_B
        ]
      ).T
      
      """
      import matplotlib.pyplot as plt
      x1 = np.linspace(-10, 10, 51)
      x2 = np.linspace(0, 50, 51)
      x1_mesh, x2_mesh = np.meshgrid(x1, x2)
      z = f(np.array((x1_mesh, x2_mesh)), pre_p_params, item_params, Y[i], S)

      fig, ax = plt.subplots(figsize=(6, 6))
      ax.contour(x1, x2, z, levels=np.logspace(-0.3, 1.2, 10))
      ax.set_xlabel("x1")
      ax.set_ylabel("x2")
      ax.set_aspect('equal')
      plt.show()
      """


      #gd_mean.minimize_mean(init_params[0], init_params[1], pre_p_params, item_params, Y[i], S)
      #print(gd_mean.opt_x, gd_mean.opt_result)
      #gd_sd.minimize_sd(init_params[1], init_params[0], pre_p_params, item_params, Y[i], S)
      #gd.minimize(init_params, pre_p_params, item_params, Y[i], S)
      gd2.minimize(0, pre_p_params, item_params, Y[i], S)
      #gd_mean, gd_sd = Gradient_armijo(f_mean, f_mean_der, f_sd, f_sd_der, init_params, pre_p_params, item_params, Y[i], S)
      sig_i[i] = 0.3**2 #gd.opt_x[1]**2
      mu_i[i] = gd2.opt_x
      # ↑↑変えたとこ

      # M-step
      # εijの更新
      eps_tmp = old_B ** 2 - 2 * old_B * mu_i[i] + sig_i[i] + mu_i[i] ** 2
      new_eps[i] = np.sqrt(eps_tmp) * old_A

      #break

    print("mu\n", mu_i)
    print("sig_i\n", sig_i)

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
    print(M)
    print(MA)
    print(MB)
    print("")
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

  return new_A, new_B, mu_i, sig_i, new_eps

