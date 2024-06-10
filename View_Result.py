#%% 
import numpy as np
import matplotlib . pyplot as plt
from scipy.stats import pearsonr

#%% Biasの計算 MAE
def calc_Bias(real_params, est_params, real_theta, est_theta):
    N = len(est_theta)
    J = real_params.shape[0]
    # Biasの平均を計算
    Bias_a = abs(real_params.T[0] - est_params.T[0]).mean() 
    Bias_b = abs(real_params.T[1] - est_params.T[1]).mean() 
    Bias_theta = abs(real_theta - est_theta).mean() 
    return Bias_a, Bias_b, Bias_theta
#%% RMSEの計算
def calc_RMSE(real_params, real_theta, idata):

    N = len(real_theta)
    J = real_params.shape[0]
    T = idata.a.shape[0]
    # RMSEを計算
    RMSE_params = np.zeros([2, J])
    RMSE_theta = np.zeros(N)

    # RMSEを計算
    for j in range(J):       
        RMSE_params[0][j] = np.power(real_params.T[0][j] - idata.a.T[j],  2).mean()
        RMSE_params[1][j] = np.power(real_params.T[1][j] - idata.b.T[j],  2).mean()
    for i in range(N):
        RMSE_theta[i] = np.power(real_theta[i] - idata.theta.T[i],  2).mean()

    #print(RMSE_params)
    RMSE_a = (np.sqrt(RMSE_params.T[0])).mean()
    RMSE_b = (np.sqrt(RMSE_params.T[1])).mean()
    RMSE_theta = np.sqrt(RMSE_theta).mean()
    RMSE_list = np.array([RMSE_a, RMSE_b, RMSE_theta])

    return RMSE_list
#%% 相関の計算
def calc_coef(real_params, est_params, real_theta, est_theta):
    # 設問の真の難易度
    b_coef, _ = pearsonr(real_params.T[1], est_params.T[1])
    #print('coef for question difficulty:', b_coef)
    a_coef, _ = pearsonr(real_params.T[0], est_params.T[0])
    #print('coef for question discrimination:', a_coef)
    # 回答者の能力パラメータ
    theta_coef, _ = pearsonr(real_theta, est_theta)
    #print('coef for participant capability:', theta_coef)
    return a_coef, b_coef, theta_coef

#%% 結果を図示
def make_Fig(real_params, est_params, real_theta, est_theta):
    import matplotlib . pyplot as plt
    plt.xlabel("predictied value") # 予測値
    plt.ylabel("true value") # 真値
    #plt.grid(True)
    plt.xlim(-0.4, 1.4)
    plt.ylim(-0.4, 1.4)
    plt.plot([-4, 4.0], [-4, 4.0], linewidth=0.5, color=("black"))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(est_params.T[0], real_params.T[0], alpha=0.5)
    plt.show()
    #
    plt.xlabel("predictied value") # 予測値
    plt.ylabel("true value") # 真値
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)
    plt.plot([-4, 4.0], [-4, 4.0], linewidth=0.5, color=("black"))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(est_params.T[0], real_params.T[0], alpha=0.5)
    plt.show()
    # 
    plt.xlabel("predictied value") # 予測値
    plt.ylabel("true value") # 真値
    #plt.xticks(np.arange(-4.0, 3.5, 0.5))
    #plt.yticks(np.arange(-4.0, 3.5, 0.5))
    plt.xlim(-4.0, 4.0)
    plt.ylim(-4.0, 4.0)
    #plt.grid(True)
    plt.plot([-8.0, 8.0], [-8.0, 8.0], linewidth=0.5, color=("black"))
    plt.scatter(est_params.T[1], real_params.T[1], alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    #
    import matplotlib . pyplot as plt
    plt.xlabel("predictied value") # 予測値
    plt.ylabel("true value") # 真値
    plt.xlim(-4.0, 4.0)
    plt.ylim(-4.0, 4.0)
    plt.plot([-4.0, 4.0], [-4.0, 4.0], linewidth=0.5, color=("black"))
    #plt.grid(True)
    plt.scatter(est_theta, real_theta, alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# %%
