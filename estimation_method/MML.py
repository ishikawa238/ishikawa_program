#%% 
from pyirt import irt
#%%
# MML(周辺最尤推定法)をpyirtを用いて実装 ワンちゃん最尤推定??
def mmle(data):
    # (1)Run by default
    item_param, user_param = irt(data)

    # (2)Supply bounds
    #item_param, user_param = irt(data, theta_bnds = [-4,4], alpha_bnds=[0.1,3], beta_bnds = [-3,3])

    # (3)Supply guess parameter
    #guessParamDict = {1:{'c':0.0}, 2:{'c':0.25}}
    #item_param, user_param = irt(data, in_guess_param = guessParamDict)
    return item_param, user_param