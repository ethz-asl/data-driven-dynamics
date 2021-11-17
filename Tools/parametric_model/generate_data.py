from generate_parametric_model import start_model_estimation
import pandas as pd
import numpy as np
import time


config = "configs/quadrotor_model.yaml"
log = "../../model_results/bad_min_eig_avg_sorted.csv"
tmp_log = "model_results/tmp.csv"
window_size = 100
step_size = 100
 
df = pd.read_csv(log)
#df = df.sample(frac=1)

N = df.shape[0]

for i in range(0,N-window_size,step_size):
    df_tmp = df.loc[range(0,i+window_size)]
    df_tmp.reset_index(drop=True)
    df_tmp.to_csv(tmp_log,index=True)
    start_model_estimation(config,tmp_log,False,False)
    time.sleep(1)
