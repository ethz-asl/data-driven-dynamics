import pandas as pd
import numpy as np
import sys
import yaml
import matplotlib.pyplot as plt
from pathlib import Path


groundtruth = "../../model_results/good.yaml"
truth = {}
f_params = ["vertical_rot_drag_lin",
            "vertical_rot_thrust_lin",
            "vertical_rot_thrust_quad",
            "c_d_fuselage_x",
            "c_d_fuselage_y",
            "c_d_fuselage_z"]

error_cols = ["error.vertical_rot_drag_lin",
            "error.vertical_rot_thrust_lin",
            "error.vertical_rot_thrust_quad",
            "error.c_d_fuselage_x",
            "error.c_d_fuselage_y",
            "error.c_d_fuselage_z"]

data_dir = Path("model_results")
#data_dir = Path("../../../bad_det_sorted")
#data_dir = Path("../../../bad_sliding")


filenames = list(sorted(data_dir.glob("multirotor_model*")))

with open(groundtruth,"r") as file:
    truth = yaml.safe_load(file)

data = []
for filename in filenames:
    file = filename.open()
    data.append(yaml.safe_load(file))

for i in data:
    i.pop("model")
    #i.pop("numper of samples")


df = pd.json_normalize(data)


for param in f_params:
    df["error."+param] = ((df["coefficients."+param]-truth["coefficients"][param]))**2

df["tot_error"]= np.sqrt(df[error_cols].sum(axis=1))

idx = range(0,df.index[-1])

fig, axs = plt.subplots(5,1)

axs[0].plot(df.index[idx],df["FIM.lin.det"][idx])
axs[0].set_title("Determinant")
axs[1].plot(df.index[idx],df["FIM.lin.inv_cond"][idx])
axs[1].set_title("Inverse cond. number")
axs[2].plot(df.index[idx],df["FIM.lin.min_eig"][idx])
axs[2].set_title("Min. Eigenvalue")
axs[3].plot(df.index[idx],df["FIM.lin.trace"][idx])
axs[3].set_title("Trace")
axs[4].plot(df.index[idx],df["tot_error"][idx])
axs[4].set_title("Error")
plt.tight_layout(h_pad=0.1)
plt.show()
print(min(df["tot_error"]))