import seaborn as sns
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_covariance_mat(X, coef_name_list):
    X = np.array(X)
    df = pd.DataFrame(X, columns=coef_name_list)
    covMatrix = df.corr()
    print(covMatrix)
    fig, ax = plt.subplots(1)
    ax = sns.heatmap(covMatrix, annot=True)
    #
