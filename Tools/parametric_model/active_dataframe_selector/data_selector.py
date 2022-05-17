class ActiveDataSelector():
    def __init__(self, data_df):
        self.data_df = data_df

    def select_dataframes(self, ratio = 10):
        idx = self.data_df.sort_values(by=["fisher_information_force"]).index[0:self.data_df.shape[0]*10//100]
        idx = idx.append(self.data_df.sort_values(by=["fisher_information_rot"]).index[0:self.data_df.shape[0]*10//100])
        idx = idx.unique()
        idx = idx.sort_values()
        self.data_df = self.data_df.loc[idx]
        self.data_df.reset_index(drop=True)
        return self.data_df
