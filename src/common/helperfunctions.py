import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd


class CreditcardDataset(Dataset):

    def __init__(self, df=None, device=None, withLabel=None):
        self.df = df.values
        self.device = device
        self.withLabel = withLabel
        #xy = df.values
        self.len = self.df.shape[0]
        
        if self.withLabel is True:
            self.features = torch.tensor(self.df[:, 0:-1], dtype= torch.float32).to(self.device)
            self.label = torch.tensor(self.df[:, [-1]], dtype= torch.float32).to(self.device)
            print('sdf')
        self.features = torch.tensor(self.df, dtype=torch.float32).to(self.device)

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return self.len



def prepareDataset(df, batch_size:int, device=None, withLabel=None):
    dataset = CreditcardDataset(df, device, withLabel)
    dataloader = DataLoader(dataset=dataset,
                            batch_size = batch_size)
    return dataloader

def standardScaler_df(df):
    df['Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))
    df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    return df

def robustScaler_df(df):
    df['Time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1, 1))
    df['Amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    return df

def dropTime(df):
    df.drop('Time', axis=1, inplace=True)
                    
def getFeatureList(df):
    label_col = []
    feature_cols = []
    label_col = [i for i in df.columns if 'Class' in i]
    feature_cols = [i for i in df.columns if i not in label_col]
    return label_col, feature_cols
    
    
    
# def prepare_df(df, drop_time='yes', scaling='standard'):
#     if (scaling == 'standard'):
#         df['Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))
#         df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
#     else:
#         df['Time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1, 1))
#         df['Amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
#     if (drop_time == 'yes'):
#         df.drop('Time', axis=1, inplace=True)
#     label_col = [i for i in df.columns if 'Class' in i]
#     feature_cols = [i for i in df.columns if i not in label_col]
#     return df, label_col, feature_cols
