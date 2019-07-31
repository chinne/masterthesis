import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd


class CreditcardDataset(Dataset):

    def __init__(self, df, device, label):
        xy = df.values
        self.len = xy.shape[0]
        self.features = torch.tensor(xy, dtype=torch.float32)
        if label is 'yes':
            self.features = torch.tensor(xy[:, 0:-1], dtype= torch.float32)
            self.label = torch.tensor(xy[:, [-1]], dtype= torch.float32) 
            self.label.to(device)
        self.features.to(device)
        

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return self.len



def prepareDataset(df, batch_size:int, device='cpu', label='None'):
    dataset = CreditcardDataset(df, device, label)
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
