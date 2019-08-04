import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

def BaseMetrics(y_pred,y_true):
    TP = np.sum( (y_pred == 1) & (y_true == 1) )
    TN = np.sum( (y_pred == 0) & (y_true == 0) )
    FP = np.sum( (y_pred == 1) & (y_true == 0) )
    FN = np.sum( (y_pred == 0) & (y_true == 1) )
    return TP, TN, FP, FN

def SimpleAccuracy(y_pred,y_true):
    TP, TN, FP, FN = BaseMetrics(y_pred,y_true)
    ACC = ( TP + TN ) / ( TP + TN + FP + FN )
    return ACC

def CheckAccuracy(real_data, generated_data, feature_cols, label_col=[], seed=42, with_class=False, data_dim=2):
    '''
    This function calulates via the XGBoost classifier the accuracy of the real and generated data.
    The best case would be, that the function returns 0.5 as accuracy.

    label: 0 = real data / 1 = generated data
    '''
    dtrain = np.vstack([real_data[:int(len(real_data)/2)], generated_data[:int(len(generated_data)/2)]]) # Use half of each real and generated set for training
    dlabels = np.hstack([np.ones(int(len(real_data)/2)), np.zeros(int(len(generated_data)/2))]) # synthetic labels
    dtest = np.vstack([real_data[int(len(real_data)/2):], generated_data[int(len(generated_data)/2):]]) # Use the other half of each set for testing
    y_true = dlabels # Labels for test samples will be the same as the labels for training samples, assuming even batch sizes

    print(len(dtrain), len(dtest))
    dtrain = xgb.DMatrix(dtrain, dlabels, feature_names=feature_cols+label_col)
    dtest = xgb.DMatrix(dtest, feature_names=feature_cols+label_col)
   

    params = {
        'max_depth': 4, 
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': seed 
        }

    xgb_model = xgb.train(params, dtrain, num_boost_round=10) # limit to ten rounds for faster evaluation
    y_pred = np.round(xgb_model.predict(dtest))
    
    return accuracy_score(y_true, y_pred)
    #return SimpleAccuracy(y_pred, y_true) # assumes balanced real and generated datasets

def PlotData(real_data_list, generated_data, feature_cols, label_col, seed=42, data_dim=2):
    
    real_samples = pd.DataFrame(real_data_list, columns=feature_cols+label_col)
    gen_samples = pd.DataFrame(generated_data, columns=feature_cols+label_col)
    
    f, axarr = plt.subplots(1, 2, figsize=(6,2) )
    # if with_class:
    #     axarr[0].scatter( real_samples[feature_cols[0]], real_samples[feature_cols[1]], c=real_samples[label_col[0]]/2 , cmap='plasma'  )
    #     axarr[1].scatter( gen_samples[ feature_cols[0]], gen_samples[ feature_cols[1]], c=gen_samples[label_col[0]]/2 , cmap='plasma'  )
        
        # For when there are multiple one-hot encoded label columns
        # for i in range(len(label_cols)):
            # temp = real_samples.loc[ real_samples[ label_cols[i] ] == 1 ]
            # axarr[0].scatter( temp[data_cols[0]], temp[data_cols[1]], c='C'+str(i), label=i )
            # temp = gen_samples.loc[ gen_samples[ label_cols[i] ] == 1 ]
            # axarr[1].scatter( temp[data_cols[0]], temp[data_cols[1]], c='C'+str(i), label=i )
        
    #else:
    axarr[0].scatter( real_samples[feature_cols[0]], real_samples[feature_cols[1]]) #, cmap='plasma'  )
    axarr[1].scatter( gen_samples[feature_cols[0]], gen_samples[feature_cols[1]]) #, cmap='plasma'  )
    axarr[0].set_title('real')
    axarr[1].set_title('generated')   
    axarr[0].set_ylabel(feature_cols[1]) # Only add y label to left plot
    for a in axarr: a.set_xlabel(feature_cols[0]) # Add x label to both plots
    axarr[1].set_xlim(axarr[0].get_xlim()), axarr[1].set_ylim(axarr[0].get_ylim()) # Use axes ranges from real data for generated data
    
    # if save:
    #     plt.save( prefix + '.xgb_check.png' )
        
    plt.show()