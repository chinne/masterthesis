import numpy as np
import xgboost as xgb

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

def CheckAccuracy(x, g_z, seed=42):#, data_cols, label_cols=[], seed=42, with_class=False, data_dim=2 ):

    dtrain = np.vstack( [ x[:int(len(x)/2)], g_z[:int(len(g_z)/2)] ] ) # Use half of each real and generated set for training
    dlabels = np.hstack( [ np.zeros(int(len(x)/2)), np.ones(int(len(g_z)/2)) ] ) # synthetic labels
    dtest = np.vstack( [ x[int(len(x)/2):], g_z[int(len(g_z)/2):] ] ) # Use the other half of each set for testing
    y_true = dlabels # Labels for test samples will be the same as the labels for training samples, assuming even batch sizes

    dtrain = xgb.DMatrix(dtrain, dlabels)#, feature_names=data_cols+label_cols)
    dtest = xgb.DMatrix(dtest)#, feature_names=data_cols+label_cols)
   

    xgb_params = {

        # 'tree_method': 'hist', # for faster evaluation
        'max_depth': 4, # for faster evaluation
        'objective': 'binary:logistic',
        'random_state': 42,
        'eval_metric': 'auc', # allows for balanced or unbalanced classes 

        }

    xgb_test = xgb.train(xgb_params, dtrain, num_boost_round=10) # limit to ten rounds for faster evaluation
    y_pred = np.round(xgb_test.predict(dtest))

    # return '{:.2f}'.format(SimpleAccuracy(y_pred, y_true)) # assumes balanced real and generated datasets
    return SimpleAccuracy(y_pred, y_true) # assumes balanced real and generated datasets

# def PlotData(x, g_z, seed=42,data_cols, label_cols=[],  with_class=False, data_dim=2, save=False, prefix='' ):
    
#     real_samples = pd.DataFrame(x, columns=data_cols+label_cols)
#     gen_samples = pd.DataFrame(g_z, columns=data_cols+label_cols)
    
#     f, axarr = plt.subplots(1, 2, figsize=(6,2) )
#     if with_class:
#         axarr[0].scatter( real_samples[data_cols[0]], real_samples[data_cols[1]], c=real_samples[label_cols[0]]/2 ) #, cmap='plasma'  )
#         axarr[1].scatter( gen_samples[ data_cols[0]], gen_samples[ data_cols[1]], c=gen_samples[label_cols[0]]/2 ) #, cmap='plasma'  )
        
#         # For when there are multiple one-hot encoded label columns
#         # for i in range(len(label_cols)):
#             # temp = real_samples.loc[ real_samples[ label_cols[i] ] == 1 ]
#             # axarr[0].scatter( temp[data_cols[0]], temp[data_cols[1]], c='C'+str(i), label=i )
#             # temp = gen_samples.loc[ gen_samples[ label_cols[i] ] == 1 ]
#             # axarr[1].scatter( temp[data_cols[0]], temp[data_cols[1]], c='C'+str(i), label=i )
        
#     else:
#         axarr[0].scatter( real_samples[data_cols[0]], real_samples[data_cols[1]]) #, cmap='plasma'  )
#         axarr[1].scatter( gen_samples[data_cols[0]], gen_samples[data_cols[1]]) #, cmap='plasma'  )
#     axarr[0].set_title('real')
#     axarr[1].set_title('generated')   
#     axarr[0].set_ylabel(data_cols[1]) # Only add y label to left plot
#     for a in axarr: a.set_xlabel(data_cols[0]) # Add x label to both plots
#     axarr[1].set_xlim(axarr[0].get_xlim()), axarr[1].set_ylim(axarr[0].get_ylim()) # Use axes ranges from real data for generated data
    
#     if save:
#         plt.save( prefix + '.xgb_check.png' )
        
#     plt.show()