import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve,roc_curve,f1_score,confusion_matrix
from sklearn.metrics import precision_score, recall_score , roc_auc_score
from sklearn.preprocessing import StandardScaler , Binarizer

def get_eval_by_threshold(y_test,pred_proba_c1,thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print('임계값 :',custom_threshold)
        get_clf_eval(y_test,custom_predict,pred_proba_c1)

    def precision_recall_curve_plot(y_test, pred_proba_c1):
    precisions,recalls,thresholds = precision_recall_curve(y_test,pred_proba_c1)
    
    plt.figure(figsize=(8,6))
    thresholds_boundary = thresholds.shape[0]
    plt.plot(thresholds,precisions[0:thresholds_boundary], linestyle='--',label = 'precision')
    plt.plot(thresholds,recalls[0:thresholds_boundary],label='recall')
    
    start ,end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    
    plt.xlabel('Threshold value');plt.ylabel('Precision and Recall value')
    plt.legend();plt.grid()
    plt.show()