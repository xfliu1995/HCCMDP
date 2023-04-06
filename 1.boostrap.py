#####qPCR#####
import pandas as pd
import numpy as np
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,auc,precision_recall_curve,average_precision_score,accuracy_score
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit,RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut,StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model  import LogisticRegression
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import roc_curve, auc,confusion_matrix
from sklearn.metrics import brier_score_loss
import argparse, sys, os, errno
from scipy.stats import chi2
from pycaleva import CalibrationEvaluator
import random

def plot_roc(prob, label):
    fpr, tpr, _ = roc_curve(label, prob)
    precision, recall, _ = precision_recall_curve(label, prob)  # recall: Identical to sensitivity,TPR; precision: PPV
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    # print (fpr, tpr,ppv)
    return roc_auc, fpr, tpr, precision, recall

def find_metrics_best_for_shuffle(fpr, tpr,cut_spe=0.95):
    '''
    used for shuffle roc plot
    '''
    a = 1 - fpr
    b = tpr
    Sensitivity = b
    Specificity = a

    Sensitivity_ =Sensitivity[Specificity>=cut_spe]
    Specificity_=Specificity[Specificity>=cut_spe]
    Sensitivity_best = np.max(Sensitivity_)
    Specificity_best=np.min(Specificity_)
    if Specificity_best==cut_spe:
        Sensitivity_adjust=Sensitivity_best
        Specificity_adjust = Specificity_best
    else:
        Specificity_adjust=cut_spe
        Sensitivity_new = Sensitivity[Specificity < cut_spe]
        Specificity_new = Specificity[Specificity < cut_spe]
        Sensitivity_2=Sensitivity_new[0]
        Specificity_2 = Specificity_new[0]
        Sensitivity_1=Sensitivity_[-1]
        Specificity_1 = Specificity_[-1]
        Sensitivity_adjust = ((Sensitivity_2-Sensitivity_1)/(Specificity_2-Specificity_1))*cut_spe \
                           + Sensitivity_2-((Sensitivity_2-Sensitivity_1)/(Specificity_2-Specificity_1))*Specificity_2
    return Sensitivity_best,Specificity_best,Sensitivity_adjust,Specificity_adjust,Sensitivity,Specificity

def clf_select(name,seed):
    if name =='DT':
        clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, criterion='gini',random_state=seed)
    elif name =='DT_cv':
        tree_para = {'max_depth': [10,30,50,100]}
        clf = GridSearchCV(DecisionTreeClassifier(random_state=seed), tree_para, cv=5)
    elif name == 'SVM':
        clf = SVC(kernel='linear', probability=True, C=1,random_state=seed)
    elif name == 'SVM_cv':
        tree_para = { 'C':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
        clf = GridSearchCV(SVC(kernel= 'linear',probability=True,random_state=seed), tree_para, cv=5)
    elif name == 'RF':
        clf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=seed)
    elif name == 'RF_cv':
        tree_para = {'n_estimators': [50, 100, 200, 500,1000], 'max_depth': [10,50, 100, 500, 1000]}
        clf = GridSearchCV(RandomForestClassifier(random_state=seed), tree_para, cv=5)
    elif name == 'LR':
        clf = LogisticRegression(penalty='l2',solver='liblinear',C=1,random_state=seed)
    elif name == 'LR_cv':
        tree_para = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
        clf = GridSearchCV(LogisticRegression(penalty='l2',solver='liblinear',random_state=seed), tree_para, cv=5)
    elif name=='XGBoost':
        clf=LGBMClassifier(random_state=seed,num_leaves=10,n_estimators=200)
    elif name=='XGBoost_cv':
        tree_para = {'num_leaves': [5,10],'n_estimators': [50,100,500]}
        clf = GridSearchCV(LGBMClassifier(random_state=seed), tree_para, cv=5)
    elif name=='NB':
        clf=GaussianNB()
    elif name=='NB_cv':
        clf = GaussianNB()
    return clf

def imputation(data_train_,data_test_,gene_list,type='type',low=-15):
    data_train=data_train_.copy()
    data_train=data_train.replace('low',low)
    data_train[gene_list]=data_train[gene_list].astype(float)
    data_train_nan=data_train_.copy()
    data_train_nan=data_train_nan.replace('low',np.nan)
    data_train_nan[gene_list]=data_train_nan[gene_list].astype(float)
    data_test=data_test_.copy()
    data_test=data_test.replace('low',low)
    data_test[gene_list]=data_test[gene_list].astype(float)

    data_mean = data_train_nan.mean(axis=0)
    for i in gene_list:
        data_test[i] =  data_test[i].fillna(data_mean[i])

    if type=='all':
        for i in gene_list:
            data_train[i] =  data_train[i].fillna(data_mean[i])
    else:
        data_HCC_train_nan=data_train_nan.loc[data_train_nan['Group']=='HCC',:]
        data_HCC_train=data_train.loc[data_train['Group']=='HCC',:]
        data_HD_train_nan=data_train_nan.loc[data_train_nan['Group']=='HD',:]
        data_HD_train=data_train.loc[data_train['Group']=='HD',:]
        data_LC_train_nan=data_train_nan.loc[data_train_nan['Group']=='LC',:]
        data_LC_train=data_train.loc[data_train['Group']=='LC',:]
        data_HBV_train_nan=data_train_nan.loc[data_train_nan['Group']=='HBV',:]
        data_HBV_train=data_train.loc[data_train['Group']=='HBV',:]

        data_mean_HCC = data_HCC_train_nan.mean(axis=0)
        for i in gene_list:
            data_HCC_train[i] =  data_HCC_train[i].fillna(data_mean_HCC[i])

        data_mean_HD = data_HD_train_nan.mean(axis=0)
        for i in gene_list:
            data_HD_train[i] =  data_HD_train[i].fillna(data_mean_HD[i])

        data_mean_LC = data_LC_train_nan.mean(axis=0)
        for i in gene_list:
            data_LC_train[i] =  data_LC_train[i].fillna(data_mean_LC[i])

        data_mean_HBV = data_HBV_train_nan.mean(axis=0)
        for i in gene_list:
            data_HBV_train[i] =  data_HBV_train[i].fillna(data_mean_HBV[i])
        data_train=pd.concat([data_HCC_train,data_HD_train,data_HBV_train,data_LC_train])

    return data_train,data_test


def get_result(save_name):
    model_seed=random.randint(0, 2000)
    model_type='RF'
    sample_weight = 'balanced'
    cancer = ['HCC', 'HD', 'HBV', 'LC']
    gene_list=['CYTOR','WDR74','GGA2-S','miR.21.5p','RN7SL1.S.fragment','SNORD89']
    data_all=pd.read_csv('data_discovery.txt',sep='\t')
    data_all['AFP final new']=data_all['AFP final'].map(lambda x:1 if x>=400 else 0 if x<400 else -999 )
    # data_all['AFP final']=data_all['AFP final'].fillna(-999)

    sample_HCC=list(data_all.loc[(data_all['Group'] == 'HCC'), 'ID_new'])
    sample_HCC_0 = list(data_all.loc[data_all.stage.isin(['A4', 'A1', 'A3', '0', 'A', 'A2']) & (data_all['Group'] == 'HCC'), 'ID_new'])
    sample_control_all=list(data_all.loc[(data_all['Group'] != 'HCC'), 'ID_new'])
    sample_control_HD=list(data_all.loc[(data_all['Group'] == 'HD'), 'ID_new'])
    sample_control_HBV=list(data_all.loc[(data_all['Group'] == 'HBV'), 'ID_new'])
    sample_control_LC=list(data_all.loc[(data_all['Group'] == 'LC'), 'ID_new'])

    sample_all=sample_HCC+sample_control_all
    sample_all_0=sample_HCC_0+sample_control_all
    sample_HCC_HD=sample_HCC+sample_control_HD
    sample_HCC_HD_0=sample_HCC_0+sample_control_HD
    sample_HCC_HBV=sample_HCC+sample_control_HBV
    sample_HCC_HBV_0=sample_HCC_0+sample_control_HBV
    sample_HCC_LC=sample_HCC+sample_control_LC
    sample_HCC_LC_0=sample_HCC_0+sample_control_LC
    sample_HCC_HBV_LC=sample_HCC+sample_control_HBV+sample_control_LC
    sample_HCC_HBV_LC_0 = sample_HCC_0 + sample_control_HBV + sample_control_LC
    save_path='/result/boostrap/'

    # 预测
    result = pd.DataFrame(columns=['roc_auc_train'])
    result_0 = pd.DataFrame(columns=['roc_auc_train_0'])

    train_index = resample(list(data_all.index), n_samples=len(data_all), replace=True,random_state=model_seed)
    test_index=list(set(list(data_all.index))-set(train_index))
    data_train = pd.DataFrame(np.array(data_all)[train_index, :])
    data_test = pd.DataFrame(np.array(data_all)[test_index, :])
    data_train.columns=data_all.columns
    data_test.columns = data_all.columns

    data_train_afp = data_train.loc[pd.notnull(data_train['AFP final']), :]
    data_test_afp = data_test.loc[pd.notnull(data_test['AFP final']), :]
    data_train['AFP final'] = data_train['AFP final'].fillna(-999)
    data_test['AFP final'] = data_test['AFP final'].fillna(-999)
    data_train, data_test = imputation(data_train, data_test, gene_list, type='all', low=-15)
    x_train_2 = np.array(data_train[gene_list + ['AFP final']])
    x_test_2 = np.array(data_test[gene_list + ['AFP final']])
    y_train_2 = np.array(data_train['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_2 = np.array(data_test['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    if sample_weight == 'balanced':
        sample_weight_ = compute_sample_weight(class_weight='balanced', y=y_train_2)
    clf_2 = clf_select(model_type, seed=0)
    clf_2.fit(x_train_2, y_train_2, sample_weight_)

    #全部
    pred_proba_train_2 = clf_2.predict_proba(x_train_2)
    fpr_train, tpr_train, thresholds = roc_curve(y_train_2, pred_proba_train_2[:, 1])
    Sensitivity_best_train, Specificity_best_train, Sensitivity_adjust_train, Specificity_adjust_train, Sensitivity_train, Specificity_train=find_metrics_best_for_shuffle(fpr_train, tpr_train,cut_spe=0.95)
    roc_auc_train_2 = auc(fpr_train, tpr_train)
    bs_train_2=brier_score_loss(y_train_2, pred_proba_train_2[:, 1])
    ce = CalibrationEvaluator(y_train_2, pred_proba_train_2[:, 1], outsample=True, n_groups=2)
    hl_train_2 =ce.hosmerlemeshow().pvalue
    pred_proba_test_2 = clf_2.predict_proba(x_test_2)
    fpr_test, tpr_test, thresholds = roc_curve(y_test_2, pred_proba_test_2[:, 1])
    Sensitivity_best_test, Specificity_best_test, Sensitivity_adjust_test, Specificity_adjust_test, Sensitivity_test, Specificity_test=find_metrics_best_for_shuffle(fpr_test, tpr_test,cut_spe=0.95)
    roc_auc_test_2 = auc(fpr_test, tpr_test)
    bs_test_2=brier_score_loss(y_test_2, pred_proba_test_2[:, 1])
    ce = CalibrationEvaluator(y_test_2, pred_proba_test_2[:, 1], outsample=True, n_groups=2)
    hl_test_2 =ce.hosmerlemeshow().pvalue

    data_train_early=data_train.loc[data_train.ID_new.isin(sample_all_0),:]
    data_test_early = data_test.loc[data_test.ID_new.isin(sample_all_0), :]
    x_train_2_early = np.array(data_train_early[gene_list + ['AFP final']])
    x_test_2_early = np.array(data_test_early[gene_list + ['AFP final']])
    y_train_2_early = np.array(data_train_early['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_2_early = np.array(data_test_early['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_2_early = clf_2.predict_proba(x_train_2_early)
    fpr_train_early, tpr_train_early, thresholds = roc_curve(y_train_2_early, pred_proba_train_2_early[:, 1])
    Sensitivity_best_train_early, Specificity_best_train_early, Sensitivity_adjust_train_early, Specificity_adjust_train_early, Sensitivity_train_early, Specificity_train_early=find_metrics_best_for_shuffle(fpr_train_early, tpr_train_early,cut_spe=0.95)
    roc_auc_train_2_early = auc(fpr_train_early, tpr_train_early)
    bs_train_2_early =brier_score_loss(y_train_2_early , pred_proba_train_2_early[:, 1])
    ce = CalibrationEvaluator(y_train_2_early, pred_proba_train_2_early[:, 1], outsample=True, n_groups=2)
    hl_train_2_early =ce.hosmerlemeshow().pvalue
    pred_proba_test_2_early = clf_2.predict_proba(x_test_2_early)
    fpr_test_early, tpr_test_early, thresholds = roc_curve(y_test_2_early, pred_proba_test_2_early[:, 1])
    Sensitivity_best_test_early, Specificity_best_test_early, Sensitivity_adjust_test_early, Specificity_adjust_test_early, Sensitivity_test_early, Specificity_test_early=find_metrics_best_for_shuffle(fpr_test_early, tpr_test_early,cut_spe=0.95)
    roc_auc_test_2_early = auc(fpr_test_early, tpr_test_early)
    bs_test_2_early =brier_score_loss(y_test_2_early , pred_proba_test_2_early[:, 1])
    ce = CalibrationEvaluator(y_test_2_early, pred_proba_test_2_early[:, 1], outsample=True, n_groups=2)
    hl_test_2_early =ce.hosmerlemeshow().pvalue
    
    #HCC VS HD
    data_train_HCC_HD=data_train.loc[data_train.ID_new.isin(sample_HCC_HD),:]
    data_test_HCC_HD = data_test.loc[data_test.ID_new.isin(sample_HCC_HD), :]
    x_train_2_HCC_HD = np.array(data_train_HCC_HD[gene_list + ['AFP final']])
    x_test_2_HCC_HD = np.array(data_test_HCC_HD[gene_list + ['AFP final']])
    y_train_2_HCC_HD = np.array(data_train_HCC_HD['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_2_HCC_HD = np.array(data_test_HCC_HD['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_2_HCC_HD = clf_2.predict_proba(x_train_2_HCC_HD)
    fpr_train_HCC_HD, tpr_train_HCC_HD, thresholds = roc_curve(y_train_2_HCC_HD, pred_proba_train_2_HCC_HD[:, 1])
    Sensitivity_best_train_HCC_HD, Specificity_best_train_HCC_HD, Sensitivity_adjust_train_HCC_HD, Specificity_adjust_train_HCC_HD, Sensitivity_train_HCC_HD, Specificity_train_HCC_HD=find_metrics_best_for_shuffle(fpr_train_HCC_HD, tpr_train_HCC_HD,cut_spe=0.95)
    roc_auc_train_2_HCC_HD = auc(fpr_train_HCC_HD, tpr_train_HCC_HD)
    bs_train_2_HCC_HD =brier_score_loss(y_train_2_HCC_HD , pred_proba_train_2_HCC_HD[:, 1])
    ce = CalibrationEvaluator(y_train_2_HCC_HD, pred_proba_train_2_HCC_HD[:, 1], outsample=True, n_groups=2)
    hl_train_2_HCC_HD =ce.hosmerlemeshow().pvalue
    pred_proba_test_2_HCC_HD= clf_2.predict_proba(x_test_2_HCC_HD)
    fpr_test_HCC_HD, tpr_test_HCC_HD, thresholds = roc_curve(y_test_2_HCC_HD, pred_proba_test_2_HCC_HD[:, 1])
    Sensitivity_best_test_HCC_HD, Specificity_best_test_HCC_HD, Sensitivity_adjust_test_HCC_HD, Specificity_adjust_test_HCC_HD, Sensitivity_test_HCC_HD, Specificity_test_HCC_HD=find_metrics_best_for_shuffle(fpr_test_HCC_HD, tpr_test_HCC_HD,cut_spe=0.95)
    roc_auc_test_2_HCC_HD = auc(fpr_test_HCC_HD, tpr_test_HCC_HD)
    bs_test_2_HCC_HD =brier_score_loss(y_test_2_HCC_HD , pred_proba_test_2_HCC_HD[:, 1])
    ce = CalibrationEvaluator(y_test_2_HCC_HD, pred_proba_test_2_HCC_HD[:, 1], outsample=True, n_groups=2)
    hl_test_2_HCC_HD =ce.hosmerlemeshow().pvalue

    data_train_HCC_HD_0=data_train.loc[data_train.ID_new.isin(sample_HCC_HD_0),:]
    data_test_HCC_HD_0 = data_test.loc[data_test.ID_new.isin(sample_HCC_HD_0), :]
    x_train_2_HCC_HD_0 = np.array(data_train_HCC_HD_0[gene_list + ['AFP final']])
    x_test_2_HCC_HD_0 = np.array(data_test_HCC_HD_0[gene_list + ['AFP final']])
    y_train_2_HCC_HD_0 = np.array(data_train_HCC_HD_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_2_HCC_HD_0 = np.array(data_test_HCC_HD_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_2_HCC_HD_0 = clf_2.predict_proba(x_train_2_HCC_HD_0)
    fpr_train_HCC_HD_0, tpr_train_HCC_HD_0, thresholds = roc_curve(y_train_2_HCC_HD_0, pred_proba_train_2_HCC_HD_0[:, 1])
    Sensitivity_best_train_HCC_HD_0, Specificity_best_train_HCC_HD_0, Sensitivity_adjust_train_HCC_HD_0, Specificity_adjust_train_HCC_HD_0, Sensitivity_train_HCC_HD_0, Specificity_train_HCC_HD_0=find_metrics_best_for_shuffle(fpr_train_HCC_HD_0, tpr_train_HCC_HD_0,cut_spe=0.95)
    roc_auc_train_2_HCC_HD_0 = auc(fpr_train_HCC_HD_0, tpr_train_HCC_HD_0)
    bs_train_2_HCC_HD_0 =brier_score_loss(y_train_2_HCC_HD_0 , pred_proba_train_2_HCC_HD_0[:, 1])
    ce = CalibrationEvaluator(y_train_2_HCC_HD_0, pred_proba_train_2_HCC_HD_0[:, 1], outsample=True, n_groups=2)
    hl_train_2_HCC_HD_0 =ce.hosmerlemeshow().pvalue
    pred_proba_test_2_HCC_HD_0= clf_2.predict_proba(x_test_2_HCC_HD_0)
    fpr_test_HCC_HD_0, tpr_test_HCC_HD_0, thresholds = roc_curve(y_test_2_HCC_HD_0, pred_proba_test_2_HCC_HD_0[:, 1])
    Sensitivity_best_test_HCC_HD_0, Specificity_best_test_HCC_HD_0, Sensitivity_adjust_test_HCC_HD_0, Specificity_adjust_test_HCC_HD_0, Sensitivity_test_HCC_HD_0, Specificity_test_HCC_HD_0=find_metrics_best_for_shuffle(fpr_test_HCC_HD_0, tpr_test_HCC_HD_0,cut_spe=0.95)
    roc_auc_test_2_HCC_HD_0 = auc(fpr_test_HCC_HD_0, tpr_test_HCC_HD_0)
    bs_test_2_HCC_HD_0 =brier_score_loss(y_test_2_HCC_HD_0 , pred_proba_test_2_HCC_HD_0[:, 1])
    ce = CalibrationEvaluator(y_test_2_HCC_HD_0, pred_proba_test_2_HCC_HD_0[:, 1], outsample=True, n_groups=2)
    hl_test_2_HCC_HD_0 =ce.hosmerlemeshow().pvalue

    #HCC VS HBV
    data_train_HCC_HBV=data_train.loc[data_train.ID_new.isin(sample_HCC_HBV),:]
    data_test_HCC_HBV = data_test.loc[data_test.ID_new.isin(sample_HCC_HBV), :]
    x_train_2_HCC_HBV = np.array(data_train_HCC_HBV[gene_list + ['AFP final']])
    x_test_2_HCC_HBV = np.array(data_test_HCC_HBV[gene_list + ['AFP final']])
    y_train_2_HCC_HBV = np.array(data_train_HCC_HBV['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_2_HCC_HBV = np.array(data_test_HCC_HBV['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_2_HCC_HBV = clf_2.predict_proba(x_train_2_HCC_HBV)
    fpr_train_HCC_HBV, tpr_train_HCC_HBV, thresholds = roc_curve(y_train_2_HCC_HBV, pred_proba_train_2_HCC_HBV[:, 1])
    Sensitivity_best_train_HCC_HBV, Specificity_best_train_HCC_HBV, Sensitivity_adjust_train_HCC_HBV, Specificity_adjust_train_HCC_HBV, Sensitivity_train_HCC_HBV, Specificity_train_HCC_HBV=find_metrics_best_for_shuffle(fpr_train_HCC_HBV, tpr_train_HCC_HBV,cut_spe=0.95)
    roc_auc_train_2_HCC_HBV = auc(fpr_train_HCC_HBV, tpr_train_HCC_HBV)
    bs_train_2_HCC_HBV =brier_score_loss(y_train_2_HCC_HBV , pred_proba_train_2_HCC_HBV[:, 1])
    ce = CalibrationEvaluator(y_train_2_HCC_HBV, pred_proba_train_2_HCC_HBV[:, 1], outsample=True, n_groups=2)
    hl_train_2_HCC_HBV =ce.hosmerlemeshow().pvalue
    pred_proba_test_2_HCC_HBV= clf_2.predict_proba(x_test_2_HCC_HBV)
    fpr_test_HCC_HBV, tpr_test_HCC_HBV, thresholds = roc_curve(y_test_2_HCC_HBV, pred_proba_test_2_HCC_HBV[:, 1])
    Sensitivity_best_test_HCC_HBV, Specificity_best_test_HCC_HBV, Sensitivity_adjust_test_HCC_HBV, Specificity_adjust_test_HCC_HBV, Sensitivity_test_HCC_HBV, Specificity_test_HCC_HBV=find_metrics_best_for_shuffle(fpr_test_HCC_HBV, tpr_test_HCC_HBV,cut_spe=0.95)
    roc_auc_test_2_HCC_HBV = auc(fpr_test_HCC_HBV, tpr_test_HCC_HBV)
    bs_test_2_HCC_HBV =brier_score_loss(y_test_2_HCC_HBV , pred_proba_test_2_HCC_HBV[:, 1])
    ce = CalibrationEvaluator(y_test_2_HCC_HBV, pred_proba_test_2_HCC_HBV[:, 1], outsample=True, n_groups=2)
    hl_test_2_HCC_HBV =ce.hosmerlemeshow().pvalue

    data_train_HCC_HBV_0=data_train.loc[data_train.ID_new.isin(sample_HCC_HBV_0),:]
    data_test_HCC_HBV_0 = data_test.loc[data_test.ID_new.isin(sample_HCC_HBV_0), :]
    x_train_2_HCC_HBV_0 = np.array(data_train_HCC_HBV_0[gene_list + ['AFP final']])
    x_test_2_HCC_HBV_0 = np.array(data_test_HCC_HBV_0[gene_list + ['AFP final']])
    y_train_2_HCC_HBV_0 = np.array(data_train_HCC_HBV_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_2_HCC_HBV_0 = np.array(data_test_HCC_HBV_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_2_HCC_HBV_0 = clf_2.predict_proba(x_train_2_HCC_HBV_0)
    fpr_train_HCC_HBV_0, tpr_train_HCC_HBV_0, thresholds = roc_curve(y_train_2_HCC_HBV_0, pred_proba_train_2_HCC_HBV_0[:, 1])
    Sensitivity_best_train_HCC_HBV_0, Specificity_best_train_HCC_HBV_0, Sensitivity_adjust_train_HCC_HBV_0, Specificity_adjust_train_HCC_HBV_0, Sensitivity_train_HCC_HBV_0, Specificity_train_HCC_HBV_0=find_metrics_best_for_shuffle(fpr_train_HCC_HBV_0, tpr_train_HCC_HBV_0,cut_spe=0.95)
    roc_auc_train_2_HCC_HBV_0 = auc(fpr_train_HCC_HBV_0, tpr_train_HCC_HBV_0)
    bs_train_2_HCC_HBV_0 =brier_score_loss(y_train_2_HCC_HBV_0 , pred_proba_train_2_HCC_HBV_0[:, 1])
    ce = CalibrationEvaluator(y_train_2_HCC_HBV_0, pred_proba_train_2_HCC_HBV_0[:, 1], outsample=True, n_groups=2)
    hl_train_2_HCC_HBV_0 =ce.hosmerlemeshow().pvalue
    pred_proba_test_2_HCC_HBV_0= clf_2.predict_proba(x_test_2_HCC_HBV_0)
    fpr_test_HCC_HBV_0, tpr_test_HCC_HBV_0, thresholds = roc_curve(y_test_2_HCC_HBV_0, pred_proba_test_2_HCC_HBV_0[:, 1])
    Sensitivity_best_test_HCC_HBV_0, Specificity_best_test_HCC_HBV_0, Sensitivity_adjust_test_HCC_HBV_0, Specificity_adjust_test_HCC_HBV_0, Sensitivity_test_HCC_HBV_0, Specificity_test_HCC_HBV_0=find_metrics_best_for_shuffle(fpr_test_HCC_HBV_0, tpr_test_HCC_HBV_0,cut_spe=0.95)
    roc_auc_test_2_HCC_HBV_0 = auc(fpr_test_HCC_HBV_0, tpr_test_HCC_HBV_0)
    bs_test_2_HCC_HBV_0 =brier_score_loss(y_test_2_HCC_HBV_0 , pred_proba_test_2_HCC_HBV_0[:, 1])
    ce = CalibrationEvaluator(y_test_2_HCC_HBV_0, pred_proba_test_2_HCC_HBV_0[:, 1], outsample=True, n_groups=2)
    hl_test_2_HCC_HBV_0 =ce.hosmerlemeshow().pvalue

    #HCC VS LC
    data_train_HCC_LC=data_train.loc[data_train.ID_new.isin(sample_HCC_LC),:]
    data_test_HCC_LC = data_test.loc[data_test.ID_new.isin(sample_HCC_LC), :]
    x_train_2_HCC_LC = np.array(data_train_HCC_LC[gene_list + ['AFP final']])
    x_test_2_HCC_LC = np.array(data_test_HCC_LC[gene_list + ['AFP final']])
    y_train_2_HCC_LC = np.array(data_train_HCC_LC['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_2_HCC_LC = np.array(data_test_HCC_LC['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_2_HCC_LC = clf_2.predict_proba(x_train_2_HCC_LC)
    fpr_train_HCC_LC, tpr_train_HCC_LC, thresholds = roc_curve(y_train_2_HCC_LC, pred_proba_train_2_HCC_LC[:, 1])
    Sensitivity_best_train_HCC_LC, Specificity_best_train_HCC_LC, Sensitivity_adjust_train_HCC_LC, Specificity_adjust_train_HCC_LC, Sensitivity_train_HCC_LC, Specificity_train_HCC_LC=find_metrics_best_for_shuffle(fpr_train_HCC_LC, tpr_train_HCC_LC,cut_spe=0.95)
    roc_auc_train_2_HCC_LC = auc(fpr_train_HCC_LC, tpr_train_HCC_LC)
    bs_train_2_HCC_LC =brier_score_loss(y_train_2_HCC_LC , pred_proba_train_2_HCC_LC[:, 1])
    ce = CalibrationEvaluator(y_train_2_HCC_LC, pred_proba_train_2_HCC_LC[:, 1], outsample=True, n_groups=2)
    hl_train_2_HCC_LC =ce.hosmerlemeshow().pvalue
    pred_proba_test_2_HCC_LC= clf_2.predict_proba(x_test_2_HCC_LC)
    fpr_test_HCC_LC, tpr_test_HCC_LC, thresholds = roc_curve(y_test_2_HCC_LC, pred_proba_test_2_HCC_LC[:, 1])
    Sensitivity_best_test_HCC_LC, Specificity_best_test_HCC_LC, Sensitivity_adjust_test_HCC_LC, Specificity_adjust_test_HCC_LC, Sensitivity_test_HCC_LC, Specificity_test_HCC_LC=find_metrics_best_for_shuffle(fpr_test_HCC_LC, tpr_test_HCC_LC,cut_spe=0.95)
    roc_auc_test_2_HCC_LC = auc(fpr_test_HCC_LC, tpr_test_HCC_LC)
    bs_test_2_HCC_LC =brier_score_loss(y_test_2_HCC_LC , pred_proba_test_2_HCC_LC[:, 1])
    ce = CalibrationEvaluator(y_test_2_HCC_LC, pred_proba_test_2_HCC_LC[:, 1], outsample=True, n_groups=2)
    hl_test_2_HCC_LC =ce.hosmerlemeshow().pvalue

    data_train_HCC_LC_0=data_train.loc[data_train.ID_new.isin(sample_HCC_LC_0),:]
    data_test_HCC_LC_0 = data_test.loc[data_test.ID_new.isin(sample_HCC_LC_0), :]
    x_train_2_HCC_LC_0 = np.array(data_train_HCC_LC_0[gene_list + ['AFP final']])
    x_test_2_HCC_LC_0 = np.array(data_test_HCC_LC_0[gene_list + ['AFP final']])
    y_train_2_HCC_LC_0 = np.array(data_train_HCC_LC_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_2_HCC_LC_0 = np.array(data_test_HCC_LC_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_2_HCC_LC_0 = clf_2.predict_proba(x_train_2_HCC_LC_0)
    fpr_train_HCC_LC_0, tpr_train_HCC_LC_0, thresholds = roc_curve(y_train_2_HCC_LC_0, pred_proba_train_2_HCC_LC_0[:, 1])
    Sensitivity_best_train_HCC_LC_0, Specificity_best_train_HCC_LC_0, Sensitivity_adjust_train_HCC_LC_0, Specificity_adjust_train_HCC_LC_0, Sensitivity_train_HCC_LC_0, Specificity_train_HCC_LC_0=find_metrics_best_for_shuffle(fpr_train_HCC_LC_0, tpr_train_HCC_LC_0,cut_spe=0.95)
    roc_auc_train_2_HCC_LC_0 = auc(fpr_train_HCC_LC_0, tpr_train_HCC_LC_0)
    bs_train_2_HCC_LC_0 =brier_score_loss(y_train_2_HCC_LC_0 , pred_proba_train_2_HCC_LC_0[:, 1])
    ce = CalibrationEvaluator(y_train_2_HCC_LC_0, pred_proba_train_2_HCC_LC_0[:, 1], outsample=True, n_groups=2)
    hl_train_2_HCC_LC_0 =ce.hosmerlemeshow().pvalue
    pred_proba_test_2_HCC_LC_0= clf_2.predict_proba(x_test_2_HCC_LC_0)
    fpr_test_HCC_LC_0, tpr_test_HCC_LC_0, thresholds = roc_curve(y_test_2_HCC_LC_0, pred_proba_test_2_HCC_LC_0[:, 1])
    Sensitivity_best_test_HCC_LC_0, Specificity_best_test_HCC_LC_0, Sensitivity_adjust_test_HCC_LC_0, Specificity_adjust_test_HCC_LC_0, Sensitivity_test_HCC_LC_0, Specificity_test_HCC_LC_0=find_metrics_best_for_shuffle(fpr_test_HCC_LC_0, tpr_test_HCC_LC_0,cut_spe=0.95)
    roc_auc_test_2_HCC_LC_0 = auc(fpr_test_HCC_LC_0, tpr_test_HCC_LC_0)
    bs_test_2_HCC_LC_0 =brier_score_loss(y_test_2_HCC_LC_0 , pred_proba_test_2_HCC_LC_0[:, 1])
    ce = CalibrationEvaluator(y_test_2_HCC_LC_0, pred_proba_test_2_HCC_LC_0[:, 1], outsample=True, n_groups=2)
    hl_test_2_HCC_LC_0 =ce.hosmerlemeshow().pvalue

    #HCC VS HBV+LC
    data_train_HCC_HBV_LC=data_train.loc[data_train.ID_new.isin(sample_HCC_HBV_LC),:]
    data_test_HCC_HBV_LC = data_test.loc[data_test.ID_new.isin(sample_HCC_HBV_LC), :]
    x_train_2_HCC_HBV_LC = np.array(data_train_HCC_HBV_LC[gene_list + ['AFP final']])
    x_test_2_HCC_HBV_LC = np.array(data_test_HCC_HBV_LC[gene_list + ['AFP final']])
    y_train_2_HCC_HBV_LC = np.array(data_train_HCC_HBV_LC['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_2_HCC_HBV_LC = np.array(data_test_HCC_HBV_LC['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_2_HCC_HBV_LC = clf_2.predict_proba(x_train_2_HCC_HBV_LC)
    fpr_train_HCC_HBV_LC, tpr_train_HCC_HBV_LC, thresholds = roc_curve(y_train_2_HCC_HBV_LC, pred_proba_train_2_HCC_HBV_LC[:, 1])
    Sensitivity_best_train_HCC_HBV_LC, Specificity_best_train_HCC_HBV_LC, Sensitivity_adjust_train_HCC_HBV_LC, Specificity_adjust_train_HCC_HBV_LC, Sensitivity_train_HCC_HBV_LC, Specificity_train_HCC_HBV_LC=find_metrics_best_for_shuffle(fpr_train_HCC_HBV_LC, tpr_train_HCC_HBV_LC,cut_spe=0.95)
    roc_auc_train_2_HCC_HBV_LC = auc(fpr_train_HCC_HBV_LC, tpr_train_HCC_HBV_LC)
    bs_train_2_HCC_HBV_LC =brier_score_loss(y_train_2_HCC_HBV_LC , pred_proba_train_2_HCC_HBV_LC[:, 1])
    ce = CalibrationEvaluator(y_train_2_HCC_HBV_LC, pred_proba_train_2_HCC_HBV_LC[:, 1], outsample=True, n_groups=2)
    hl_train_2_HCC_HBV_LC =ce.hosmerlemeshow().pvalue
    pred_proba_test_2_HCC_HBV_LC= clf_2.predict_proba(x_test_2_HCC_HBV_LC)
    fpr_test_HCC_HBV_LC, tpr_test_HCC_HBV_LC, thresholds = roc_curve(y_test_2_HCC_HBV_LC, pred_proba_test_2_HCC_HBV_LC[:, 1])
    Sensitivity_best_test_HCC_HBV_LC, Specificity_best_test_HCC_HBV_LC, Sensitivity_adjust_test_HCC_HBV_LC, Specificity_adjust_test_HCC_HBV_LC, Sensitivity_test_HCC_HBV_LC, Specificity_test_HCC_HBV_LC=find_metrics_best_for_shuffle(fpr_test_HCC_HBV_LC, tpr_test_HCC_HBV_LC,cut_spe=0.95)
    roc_auc_test_2_HCC_HBV_LC = auc(fpr_test_HCC_HBV_LC, tpr_test_HCC_HBV_LC)
    bs_test_2_HCC_HBV_LC =brier_score_loss(y_test_2_HCC_HBV_LC , pred_proba_test_2_HCC_HBV_LC[:, 1])
    ce = CalibrationEvaluator(y_test_2_HCC_HBV_LC, pred_proba_test_2_HCC_HBV_LC[:, 1], outsample=True, n_groups=2)
    hl_test_2_HCC_HBV_LC =ce.hosmerlemeshow().pvalue

    data_train_HCC_HBV_LC_0=data_train.loc[data_train.ID_new.isin(sample_HCC_HBV_LC_0),:]
    data_test_HCC_HBV_LC_0 = data_test.loc[data_test.ID_new.isin(sample_HCC_HBV_LC_0), :]
    x_train_2_HCC_HBV_LC_0 = np.array(data_train_HCC_HBV_LC_0[gene_list + ['AFP final']])
    x_test_2_HCC_HBV_LC_0 = np.array(data_test_HCC_HBV_LC_0[gene_list + ['AFP final']])
    y_train_2_HCC_HBV_LC_0 = np.array(data_train_HCC_HBV_LC_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_2_HCC_HBV_LC_0 = np.array(data_test_HCC_HBV_LC_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_2_HCC_HBV_LC_0 = clf_2.predict_proba(x_train_2_HCC_HBV_LC_0)
    fpr_train_HCC_HBV_LC_0, tpr_train_HCC_HBV_LC_0, thresholds = roc_curve(y_train_2_HCC_HBV_LC_0, pred_proba_train_2_HCC_HBV_LC_0[:, 1])
    Sensitivity_best_train_HCC_HBV_LC_0, Specificity_best_train_HCC_HBV_LC_0, Sensitivity_adjust_train_HCC_HBV_LC_0, Specificity_adjust_train_HCC_HBV_LC_0, Sensitivity_train_HCC_HBV_LC_0, Specificity_train_HCC_HBV_LC_0=find_metrics_best_for_shuffle(fpr_train_HCC_HBV_LC_0, tpr_train_HCC_HBV_LC_0,cut_spe=0.95)
    roc_auc_train_2_HCC_HBV_LC_0 = auc(fpr_train_HCC_HBV_LC_0, tpr_train_HCC_HBV_LC_0)
    bs_train_2_HCC_HBV_LC_0 =brier_score_loss(y_train_2_HCC_HBV_LC_0 , pred_proba_train_2_HCC_HBV_LC_0[:, 1])
    ce = CalibrationEvaluator(y_train_2_HCC_HBV_LC_0, pred_proba_train_2_HCC_HBV_LC_0[:, 1], outsample=True, n_groups=2)
    hl_train_2_HCC_HBV_LC_0 =ce.hosmerlemeshow().pvalue
    pred_proba_test_2_HCC_HBV_LC_0= clf_2.predict_proba(x_test_2_HCC_HBV_LC_0)
    fpr_test_HCC_HBV_LC_0, tpr_test_HCC_HBV_LC_0, thresholds = roc_curve(y_test_2_HCC_HBV_LC_0, pred_proba_test_2_HCC_HBV_LC_0[:, 1])
    Sensitivity_best_test_HCC_HBV_LC_0, Specificity_best_test_HCC_HBV_LC_0, Sensitivity_adjust_test_HCC_HBV_LC_0, Specificity_adjust_test_HCC_HBV_LC_0, Sensitivity_test_HCC_HBV_LC_0, Specificity_test_HCC_HBV_LC_0=find_metrics_best_for_shuffle(fpr_test_HCC_HBV_LC_0, tpr_test_HCC_HBV_LC_0,cut_spe=0.95)
    roc_auc_test_2_HCC_HBV_LC_0 = auc(fpr_test_HCC_HBV_LC_0, tpr_test_HCC_HBV_LC_0)
    bs_test_2_HCC_HBV_LC_0 =brier_score_loss(y_test_2_HCC_HBV_LC_0 , pred_proba_test_2_HCC_HBV_LC_0[:, 1])
    ce = CalibrationEvaluator(y_test_2_HCC_HBV_LC_0, pred_proba_test_2_HCC_HBV_LC_0[:, 1], outsample=True, n_groups=2)
    hl_test_2_HCC_HBV_LC_0 =ce.hosmerlemeshow().pvalue

    ###################

    gene_list=[]
    x_train_afp_2 = np.array(data_train_afp[gene_list + ['AFP final']])
    x_test_afp_2 = np.array(data_test_afp[gene_list + ['AFP final']])
    y_train_afp_2 = np.array(data_train_afp['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_afp_2 = np.array(data_test_afp['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    if sample_weight == 'balanced':
        sample_weight_ = compute_sample_weight(class_weight='balanced', y=y_train_afp_2)
    clf_2_afp = clf_select(model_type, seed=0)
    clf_2_afp.fit(x_train_afp_2, y_train_afp_2, sample_weight_)

    #ALL
    pred_proba_train_afp_2 = clf_2_afp.predict_proba(x_train_afp_2)
    fpr_train_afp, tpr_train_afp, thresholds = roc_curve(y_train_afp_2, pred_proba_train_afp_2[:, 1])
    Sensitivity_best_train_afp, Specificity_best_train_afp, Sensitivity_adjust_train_afp, Specificity_adjust_train_afp, Sensitivity_train_afp, Specificity_train_afp=find_metrics_best_for_shuffle(fpr_train_afp, tpr_train_afp,cut_spe=0.95)
    roc_auc_train_afp_2 = auc(fpr_train_afp, tpr_train_afp)
    bs_train_afp_2 =brier_score_loss(y_train_afp_2 , pred_proba_train_afp_2[:, 1])
    ce = CalibrationEvaluator(y_train_afp_2, pred_proba_train_afp_2[:, 1], outsample=True, n_groups=2)
    hl_train_afp_2 =ce.hosmerlemeshow().pvalue
    pred_proba_test_afp_2 = clf_2_afp.predict_proba(x_test_afp_2)
    fpr_test_afp, tpr_test_afp, thresholds = roc_curve(y_test_afp_2, pred_proba_test_afp_2[:, 1])
    Sensitivity_best_test_afp, Specificity_best_test_afp, Sensitivity_adjust_test_afp, Specificity_adjust_test_afp, Sensitivity_test_afp, Specificity_test_afp=find_metrics_best_for_shuffle(fpr_test_afp, tpr_test_afp,cut_spe=0.95)
    roc_auc_test_afp_2 = auc(fpr_test_afp, tpr_test_afp)
    bs_test_afp_2 =brier_score_loss(y_test_afp_2 , pred_proba_test_afp_2[:, 1])
    ce = CalibrationEvaluator(y_test_afp_2, pred_proba_test_afp_2[:, 1], outsample=True, n_groups=2)
    hl_test_afp_2 =ce.hosmerlemeshow().pvalue

    data_train_afp_early=data_train_afp.loc[data_train_afp.ID_new.isin(sample_all_0),:]
    data_test_afp_early = data_test_afp.loc[data_test_afp.ID_new.isin(sample_all_0), :]
    x_train_afp_2_early = np.array(data_train_afp_early[gene_list + ['AFP final']])
    x_test_afp_2_early = np.array(data_test_afp_early[gene_list + ['AFP final']])
    y_train_afp_2_early = np.array(data_train_afp_early['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_afp_2_early = np.array(data_test_afp_early['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_afp_2_early = clf_2_afp.predict_proba(x_train_afp_2_early)
    fpr_train_afp_early, tpr_train_afp_early, thresholds = roc_curve(y_train_afp_2_early, pred_proba_train_afp_2_early[:, 1])
    Sensitivity_best_train_afp_early, Specificity_best_train_afp_early, Sensitivity_adjust_train_afp_early, Specificity_adjust_train_afp_early, Sensitivity_train_afp_early, Specificity_train_afp_early=find_metrics_best_for_shuffle(fpr_train_afp_early, tpr_train_afp_early,cut_spe=0.95)
    roc_auc_train_afp_2_early = auc(fpr_train_afp_early, tpr_train_afp_early)
    bs_train_afp_2_early =brier_score_loss(y_train_afp_2_early , pred_proba_train_afp_2_early[:, 1])
    ce = CalibrationEvaluator(y_train_afp_2_early, pred_proba_train_afp_2_early[:, 1], outsample=True, n_groups=2)
    hl_train_afp_2_early =ce.hosmerlemeshow().pvalue
    pred_proba_test_afp_2_early = clf_2_afp.predict_proba(x_test_afp_2_early)
    fpr_test_afp_early, tpr_test_afp_early, thresholds = roc_curve(y_test_afp_2_early, pred_proba_test_afp_2_early[:, 1])
    Sensitivity_best_test_afp_early, Specificity_best_test_afp_early, Sensitivity_adjust_test_afp_early, Specificity_adjust_test_afp_early, Sensitivity_test_afp_early, Specificity_test_afp_early=find_metrics_best_for_shuffle(fpr_test_afp_early, tpr_test_afp_early,cut_spe=0.95)
    roc_auc_test_afp_2_early = auc(fpr_test_afp_early, tpr_test_afp_early)
    bs_test_afp_2_early =brier_score_loss(y_test_afp_2_early , pred_proba_test_afp_2_early[:, 1])
    ce = CalibrationEvaluator(y_test_afp_2_early, pred_proba_test_afp_2_early[:, 1], outsample=True, n_groups=2)
    hl_test_afp_2_early =ce.hosmerlemeshow().pvalue

    #HCC VS HD
    data_train_afp_HCC_HD=data_train_afp.loc[data_train_afp.ID_new.isin(sample_HCC_HD),:]
    data_test_afp_HCC_HD = data_test_afp.loc[data_test_afp.ID_new.isin(sample_HCC_HD), :]
    x_train_afp_2_HCC_HD = np.array(data_train_afp_HCC_HD[gene_list + ['AFP final']])
    x_test_afp_2_HCC_HD = np.array(data_test_afp_HCC_HD[gene_list + ['AFP final']])
    y_train_afp_2_HCC_HD = np.array(data_train_afp_HCC_HD['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_afp_2_HCC_HD = np.array(data_test_afp_HCC_HD['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_afp_2_HCC_HD = clf_2_afp.predict_proba(x_train_afp_2_HCC_HD)
    fpr_train_afp_HCC_HD, tpr_train_afp_HCC_HD, thresholds = roc_curve(y_train_afp_2_HCC_HD, pred_proba_train_afp_2_HCC_HD[:, 1])
    Sensitivity_best_train_afp_HCC_HD, Specificity_best_train_afp_HCC_HD, Sensitivity_adjust_train_afp_HCC_HD, Specificity_adjust_train_afp_HCC_HD, Sensitivity_train_afp_HCC_HD, Specificity_train_afp_HCC_HD=find_metrics_best_for_shuffle(fpr_train_afp_HCC_HD, tpr_train_afp_HCC_HD,cut_spe=0.95)
    roc_auc_train_afp_2_HCC_HD = auc(fpr_train_afp_HCC_HD, tpr_train_afp_HCC_HD)
    bs_train_afp_2_HCC_HD =brier_score_loss(y_train_afp_2_HCC_HD , pred_proba_train_afp_2_HCC_HD[:, 1])
    ce = CalibrationEvaluator(y_train_afp_2_HCC_HD, pred_proba_train_afp_2_HCC_HD[:, 1], outsample=True, n_groups=2)
    hl_train_afp_2_HCC_HD =ce.hosmerlemeshow().pvalue
    pred_proba_test_afp_2_HCC_HD= clf_2_afp.predict_proba(x_test_afp_2_HCC_HD)
    fpr_test_afp_HCC_HD, tpr_test_afp_HCC_HD, thresholds = roc_curve(y_test_afp_2_HCC_HD, pred_proba_test_afp_2_HCC_HD[:, 1])
    Sensitivity_best_test_afp_HCC_HD, Specificity_best_test_afp_HCC_HD, Sensitivity_adjust_test_afp_HCC_HD, Specificity_adjust_test_afp_HCC_HD, Sensitivity_test_afp_HCC_HD, Specificity_test_afp_HCC_HD=find_metrics_best_for_shuffle(fpr_test_afp_HCC_HD, tpr_test_afp_HCC_HD,cut_spe=0.95)
    roc_auc_test_afp_2_HCC_HD = auc(fpr_test_afp_HCC_HD, tpr_test_afp_HCC_HD)
    bs_test_afp_2_HCC_HD =brier_score_loss(y_test_afp_2_HCC_HD , pred_proba_test_afp_2_HCC_HD[:, 1])
    ce = CalibrationEvaluator(y_test_afp_2_HCC_HD, pred_proba_test_afp_2_HCC_HD[:, 1], outsample=True, n_groups=2)
    hl_test_afp_2_HCC_HD =ce.hosmerlemeshow().pvalue

    data_train_afp_HCC_HD_0=data_train_afp.loc[data_train_afp.ID_new.isin(sample_HCC_HD_0),:]
    data_test_afp_HCC_HD_0 = data_test_afp.loc[data_test_afp.ID_new.isin(sample_HCC_HD_0), :]
    x_train_afp_2_HCC_HD_0 = np.array(data_train_afp_HCC_HD_0[gene_list + ['AFP final']])
    x_test_afp_2_HCC_HD_0 = np.array(data_test_afp_HCC_HD_0[gene_list + ['AFP final']])
    y_train_afp_2_HCC_HD_0 = np.array(data_train_afp_HCC_HD_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_afp_2_HCC_HD_0 = np.array(data_test_afp_HCC_HD_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_afp_2_HCC_HD_0 = clf_2_afp.predict_proba(x_train_afp_2_HCC_HD_0)
    fpr_train_afp_HCC_HD_0, tpr_train_afp_HCC_HD_0, thresholds = roc_curve(y_train_afp_2_HCC_HD_0, pred_proba_train_afp_2_HCC_HD_0[:, 1])
    Sensitivity_best_train_afp_HCC_HD_0, Specificity_best_train_afp_HCC_HD_0, Sensitivity_adjust_train_afp_HCC_HD_0, Specificity_adjust_train_afp_HCC_HD_0, Sensitivity_train_afp_HCC_HD_0, Specificity_train_afp_HCC_HD_0=find_metrics_best_for_shuffle(fpr_train_afp_HCC_HD_0, tpr_train_afp_HCC_HD_0,cut_spe=0.95)
    roc_auc_train_afp_2_HCC_HD_0 = auc(fpr_train_afp_HCC_HD_0, tpr_train_afp_HCC_HD_0)
    bs_train_afp_2_HCC_HD_0 =brier_score_loss(y_train_afp_2_HCC_HD_0 , pred_proba_train_afp_2_HCC_HD_0[:, 1])
    ce = CalibrationEvaluator(y_train_afp_2_HCC_HD_0, pred_proba_train_afp_2_HCC_HD_0[:, 1], outsample=True, n_groups=2)
    hl_train_afp_2_HCC_HD_0 =ce.hosmerlemeshow().pvalue
    pred_proba_test_afp_2_HCC_HD_0= clf_2_afp.predict_proba(x_test_afp_2_HCC_HD_0)
    fpr_test_afp_HCC_HD_0, tpr_test_afp_HCC_HD_0, thresholds = roc_curve(y_test_afp_2_HCC_HD_0, pred_proba_test_afp_2_HCC_HD_0[:, 1])
    Sensitivity_best_test_afp_HCC_HD_0, Specificity_best_test_afp_HCC_HD_0, Sensitivity_adjust_test_afp_HCC_HD_0, Specificity_adjust_test_afp_HCC_HD_0, Sensitivity_test_afp_HCC_HD_0, Specificity_test_afp_HCC_HD_0=find_metrics_best_for_shuffle(fpr_test_afp_HCC_HD_0, tpr_test_afp_HCC_HD_0,cut_spe=0.95)
    roc_auc_test_afp_2_HCC_HD_0 = auc(fpr_test_afp_HCC_HD_0, tpr_test_afp_HCC_HD_0)
    bs_test_afp_2_HCC_HD_0 =brier_score_loss(y_test_afp_2_HCC_HD_0 , pred_proba_test_afp_2_HCC_HD_0[:, 1])
    ce = CalibrationEvaluator(y_test_afp_2_HCC_HD_0, pred_proba_test_afp_2_HCC_HD_0[:, 1], outsample=True, n_groups=2)
    hl_test_afp_2_HCC_HD_0 =ce.hosmerlemeshow().pvalue

    #HCC VS HBV
    data_train_afp_HCC_HBV=data_train_afp.loc[data_train_afp.ID_new.isin(sample_HCC_HBV),:]
    data_test_afp_HCC_HBV = data_test_afp.loc[data_test_afp.ID_new.isin(sample_HCC_HBV), :]
    x_train_afp_2_HCC_HBV = np.array(data_train_afp_HCC_HBV[gene_list + ['AFP final']])
    x_test_afp_2_HCC_HBV = np.array(data_test_afp_HCC_HBV[gene_list + ['AFP final']])
    y_train_afp_2_HCC_HBV = np.array(data_train_afp_HCC_HBV['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_afp_2_HCC_HBV = np.array(data_test_afp_HCC_HBV['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_afp_2_HCC_HBV = clf_2_afp.predict_proba(x_train_afp_2_HCC_HBV)
    fpr_train_afp_HCC_HBV, tpr_train_afp_HCC_HBV, thresholds = roc_curve(y_train_afp_2_HCC_HBV, pred_proba_train_afp_2_HCC_HBV[:, 1])
    Sensitivity_best_train_afp_HCC_HBV, Specificity_best_train_afp_HCC_HBV, Sensitivity_adjust_train_afp_HCC_HBV, Specificity_adjust_train_afp_HCC_HBV, Sensitivity_train_afp_HCC_HBV, Specificity_train_afp_HCC_HBV=find_metrics_best_for_shuffle(fpr_train_afp_HCC_HBV, tpr_train_afp_HCC_HBV,cut_spe=0.95)
    roc_auc_train_afp_2_HCC_HBV = auc(fpr_train_afp_HCC_HBV, tpr_train_afp_HCC_HBV)
    bs_train_afp_2_HCC_HBV =brier_score_loss(y_train_afp_2_HCC_HBV , pred_proba_train_afp_2_HCC_HBV[:, 1])
    ce = CalibrationEvaluator(y_train_afp_2_HCC_HBV, pred_proba_train_afp_2_HCC_HBV[:, 1], outsample=True, n_groups=2)
    hl_train_afp_2_HCC_HBV =ce.hosmerlemeshow().pvalue
    pred_proba_test_afp_2_HCC_HBV= clf_2_afp.predict_proba(x_test_afp_2_HCC_HBV)
    fpr_test_afp_HCC_HBV, tpr_test_afp_HCC_HBV, thresholds = roc_curve(y_test_afp_2_HCC_HBV, pred_proba_test_afp_2_HCC_HBV[:, 1])
    Sensitivity_best_test_afp_HCC_HBV, Specificity_best_test_afp_HCC_HBV, Sensitivity_adjust_test_afp_HCC_HBV, Specificity_adjust_test_afp_HCC_HBV, Sensitivity_test_afp_HCC_HBV, Specificity_test_afp_HCC_HBV=find_metrics_best_for_shuffle(fpr_test_afp_HCC_HBV, tpr_test_afp_HCC_HBV,cut_spe=0.95)
    roc_auc_test_afp_2_HCC_HBV = auc(fpr_test_afp_HCC_HBV, tpr_test_afp_HCC_HBV)
    bs_test_afp_2_HCC_HBV =brier_score_loss(y_test_afp_2_HCC_HBV , pred_proba_test_afp_2_HCC_HBV[:, 1])
    ce = CalibrationEvaluator(y_test_afp_2_HCC_HBV, pred_proba_test_afp_2_HCC_HBV[:, 1], outsample=True, n_groups=2)
    hl_test_afp_2_HCC_HBV =ce.hosmerlemeshow().pvalue

    data_train_afp_HCC_HBV_0=data_train_afp.loc[data_train_afp.ID_new.isin(sample_HCC_HBV_0),:]
    data_test_afp_HCC_HBV_0 = data_test_afp.loc[data_test_afp.ID_new.isin(sample_HCC_HBV_0), :]
    x_train_afp_2_HCC_HBV_0 = np.array(data_train_afp_HCC_HBV_0[gene_list + ['AFP final']])
    x_test_afp_2_HCC_HBV_0 = np.array(data_test_afp_HCC_HBV_0[gene_list + ['AFP final']])
    y_train_afp_2_HCC_HBV_0 = np.array(data_train_afp_HCC_HBV_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_afp_2_HCC_HBV_0 = np.array(data_test_afp_HCC_HBV_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_afp_2_HCC_HBV_0 = clf_2_afp.predict_proba(x_train_afp_2_HCC_HBV_0)
    fpr_train_afp_HCC_HBV_0, tpr_train_afp_HCC_HBV_0, thresholds = roc_curve(y_train_afp_2_HCC_HBV_0, pred_proba_train_afp_2_HCC_HBV_0[:, 1])
    Sensitivity_best_train_afp_HCC_HBV_0, Specificity_best_train_afp_HCC_HBV_0, Sensitivity_adjust_train_afp_HCC_HBV_0, Specificity_adjust_train_afp_HCC_HBV_0, Sensitivity_train_afp_HCC_HBV_0, Specificity_train_afp_HCC_HBV_0=find_metrics_best_for_shuffle(fpr_train_afp_HCC_HBV_0, tpr_train_afp_HCC_HBV_0,cut_spe=0.95)
    roc_auc_train_afp_2_HCC_HBV_0 = auc(fpr_train_afp_HCC_HBV_0, tpr_train_afp_HCC_HBV_0)
    bs_train_afp_2_HCC_HBV_0 =brier_score_loss(y_train_afp_2_HCC_HBV_0 , pred_proba_train_afp_2_HCC_HBV_0[:, 1])
    ce = CalibrationEvaluator(y_train_afp_2_HCC_HBV_0, pred_proba_train_afp_2_HCC_HBV_0[:, 1], outsample=True, n_groups=2)
    hl_train_afp_2_HCC_HBV_0 =ce.hosmerlemeshow().pvalue
    pred_proba_test_afp_2_HCC_HBV_0= clf_2_afp.predict_proba(x_test_afp_2_HCC_HBV_0)
    fpr_test_afp_HCC_HBV_0, tpr_test_afp_HCC_HBV_0, thresholds = roc_curve(y_test_afp_2_HCC_HBV_0, pred_proba_test_afp_2_HCC_HBV_0[:, 1])
    Sensitivity_best_test_afp_HCC_HBV_0, Specificity_best_test_afp_HCC_HBV_0, Sensitivity_adjust_test_afp_HCC_HBV_0, Specificity_adjust_test_afp_HCC_HBV_0, Sensitivity_test_afp_HCC_HBV_0, Specificity_test_afp_HCC_HBV_0=find_metrics_best_for_shuffle(fpr_test_afp_HCC_HBV_0, tpr_test_afp_HCC_HBV_0,cut_spe=0.95)
    roc_auc_test_afp_2_HCC_HBV_0 = auc(fpr_test_afp_HCC_HBV_0, tpr_test_afp_HCC_HBV_0)
    bs_test_afp_2_HCC_HBV_0 =brier_score_loss(y_test_afp_2_HCC_HBV_0 , pred_proba_test_afp_2_HCC_HBV_0[:, 1])
    ce = CalibrationEvaluator(y_test_afp_2_HCC_HBV_0, pred_proba_test_afp_2_HCC_HBV_0[:, 1], outsample=True, n_groups=2)
    hl_test_afp_2_HCC_HBV_0 =ce.hosmerlemeshow().pvalue

    #HCC VS LC
    data_train_afp_HCC_LC=data_train_afp.loc[data_train_afp.ID_new.isin(sample_HCC_LC),:]
    data_test_afp_HCC_LC = data_test_afp.loc[data_test_afp.ID_new.isin(sample_HCC_LC), :]
    x_train_afp_2_HCC_LC = np.array(data_train_afp_HCC_LC[gene_list + ['AFP final']])
    x_test_afp_2_HCC_LC = np.array(data_test_afp_HCC_LC[gene_list + ['AFP final']])
    y_train_afp_2_HCC_LC = np.array(data_train_afp_HCC_LC['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_afp_2_HCC_LC = np.array(data_test_afp_HCC_LC['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_afp_2_HCC_LC = clf_2_afp.predict_proba(x_train_afp_2_HCC_LC)
    fpr_train_afp_HCC_LC, tpr_train_afp_HCC_LC, thresholds = roc_curve(y_train_afp_2_HCC_LC, pred_proba_train_afp_2_HCC_LC[:, 1])
    Sensitivity_best_train_afp_HCC_LC, Specificity_best_train_afp_HCC_LC, Sensitivity_adjust_train_afp_HCC_LC, Specificity_adjust_train_afp_HCC_LC, Sensitivity_train_afp_HCC_LC, Specificity_train_afp_HCC_LC=find_metrics_best_for_shuffle(fpr_train_afp_HCC_LC, tpr_train_afp_HCC_LC,cut_spe=0.95)
    roc_auc_train_afp_2_HCC_LC = auc(fpr_train_afp_HCC_LC, tpr_train_afp_HCC_LC)
    bs_train_afp_2_HCC_LC =brier_score_loss(y_train_afp_2_HCC_LC , pred_proba_train_afp_2_HCC_LC[:, 1])
    ce = CalibrationEvaluator(y_train_afp_2_HCC_LC, pred_proba_train_afp_2_HCC_LC[:, 1], outsample=True, n_groups=2)
    hl_train_afp_2_HCC_LC =ce.hosmerlemeshow().pvalue
    pred_proba_test_afp_2_HCC_LC= clf_2_afp.predict_proba(x_test_afp_2_HCC_LC)
    fpr_test_afp_HCC_LC, tpr_test_afp_HCC_LC, thresholds = roc_curve(y_test_afp_2_HCC_LC, pred_proba_test_afp_2_HCC_LC[:, 1])
    Sensitivity_best_test_afp_HCC_LC, Specificity_best_test_afp_HCC_LC, Sensitivity_adjust_test_afp_HCC_LC, Specificity_adjust_test_afp_HCC_LC, Sensitivity_test_afp_HCC_LC, Specificity_test_afp_HCC_LC=find_metrics_best_for_shuffle(fpr_test_afp_HCC_LC, tpr_test_afp_HCC_LC,cut_spe=0.95)
    roc_auc_test_afp_2_HCC_LC = auc(fpr_test_afp_HCC_LC, tpr_test_afp_HCC_LC)
    bs_test_afp_2_HCC_LC =brier_score_loss(y_test_afp_2_HCC_LC , pred_proba_test_afp_2_HCC_LC[:, 1])
    ce = CalibrationEvaluator(y_test_afp_2_HCC_LC, pred_proba_test_afp_2_HCC_LC[:, 1], outsample=True, n_groups=2)
    hl_test_afp_2_HCC_LC =ce.hosmerlemeshow().pvalue

    data_train_afp_HCC_LC_0=data_train_afp.loc[data_train_afp.ID_new.isin(sample_HCC_LC_0),:]
    data_test_afp_HCC_LC_0 = data_test_afp.loc[data_test_afp.ID_new.isin(sample_HCC_LC_0), :]
    x_train_afp_2_HCC_LC_0 = np.array(data_train_afp_HCC_LC_0[gene_list + ['AFP final']])
    x_test_afp_2_HCC_LC_0 = np.array(data_test_afp_HCC_LC_0[gene_list + ['AFP final']])
    y_train_afp_2_HCC_LC_0 = np.array(data_train_afp_HCC_LC_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_afp_2_HCC_LC_0 = np.array(data_test_afp_HCC_LC_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_afp_2_HCC_LC_0 = clf_2_afp.predict_proba(x_train_afp_2_HCC_LC_0)
    fpr_train_afp_HCC_LC_0, tpr_train_afp_HCC_LC_0, thresholds = roc_curve(y_train_afp_2_HCC_LC_0, pred_proba_train_afp_2_HCC_LC_0[:, 1])
    Sensitivity_best_train_afp_HCC_LC_0, Specificity_best_train_afp_HCC_LC_0, Sensitivity_adjust_train_afp_HCC_LC_0, Specificity_adjust_train_afp_HCC_LC_0, Sensitivity_train_afp_HCC_LC_0, Specificity_train_afp_HCC_LC_0=find_metrics_best_for_shuffle(fpr_train_afp_HCC_LC_0, tpr_train_afp_HCC_LC_0,cut_spe=0.95)
    roc_auc_train_afp_2_HCC_LC_0 = auc(fpr_train_afp_HCC_LC_0, tpr_train_afp_HCC_LC_0)
    bs_train_afp_2_HCC_LC_0 =brier_score_loss(y_train_afp_2_HCC_LC_0 , pred_proba_train_afp_2_HCC_LC_0[:, 1])
    ce = CalibrationEvaluator(y_train_afp_2_HCC_LC_0, pred_proba_train_afp_2_HCC_LC_0[:, 1], outsample=True, n_groups=2)
    hl_train_afp_2_HCC_LC_0 =ce.hosmerlemeshow().pvalue
    pred_proba_test_afp_2_HCC_LC_0= clf_2_afp.predict_proba(x_test_afp_2_HCC_LC_0)
    fpr_test_afp_HCC_LC_0, tpr_test_afp_HCC_LC_0, thresholds = roc_curve(y_test_afp_2_HCC_LC_0, pred_proba_test_afp_2_HCC_LC_0[:, 1])
    Sensitivity_best_test_afp_HCC_LC_0, Specificity_best_test_afp_HCC_LC_0, Sensitivity_adjust_test_afp_HCC_LC_0, Specificity_adjust_test_afp_HCC_LC_0, Sensitivity_test_afp_HCC_LC_0, Specificity_test_afp_HCC_LC_0=find_metrics_best_for_shuffle(fpr_test_afp_HCC_LC_0, tpr_test_afp_HCC_LC_0,cut_spe=0.95)
    roc_auc_test_afp_2_HCC_LC_0 = auc(fpr_test_afp_HCC_LC_0, tpr_test_afp_HCC_LC_0)
    bs_test_afp_2_HCC_LC_0 =brier_score_loss(y_test_afp_2_HCC_LC_0 , pred_proba_test_afp_2_HCC_LC_0[:, 1])
    ce = CalibrationEvaluator(y_test_afp_2_HCC_LC_0, pred_proba_test_afp_2_HCC_LC_0[:, 1], outsample=True, n_groups=2)
    hl_test_afp_2_HCC_LC_0 =ce.hosmerlemeshow().pvalue

    #HCC VS HBV+LC
    data_train_afp_HCC_HBV_LC=data_train_afp.loc[data_train_afp.ID_new.isin(sample_HCC_HBV_LC),:]
    data_test_afp_HCC_HBV_LC = data_test_afp.loc[data_test_afp.ID_new.isin(sample_HCC_HBV_LC), :]
    x_train_afp_2_HCC_HBV_LC = np.array(data_train_afp_HCC_HBV_LC[gene_list + ['AFP final']])
    x_test_afp_2_HCC_HBV_LC = np.array(data_test_afp_HCC_HBV_LC[gene_list + ['AFP final']])
    y_train_afp_2_HCC_HBV_LC = np.array(data_train_afp_HCC_HBV_LC['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_afp_2_HCC_HBV_LC = np.array(data_test_afp_HCC_HBV_LC['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_afp_2_HCC_HBV_LC = clf_2_afp.predict_proba(x_train_afp_2_HCC_HBV_LC)
    fpr_train_afp_HCC_HBV_LC, tpr_train_afp_HCC_HBV_LC, thresholds = roc_curve(y_train_afp_2_HCC_HBV_LC, pred_proba_train_afp_2_HCC_HBV_LC[:, 1])
    Sensitivity_best_train_afp_HCC_HBV_LC, Specificity_best_train_afp_HCC_HBV_LC, Sensitivity_adjust_train_afp_HCC_HBV_LC, Specificity_adjust_train_afp_HCC_HBV_LC, Sensitivity_train_afp_HCC_HBV_LC, Specificity_train_afp_HCC_HBV_LC=find_metrics_best_for_shuffle(fpr_train_afp_HCC_HBV_LC, tpr_train_afp_HCC_HBV_LC,cut_spe=0.95)
    roc_auc_train_afp_2_HCC_HBV_LC = auc(fpr_train_afp_HCC_HBV_LC, tpr_train_afp_HCC_HBV_LC)
    bs_train_afp_2_HCC_HBV_LC  =brier_score_loss(y_train_afp_2_HCC_HBV_LC , pred_proba_train_afp_2_HCC_HBV_LC[:, 1])
    ce = CalibrationEvaluator(y_train_afp_2_HCC_HBV_LC, pred_proba_train_afp_2_HCC_HBV_LC[:, 1], outsample=True, n_groups=2)
    hl_train_afp_2_HCC_HBV_LC  =ce.hosmerlemeshow().pvalue
    pred_proba_test_afp_2_HCC_HBV_LC= clf_2_afp.predict_proba(x_test_afp_2_HCC_HBV_LC)
    fpr_test_afp_HCC_HBV_LC, tpr_test_afp_HCC_HBV_LC, thresholds = roc_curve(y_test_afp_2_HCC_HBV_LC, pred_proba_test_afp_2_HCC_HBV_LC[:, 1])
    Sensitivity_best_test_afp_HCC_HBV_LC, Specificity_best_test_afp_HCC_HBV_LC, Sensitivity_adjust_test_afp_HCC_HBV_LC, Specificity_adjust_test_afp_HCC_HBV_LC, Sensitivity_test_afp_HCC_HBV_LC, Specificity_test_afp_HCC_HBV_LC=find_metrics_best_for_shuffle(fpr_test_afp_HCC_HBV_LC, tpr_test_afp_HCC_HBV_LC,cut_spe=0.95)
    roc_auc_test_afp_2_HCC_HBV_LC = auc(fpr_test_afp_HCC_HBV_LC, tpr_test_afp_HCC_HBV_LC)
    bs_test_afp_2_HCC_HBV_LC   =brier_score_loss(y_test_afp_2_HCC_HBV_LC , pred_proba_test_afp_2_HCC_HBV_LC[:, 1])
    ce = CalibrationEvaluator(y_test_afp_2_HCC_HBV_LC, pred_proba_test_afp_2_HCC_HBV_LC[:, 1], outsample=True, n_groups=2)
    hl_test_afp_2_HCC_HBV_LC   =ce.hosmerlemeshow().pvalue

    data_train_afp_HCC_HBV_LC_0=data_train_afp.loc[data_train_afp.ID_new.isin(sample_HCC_HBV_LC_0),:]
    data_test_afp_HCC_HBV_LC_0 = data_test_afp.loc[data_test_afp.ID_new.isin(sample_HCC_HBV_LC_0), :]
    x_train_afp_2_HCC_HBV_LC_0 = np.array(data_train_afp_HCC_HBV_LC_0[gene_list + ['AFP final']])
    x_test_afp_2_HCC_HBV_LC_0 = np.array(data_test_afp_HCC_HBV_LC_0[gene_list + ['AFP final']])
    y_train_afp_2_HCC_HBV_LC_0 = np.array(data_train_afp_HCC_HBV_LC_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
    y_test_afp_2_HCC_HBV_LC_0 = np.array(data_test_afp_HCC_HBV_LC_0['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

    pred_proba_train_afp_2_HCC_HBV_LC_0 = clf_2_afp.predict_proba(x_train_afp_2_HCC_HBV_LC_0)
    fpr_train_afp_HCC_HBV_LC_0, tpr_train_afp_HCC_HBV_LC_0, thresholds = roc_curve(y_train_afp_2_HCC_HBV_LC_0, pred_proba_train_afp_2_HCC_HBV_LC_0[:, 1])
    Sensitivity_best_train_afp_HCC_HBV_LC_0, Specificity_best_train_afp_HCC_HBV_LC_0, Sensitivity_adjust_train_afp_HCC_HBV_LC_0, Specificity_adjust_train_afp_HCC_HBV_LC_0, Sensitivity_train_afp_HCC_HBV_LC_0, Specificity_train_afp_HCC_HBV_LC_0=find_metrics_best_for_shuffle(fpr_train_afp_HCC_HBV_LC_0, tpr_train_afp_HCC_HBV_LC_0,cut_spe=0.95)
    roc_auc_train_afp_2_HCC_HBV_LC_0 = auc(fpr_train_afp_HCC_HBV_LC_0, tpr_train_afp_HCC_HBV_LC_0)
    bs_train_afp_2_HCC_HBV_LC_0  =brier_score_loss(y_train_afp_2_HCC_HBV_LC_0 , pred_proba_train_afp_2_HCC_HBV_LC_0[:, 1])
    ce = CalibrationEvaluator(y_train_afp_2_HCC_HBV_LC_0, pred_proba_train_afp_2_HCC_HBV_LC_0[:, 1], outsample=True, n_groups=2)
    hl_train_afp_2_HCC_HBV_LC_0  =ce.hosmerlemeshow().pvalue
    pred_proba_test_afp_2_HCC_HBV_LC_0= clf_2_afp.predict_proba(x_test_afp_2_HCC_HBV_LC_0)
    fpr_test_afp_HCC_HBV_LC_0, tpr_test_afp_HCC_HBV_LC_0, thresholds = roc_curve(y_test_afp_2_HCC_HBV_LC_0, pred_proba_test_afp_2_HCC_HBV_LC_0[:, 1])
    Sensitivity_best_test_afp_HCC_HBV_LC_0, Specificity_best_test_afp_HCC_HBV_LC_0, Sensitivity_adjust_test_afp_HCC_HBV_LC_0, Specificity_adjust_test_afp_HCC_HBV_LC_0, Sensitivity_test_afp_HCC_HBV_LC_0, Specificity_test_afp_HCC_HBV_LC_0=find_metrics_best_for_shuffle(fpr_test_afp_HCC_HBV_LC_0, tpr_test_afp_HCC_HBV_LC_0,cut_spe=0.95)
    roc_auc_test_afp_2_HCC_HBV_LC_0 = auc(fpr_test_afp_HCC_HBV_LC_0, tpr_test_afp_HCC_HBV_LC_0)
    bs_test_afp_2_HCC_HBV_LC_0 =brier_score_loss(y_test_afp_2_HCC_HBV_LC_0 , pred_proba_test_afp_2_HCC_HBV_LC_0[:, 1])
    ce = CalibrationEvaluator(y_test_afp_2_HCC_HBV_LC_0, pred_proba_test_afp_2_HCC_HBV_LC_0[:, 1], outsample=True, n_groups=2)
    hl_test_afp_2_HCC_HBV_LC_0  =ce.hosmerlemeshow().pvalue

    num=0
    result.loc[num,'roc_auc_train']=roc_auc_train_2
    result.loc[num,'bs_train']=bs_train_2
    result.loc[num,'hl_train']=hl_train_2
    result.loc[num,'Sensitivity_best_train']=Sensitivity_best_train
    result.loc[num, 'Specificity_best_train'] = Specificity_best_train
    result.loc[num, 'Sensitivity_adjust_train'] = Sensitivity_adjust_train
    result.loc[num, 'Specificity_adjust_train'] = Specificity_adjust_train
    result.loc[num, 'y_train'] = ','.join(np.array(y_train_2).astype('str').tolist())
    result.loc[num, 'predict_proba_train'] = ','.join(np.array(pred_proba_train_2[:, 1]).astype('str').tolist())
    result.loc[num,'roc_auc_test']=roc_auc_test_2
    result.loc[num,'bs_test']=bs_test_2
    result.loc[num, 'hl_test'] = hl_test_2
    result.loc[num,'Sensitivity_best_test']=Sensitivity_best_test
    result.loc[num, 'Specificity_best_test'] = Specificity_best_test
    result.loc[num, 'Sensitivity_adjust_test'] = Sensitivity_adjust_test
    result.loc[num, 'Specificity_adjust_test'] = Specificity_adjust_test
    result.loc[num, 'y_test'] = ','.join(np.array(y_test_2).astype('str').tolist())
    result.loc[num, 'predict_proba_test'] = ','.join(np.array(pred_proba_test_2[:, 1]).astype('str').tolist())

    result.loc[num,'roc_auc_train_HCC_HD']=roc_auc_train_2_HCC_HD
    result.loc[num,'bs_train_HCC_HD']=bs_train_2_HCC_HD
    result.loc[num, 'hl_train_HCC_HD'] = hl_train_2_HCC_HD
    result.loc[num,'Sensitivity_best_train_HCC_HD']=Sensitivity_best_train_HCC_HD
    result.loc[num, 'Specificity_best_train_HCC_HD'] = Specificity_best_train_HCC_HD
    result.loc[num, 'Sensitivity_adjust_train_HCC_HD'] = Sensitivity_adjust_train_HCC_HD
    result.loc[num, 'Specificity_adjust_train_HCC_HD'] = Specificity_adjust_train_HCC_HD
    result.loc[num, 'y_train_HCC_HD'] = ','.join(np.array(y_train_2_HCC_HD).astype('str').tolist())
    result.loc[num, 'predict_proba_train_HCC_HD'] = ','.join(np.array(pred_proba_train_2_HCC_HD[:, 1]).astype('str').tolist())
    result.loc[num,'roc_auc_test_HCC_HD']=roc_auc_test_2_HCC_HD
    result.loc[num,'bs_test_HCC_HD']=bs_test_2_HCC_HD
    result.loc[num, 'hl_test_HCC_HD'] = hl_test_2_HCC_HD
    result.loc[num,'Sensitivity_best_test_HCC_HD']=Sensitivity_best_test_HCC_HD
    result.loc[num, 'Specificity_best_test_HCC_HD'] = Specificity_best_test_HCC_HD
    result.loc[num, 'Sensitivity_adjust_test_HCC_HD'] = Sensitivity_adjust_test_HCC_HD
    result.loc[num, 'Specificity_adjust_test_HCC_HD'] = Specificity_adjust_test_HCC_HD
    result.loc[num, 'y_test_HCC_HD'] = ','.join(np.array(y_test_2_HCC_HD).astype('str').tolist())
    result.loc[num, 'predict_proba_test_HCC_HD'] = ','.join(np.array(pred_proba_test_2_HCC_HD[:, 1]).astype('str').tolist())

    result.loc[num,'roc_auc_train_HCC_HBV']=roc_auc_train_2_HCC_HBV
    result.loc[num,'bs_train_HCC_HBV']=bs_train_2_HCC_HBV
    result.loc[num,'hl_train_HCC_HBV']=hl_train_2_HCC_HBV
    result.loc[num,'Sensitivity_best_train_HCC_HBV']=Sensitivity_best_train_HCC_HBV
    result.loc[num, 'Specificity_best_train_HCC_HBV'] = Specificity_best_train_HCC_HBV
    result.loc[num, 'Sensitivity_adjust_train_HCC_HBV'] = Sensitivity_adjust_train_HCC_HBV
    result.loc[num, 'Specificity_adjust_train_HCC_HBV'] = Specificity_adjust_train_HCC_HBV
    result.loc[num, 'y_train_HCC_HBV'] = ','.join(np.array(y_train_2_HCC_HBV).astype('str').tolist())
    result.loc[num, 'predict_proba_train_HCC_HBV'] = ','.join(np.array(pred_proba_train_2_HCC_HBV[:, 1]).astype('str').tolist())
    result.loc[num,'roc_auc_test_HCC_HBV']=roc_auc_test_2_HCC_HBV
    result.loc[num,'bs_test_HCC_HBV']=bs_test_2_HCC_HBV
    result.loc[num, 'hl_test_HCC_HBV'] = hl_test_2_HCC_HBV
    result.loc[num,'Sensitivity_best_test_HCC_HBV']=Sensitivity_best_test_HCC_HBV
    result.loc[num, 'Specificity_best_test_HCC_HBV'] = Specificity_best_test_HCC_HBV
    result.loc[num, 'Sensitivity_adjust_test_HCC_HBV'] = Sensitivity_adjust_test_HCC_HBV
    result.loc[num, 'Specificity_adjust_test_HCC_HBV'] = Specificity_adjust_test_HCC_HBV
    result.loc[num, 'y_test_HCC_HBV'] = ','.join(np.array(y_test_2_HCC_HBV).astype('str').tolist())
    result.loc[num, 'predict_proba_test_HCC_HBV'] = ','.join(np.array(pred_proba_test_2_HCC_HBV[:, 1]).astype('str').tolist())

    result.loc[num,'roc_auc_train_HCC_LC']=roc_auc_train_2_HCC_LC
    result.loc[num,'bs_train_HCC_LC']=bs_train_2_HCC_LC
    result.loc[num, 'hl_train_HCC_LC'] = hl_train_2_HCC_LC
    result.loc[num,'Sensitivity_best_train_HCC_LC']=Sensitivity_best_train_HCC_LC
    result.loc[num, 'Specificity_best_train_HCC_LC'] = Specificity_best_train_HCC_LC
    result.loc[num, 'Sensitivity_adjust_train_HCC_LC'] = Sensitivity_adjust_train_HCC_LC
    result.loc[num, 'Specificity_adjust_train_HCC_LC'] = Specificity_adjust_train_HCC_LC
    result.loc[num, 'y_train_HCC_LC'] = ','.join(np.array(y_train_2_HCC_LC).astype('str').tolist())
    result.loc[num, 'predict_proba_train_HCC_LC'] = ','.join(np.array(pred_proba_train_2_HCC_LC[:, 1]).astype('str').tolist())
    result.loc[num,'roc_auc_test_HCC_LC']=roc_auc_test_2_HCC_LC
    result.loc[num,'bs_test_HCC_LC']=bs_test_2_HCC_LC
    result.loc[num, 'hl_test_HCC_LC'] = hl_test_2_HCC_LC
    result.loc[num,'Sensitivity_best_test_HCC_LC']=Sensitivity_best_test_HCC_LC
    result.loc[num, 'Specificity_best_test_HCC_LC'] = Specificity_best_test_HCC_LC
    result.loc[num, 'Sensitivity_adjust_test_HCC_LC'] = Sensitivity_adjust_test_HCC_LC
    result.loc[num, 'Specificity_adjust_test_HCC_LC'] = Specificity_adjust_test_HCC_LC
    result.loc[num, 'y_test_HCC_LC'] = ','.join(np.array(y_test_2_HCC_LC).astype('str').tolist())
    result.loc[num, 'predict_proba_test_HCC_LC'] = ','.join(np.array(pred_proba_test_2_HCC_LC[:, 1]).astype('str').tolist())

    result.loc[num,'roc_auc_train_HCC_HBV_LC']=roc_auc_train_2_HCC_HBV_LC
    result.loc[num,'bs_train_HCC_HBV_LC']=bs_train_2_HCC_HBV_LC
    result.loc[num, 'hl_train_HCC_HBV_LC'] = hl_train_2_HCC_HBV_LC
    result.loc[num,'Sensitivity_best_train_HCC_HBV_LC']=Sensitivity_best_train_HCC_HBV_LC
    result.loc[num, 'Specificity_best_train_HCC_HBV_LC'] = Specificity_best_train_HCC_HBV_LC
    result.loc[num, 'Sensitivity_adjust_train_HCC_HBV_LC'] = Sensitivity_adjust_train_HCC_HBV_LC
    result.loc[num, 'Specificity_adjust_train_HCC_HBV_LC'] = Specificity_adjust_train_HCC_HBV_LC
    result.loc[num, 'y_train_HCC_HBV_LC'] = ','.join(np.array(y_train_2_HCC_HBV_LC).astype('str').tolist())
    result.loc[num, 'predict_proba_train_HCC_HBV_LC'] = ','.join(np.array(pred_proba_train_2_HCC_HBV_LC[:, 1]).astype('str').tolist())
    result.loc[num,'roc_auc_test_HCC_HBV_LC']=roc_auc_test_2_HCC_HBV_LC
    result.loc[num,'bs_test_HCC_HBV_LC']=bs_test_2_HCC_HBV_LC
    result.loc[num, 'hl_test_HCC_HBV_LC'] = hl_test_2_HCC_HBV_LC
    result.loc[num,'Sensitivity_best_test_HCC_HBV_LC']=Sensitivity_best_test_HCC_HBV_LC
    result.loc[num, 'Specificity_best_test_HCC_HBV_LC'] = Specificity_best_test_HCC_HBV_LC
    result.loc[num, 'Sensitivity_adjust_test_HCC_HBV_LC'] = Sensitivity_adjust_test_HCC_HBV_LC
    result.loc[num, 'Specificity_adjust_test_HCC_HBV_LC'] = Specificity_adjust_test_HCC_HBV_LC
    result.loc[num, 'y_test_HCC_HBV_LC'] = ','.join(np.array(y_test_2_HCC_HBV_LC).astype('str').tolist())
    result.loc[num, 'predict_proba_test_HCC_HBV_LC'] = ','.join(np.array(pred_proba_test_2_HCC_HBV_LC[:, 1]).astype('str').tolist())

    result.loc[num, 'roc_auc_train_afp'] = roc_auc_train_afp_2
    result.loc[num, 'bs_train_afp'] = bs_train_afp_2
    result.loc[num, 'hl_train_afp'] = hl_train_afp_2
    result.loc[num,'Sensitivity_best_train_afp']=Sensitivity_best_train_afp
    result.loc[num, 'Specificity_best_train_afp'] = Specificity_best_train_afp
    result.loc[num, 'Sensitivity_adjust_train_afp'] = Sensitivity_adjust_train_afp
    result.loc[num, 'Specificity_adjust_train_afp'] = Specificity_adjust_train_afp
    result.loc[num, 'y_train_afp'] = ','.join(np.array(y_train_afp_2).astype('str').tolist())
    result.loc[num, 'predict_proba_train_afp'] = ','.join(np.array(pred_proba_train_afp_2[:, 1]).astype('str').tolist())
    result.loc[num, 'roc_auc_test_afp'] = roc_auc_test_afp_2
    result.loc[num, 'bs_test_afp'] =bs_test_afp_2
    result.loc[num, 'hl_test_afp'] = hl_test_afp_2
    result.loc[num,'Sensitivity_best_test_afp']=Sensitivity_best_test_afp
    result.loc[num, 'Specificity_best_test_afp'] = Specificity_best_test_afp
    result.loc[num, 'Sensitivity_adjust_test_afp'] = Sensitivity_adjust_test_afp
    result.loc[num, 'Specificity_adjust_test_afp'] = Specificity_adjust_test_afp
    result.loc[num, 'y_test_afp'] = ','.join(np.array(y_test_afp_2).astype('str').tolist())
    result.loc[num, 'predict_proba_test_afp'] = ','.join(np.array(pred_proba_test_afp_2[:, 1]).astype('str').tolist())

    result.loc[num, 'roc_auc_train_afp_HCC_HD'] = roc_auc_train_afp_2_HCC_HD
    result.loc[num, 'bs_train_afp_HCC_HD'] = bs_train_afp_2_HCC_HD
    result.loc[num, 'hl_train_afp_HCC_HD'] = hl_train_afp_2_HCC_HD
    result.loc[num,'Sensitivity_best_train_afp_HCC_HD']=Sensitivity_best_train_afp_HCC_HD
    result.loc[num, 'Specificity_best_train_afp_HCC_HD'] = Specificity_best_train_afp_HCC_HD
    result.loc[num, 'Sensitivity_adjust_train_afp_HCC_HD'] = Sensitivity_adjust_train_afp_HCC_HD
    result.loc[num, 'Specificity_adjust_train_afp_HCC_HD'] = Specificity_adjust_train_afp_HCC_HD
    result.loc[num, 'y_train_afp_HCC_HD'] = ','.join(np.array(y_train_afp_2_HCC_HD).astype('str').tolist())
    result.loc[num, 'predict_proba_train_afp_HCC_HD'] = ','.join(np.array(pred_proba_train_afp_2_HCC_HD[:, 1]).astype('str').tolist())
    result.loc[num, 'roc_auc_test_afp_HCC_HD'] = roc_auc_test_afp_2_HCC_HD
    result.loc[num, 'bs_test_afp_HCC_HD'] = bs_test_afp_2_HCC_HD
    result.loc[num, 'hl_test_afp_HCC_HD'] = hl_test_afp_2_HCC_HD
    result.loc[num,'Sensitivity_best_test_afp_HCC_HD']=Sensitivity_best_test_afp_HCC_HD
    result.loc[num, 'Specificity_best_test_afp_HCC_HD'] = Specificity_best_test_afp_HCC_HD
    result.loc[num, 'Sensitivity_adjust_test_afp_HCC_HD'] = Sensitivity_adjust_test_afp_HCC_HD
    result.loc[num, 'Specificity_adjust_test_afp_HCC_HD'] = Specificity_adjust_test_afp_HCC_HD
    result.loc[num, 'y_test_afp_HCC_HD'] = ','.join(np.array(y_test_afp_2_HCC_HD).astype('str').tolist())
    result.loc[num, 'predict_proba_test_afp_HCC_HD'] = ','.join(np.array(pred_proba_test_afp_2_HCC_HD[:, 1]).astype('str').tolist())

    result.loc[num, 'roc_auc_train_afp_HCC_HBV'] = roc_auc_train_afp_2_HCC_HBV
    result.loc[num, 'bs_train_afp_HCC_HBV'] = bs_train_afp_2_HCC_HBV
    result.loc[num, 'hl_train_afp_HCC_HBV'] = hl_train_afp_2_HCC_HBV
    result.loc[num,'Sensitivity_best_train_afp_HCC_HBV']=Sensitivity_best_train_afp_HCC_HBV
    result.loc[num, 'Specificity_best_train_afp_HCC_HBV'] = Specificity_best_train_afp_HCC_HBV
    result.loc[num, 'Sensitivity_adjust_train_afp_HCC_HBV'] = Sensitivity_adjust_train_afp_HCC_HBV
    result.loc[num, 'Specificity_adjust_train_afp_HCC_HBV'] = Specificity_adjust_train_afp_HCC_HBV
    result.loc[num, 'y_train_afp_HCC_HBV'] = ','.join(np.array(y_train_afp_2_HCC_HBV).astype('str').tolist())
    result.loc[num, 'predict_proba_train_afp_HCC_HBV'] = ','.join(np.array(pred_proba_train_afp_2_HCC_HBV[:, 1]).astype('str').tolist())
    result.loc[num, 'roc_auc_test_afp_HCC_HBV'] = roc_auc_test_afp_2_HCC_HBV
    result.loc[num, 'bs_test_afp_HCC_HBV'] = bs_test_afp_2_HCC_HBV
    result.loc[num, 'hl_test_afp_HCC_HBV'] = hl_test_afp_2_HCC_HBV
    result.loc[num,'Sensitivity_best_test_afp_HCC_HBV']=Sensitivity_best_test_afp_HCC_HBV
    result.loc[num, 'Specificity_best_test_afp_HCC_HBV'] = Specificity_best_test_afp_HCC_HBV
    result.loc[num, 'Sensitivity_adjust_test_afp_HCC_HBV'] = Sensitivity_adjust_test_afp_HCC_HBV
    result.loc[num, 'Specificity_adjust_test_afp_HCC_HBV'] = Specificity_adjust_test_afp_HCC_HBV
    result.loc[num, 'y_test_afp_HCC_HBV'] = ','.join(np.array(y_test_afp_2_HCC_HBV).astype('str').tolist())
    result.loc[num, 'predict_proba_test_afp_HCC_HBV'] = ','.join(np.array(pred_proba_test_afp_2_HCC_HBV[:, 1]).astype('str').tolist())

    result.loc[num, 'roc_auc_train_afp_HCC_LC'] = roc_auc_train_afp_2_HCC_LC
    result.loc[num, 'bs_train_afp_HCC_LC'] = bs_train_afp_2_HCC_LC
    result.loc[num, 'hl_train_afp_HCC_LC'] = hl_train_afp_2_HCC_LC
    result.loc[num,'Sensitivity_best_train_afp_HCC_LC']=Sensitivity_best_train_afp_HCC_LC
    result.loc[num, 'Specificity_best_train_afp_HCC_LC'] = Specificity_best_train_afp_HCC_LC
    result.loc[num, 'Sensitivity_adjust_train_afp_HCC_LC'] = Sensitivity_adjust_train_afp_HCC_LC
    result.loc[num, 'Specificity_adjust_train_afp_HCC_LC'] = Specificity_adjust_train_afp_HCC_LC
    result.loc[num, 'y_train_afp_HCC_LC'] = ','.join(np.array(y_train_afp_2_HCC_LC).astype('str').tolist())
    result.loc[num, 'predict_proba_train_afp_HCC_LC'] = ','.join(np.array(pred_proba_train_afp_2_HCC_LC[:, 1]).astype('str').tolist())
    result.loc[num, 'roc_auc_test_afp_HCC_LC'] = roc_auc_test_afp_2_HCC_LC
    result.loc[num, 'bs_test_afp_HCC_LC'] = bs_test_afp_2_HCC_LC
    result.loc[num, 'hl_test_afp_HCC_LC'] = hl_test_afp_2_HCC_LC
    result.loc[num,'Sensitivity_best_test_afp_HCC_LC']=Sensitivity_best_test_afp_HCC_LC
    result.loc[num, 'Specificity_best_test_afp_HCC_LC'] = Specificity_best_test_afp_HCC_LC
    result.loc[num, 'Sensitivity_adjust_test_afp_HCC_LC'] = Sensitivity_adjust_test_afp_HCC_LC
    result.loc[num, 'Specificity_adjust_test_afp_HCC_LC'] = Specificity_adjust_test_afp_HCC_LC
    result.loc[num, 'y_test_afp_HCC_LC'] = ','.join(np.array(y_test_afp_2_HCC_LC).astype('str').tolist())
    result.loc[num, 'predict_proba_test_afp_HCC_LC'] = ','.join(np.array(pred_proba_test_afp_2_HCC_LC[:, 1]).astype('str').tolist())

    result.loc[num, 'roc_auc_train_afp_HCC_HBV_LC'] = roc_auc_train_afp_2_HCC_HBV_LC
    result.loc[num, 'bs_train_afp_HCC_HBV_LC'] = bs_train_afp_2_HCC_HBV_LC
    result.loc[num, 'hl_train_afp_HCC_HBV_LC'] = hl_train_afp_2_HCC_HBV_LC
    result.loc[num,'Sensitivity_best_train_afp_HCC_HBV_LC']=Sensitivity_best_train_afp_HCC_HBV_LC
    result.loc[num, 'Specificity_best_train_afp_HCC_HBV_LC'] = Specificity_best_train_afp_HCC_HBV_LC
    result.loc[num, 'Sensitivity_adjust_train_afp_HCC_HBV_LC'] = Sensitivity_adjust_train_afp_HCC_HBV_LC
    result.loc[num, 'Specificity_adjust_train_afp_HCC_HBV_LC'] = Specificity_adjust_train_afp_HCC_HBV_LC
    result.loc[num, 'y_train_afp_HCC_HBV_LC'] = ','.join(np.array(y_train_afp_2_HCC_HBV_LC).astype('str').tolist())
    result.loc[num, 'predict_proba_train_afp_HCC_HBV_LC'] = ','.join(np.array(pred_proba_train_afp_2_HCC_HBV_LC[:, 1]).astype('str').tolist())
    result.loc[num, 'roc_auc_test_afp_HCC_HBV_LC'] = roc_auc_test_afp_2_HCC_HBV_LC
    result.loc[num, 'bs_test_afp_HCC_HBV_LC'] = bs_test_afp_2_HCC_HBV_LC
    result.loc[num, 'hl_test_afp_HCC_HBV_LC'] = hl_test_afp_2_HCC_HBV_LC
    result.loc[num,'Sensitivity_best_test_afp_HCC_HBV_LC']=Sensitivity_best_test_afp_HCC_HBV_LC
    result.loc[num, 'Specificity_best_test_afp_HCC_HBV_LC'] = Specificity_best_test_afp_HCC_HBV_LC
    result.loc[num, 'Sensitivity_adjust_test_afp_HCC_HBV_LC'] = Sensitivity_adjust_test_afp_HCC_HBV_LC
    result.loc[num, 'Specificity_adjust_test_afp_HCC_HBV_LC'] = Specificity_adjust_test_afp_HCC_HBV_LC
    result.loc[num, 'y_test_afp_HCC_HBV_LC'] = ','.join(np.array(y_test_afp_2_HCC_HBV_LC).astype('str').tolist())
    result.loc[num, 'predict_proba_test_afp_HCC_HBV_LC'] = ','.join(np.array(pred_proba_test_afp_2_HCC_HBV_LC[:, 1]).astype('str').tolist())

    result_0.loc[num,'roc_auc_train_0']=roc_auc_train_2_early
    result_0.loc[num,'bs_train_0']=bs_train_2_early
    result_0.loc[num,'hl_train_0']=hl_train_2_early
    result_0.loc[num,'Sensitivity_best_train_0']=Sensitivity_best_train_early
    result_0.loc[num, 'Specificity_best_train_0'] = Specificity_best_train_early
    result_0.loc[num, 'Sensitivity_adjust_train_0'] = Sensitivity_adjust_train_early
    result_0.loc[num, 'Specificity_adjust_train_0'] = Specificity_adjust_train_early
    result_0.loc[num, 'y_train_0'] = ','.join(np.array(y_train_2_early).astype('str').tolist())
    result_0.loc[num, 'predict_proba_train_0'] = ','.join(np.array(pred_proba_train_2_early[:, 1]).astype('str').tolist())
    result_0.loc[num,'roc_auc_test_0']=roc_auc_test_2_early
    result_0.loc[num,'bs_test_0']=bs_test_2_early
    result_0.loc[num,'hl_test_0']=hl_test_2_early
    result_0.loc[num,'Sensitivity_best_test_0']=Sensitivity_best_test_early
    result_0.loc[num, 'Specificity_best_test_0'] = Specificity_best_test_early
    result_0.loc[num, 'Sensitivity_adjust_test_0'] = Sensitivity_adjust_test_early
    result_0.loc[num, 'Specificity_adjust_test_0'] = Specificity_adjust_test_early
    result_0.loc[num, 'y_test_0'] = ','.join(np.array(y_test_2_early).astype('str').tolist())
    result_0.loc[num, 'predict_proba_test_0'] = ','.join(np.array(pred_proba_test_2_early[:, 1]).astype('str').tolist())

    result_0.loc[num,'roc_auc_train_HCC_HD_0']=roc_auc_train_2_HCC_HD_0
    result_0.loc[num,'bs_train_HCC_HD_0']=bs_train_2_HCC_HD_0
    result_0.loc[num,'hl_train_HCC_HD_0']=hl_train_2_HCC_HD_0
    result_0.loc[num,'Sensitivity_best_train_HCC_HD_0']=Sensitivity_best_train_HCC_HD_0
    result_0.loc[num, 'Specificity_best_train_HCC_HD_0'] = Specificity_best_train_HCC_HD_0
    result_0.loc[num, 'Sensitivity_adjust_train_HCC_HD_0'] = Sensitivity_adjust_train_HCC_HD_0
    result_0.loc[num, 'Specificity_adjust_train_HCC_HD_0'] = Specificity_adjust_train_HCC_HD_0
    result_0.loc[num, 'y_train_HCC_HD_0'] = ','.join(np.array(y_train_2_HCC_HD_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_train_HCC_HD_0'] = ','.join(np.array(pred_proba_train_2_HCC_HD_0[:, 1]).astype('str').tolist())
    result_0.loc[num,'roc_auc_test_HCC_HD_0']=roc_auc_test_2_HCC_HD_0
    result_0.loc[num,'bs_test_HCC_HD_0']=bs_test_2_HCC_HD_0
    result_0.loc[num,'hl_test_HCC_HD_0']=hl_test_2_HCC_HD_0
    result_0.loc[num,'Sensitivity_best_test_HCC_HD_0']=Sensitivity_best_test_HCC_HD_0
    result_0.loc[num, 'Specificity_best_test_HCC_HD_0'] = Specificity_best_test_HCC_HD_0
    result_0.loc[num, 'Sensitivity_adjust_test_HCC_HD_0'] = Sensitivity_adjust_test_HCC_HD_0
    result_0.loc[num, 'Specificity_adjust_test_HCC_HD_0'] = Specificity_adjust_test_HCC_HD_0
    result_0.loc[num, 'y_test_HCC_HD_0'] = ','.join(np.array(y_test_2_HCC_HD_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_test_HCC_HD_0'] = ','.join(np.array(pred_proba_test_2_HCC_HD_0[:, 1]).astype('str').tolist())

    result_0.loc[num,'roc_auc_train_HCC_HBV_0']=roc_auc_train_2_HCC_HBV_0
    result_0.loc[num,'bs_train_HCC_HBV_0']=bs_train_2_HCC_HBV_0
    result_0.loc[num,'hl_train_HCC_HBV_0']=hl_train_2_HCC_HBV_0
    result_0.loc[num,'Sensitivity_best_train_HCC_HBV_0']=Sensitivity_best_train_HCC_HBV_0
    result_0.loc[num, 'Specificity_best_train_HCC_HBV_0'] = Specificity_best_train_HCC_HBV_0
    result_0.loc[num, 'Sensitivity_adjust_train_HCC_HBV_0'] = Sensitivity_adjust_train_HCC_HBV_0
    result_0.loc[num, 'Specificity_adjust_train_HCC_HBV_0'] = Specificity_adjust_train_HCC_HBV_0
    result_0.loc[num, 'y_train_HCC_HBV_0'] = ','.join(np.array(y_train_2_HCC_HBV_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_train_HCC_HBV_0'] = ','.join(np.array(pred_proba_train_2_HCC_HBV_0[:, 1]).astype('str').tolist())
    result_0.loc[num,'roc_auc_test_HCC_HBV_0']=roc_auc_test_2_HCC_HBV_0
    result_0.loc[num,'bs_test_HCC_HBV_0']=bs_test_2_HCC_HBV_0
    result_0.loc[num,'hl_test_HCC_HBV_0']=hl_test_2_HCC_HBV_0
    result_0.loc[num,'Sensitivity_best_test_HCC_HBV_0']=Sensitivity_best_test_HCC_HBV_0
    result_0.loc[num, 'Specificity_best_test_HCC_HBV_0'] = Specificity_best_test_HCC_HBV_0
    result_0.loc[num, 'Sensitivity_adjust_test_HCC_HBV_0'] = Sensitivity_adjust_test_HCC_HBV_0
    result_0.loc[num, 'Specificity_adjust_test_HCC_HBV_0'] = Specificity_adjust_test_HCC_HBV_0
    result_0.loc[num, 'y_test_HCC_HBV_0'] = ','.join(np.array(y_test_2_HCC_HBV_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_test_HCC_HBV_0'] = ','.join(np.array(pred_proba_test_2_HCC_HBV_0[:, 1]).astype('str').tolist())

    result_0.loc[num,'roc_auc_train_HCC_LC_0']=roc_auc_train_2_HCC_LC_0
    result_0.loc[num,'bs_train_HCC_LC_0']=bs_train_2_HCC_LC_0
    result_0.loc[num,'hl_train_HCC_LC_0']=hl_train_2_HCC_LC_0
    result_0.loc[num,'Sensitivity_best_train_HCC_LC_0']=Sensitivity_best_train_HCC_LC_0
    result_0.loc[num, 'Specificity_best_train_HCC_LC_0'] = Specificity_best_train_HCC_LC_0
    result_0.loc[num, 'Sensitivity_adjust_train_HCC_LC_0'] = Sensitivity_adjust_train_HCC_LC_0
    result_0.loc[num, 'Specificity_adjust_train_HCC_LC_0'] = Specificity_adjust_train_HCC_LC_0
    result_0.loc[num, 'y_train_HCC_LC_0'] = ','.join(np.array(y_train_2_HCC_LC_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_train_HCC_LC_0'] = ','.join(np.array(pred_proba_train_2_HCC_LC_0[:, 1]).astype('str').tolist())
    result_0.loc[num,'roc_auc_test_HCC_LC_0']=roc_auc_test_2_HCC_LC_0
    result_0.loc[num,'bs_test_HCC_LC_0']=bs_test_2_HCC_LC_0
    result_0.loc[num,'hl_test_HCC_LC_0']=hl_test_2_HCC_LC_0
    result_0.loc[num,'Sensitivity_best_test_HCC_LC_0']=Sensitivity_best_test_HCC_LC_0
    result_0.loc[num, 'Specificity_best_test_HCC_LC_0'] = Specificity_best_test_HCC_LC_0
    result_0.loc[num, 'Sensitivity_adjust_test_HCC_LC_0'] = Sensitivity_adjust_test_HCC_LC_0
    result_0.loc[num, 'Specificity_adjust_test_HCC_LC_0'] = Specificity_adjust_test_HCC_LC_0
    result_0.loc[num, 'y_test_HCC_LC_0'] = ','.join(np.array(y_test_2_HCC_LC_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_test_HCC_LC_0'] = ','.join(np.array(pred_proba_test_2_HCC_LC_0[:, 1]).astype('str').tolist())

    result_0.loc[num,'roc_auc_train_HCC_HBV_LC_0']=roc_auc_train_2_HCC_HBV_LC_0
    result_0.loc[num,'bs_train_HCC_HBV_LC_0']=bs_train_2_HCC_HBV_LC_0
    result_0.loc[num,'hl_train_HCC_HBV_LC_0']=hl_train_2_HCC_HBV_LC_0
    result_0.loc[num,'Sensitivity_best_train_HCC_HBV_LC_0']=Sensitivity_best_train_HCC_HBV_LC_0
    result_0.loc[num, 'Specificity_best_train_HCC_HBV_LC_0'] = Specificity_best_train_HCC_HBV_LC_0
    result_0.loc[num, 'Sensitivity_adjust_train_HCC_HBV_LC_0'] = Sensitivity_adjust_train_HCC_HBV_LC_0
    result_0.loc[num, 'Specificity_adjust_train_HCC_HBV_LC_0'] = Specificity_adjust_train_HCC_HBV_LC_0
    result_0.loc[num, 'y_train_HCC_HBV_LC_0'] = ','.join(np.array(y_train_2_HCC_HBV_LC_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_train_HCC_HBV_LC_0'] = ','.join(np.array(pred_proba_train_2_HCC_HBV_LC_0[:, 1]).astype('str').tolist())
    result_0.loc[num,'roc_auc_test_HCC_HBV_LC_0']=roc_auc_test_2_HCC_HBV_LC_0
    result_0.loc[num,'bs_test_HCC_HBV_LC_0']=bs_test_2_HCC_HBV_LC_0
    result_0.loc[num,'hl_test_HCC_HBV_LC_0']=hl_test_2_HCC_HBV_LC_0
    result_0.loc[num,'Sensitivity_best_test_HCC_HBV_LC_0']=Sensitivity_best_test_HCC_HBV_LC_0
    result_0.loc[num, 'Specificity_best_test_HCC_HBV_LC_0'] = Specificity_best_test_HCC_HBV_LC_0
    result_0.loc[num, 'Sensitivity_adjust_test_HCC_HBV_LC_0'] = Sensitivity_adjust_test_HCC_HBV_LC_0
    result_0.loc[num, 'Specificity_adjust_test_HCC_HBV_LC_0'] = Specificity_adjust_test_HCC_HBV_LC_0
    result_0.loc[num, 'y_test_HCC_HBV_LC_0'] = ','.join(np.array(y_test_2_HCC_HBV_LC_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_test_HCC_HBV_LC_0'] = ','.join(np.array(pred_proba_test_2_HCC_HBV_LC_0[:, 1]).astype('str').tolist())

    result_0.loc[num, 'roc_auc_train_afp_0'] = roc_auc_train_afp_2_early
    result_0.loc[num,'Sensitivity_best_train_afp_0']=Sensitivity_best_train_afp_early
    result_0.loc[num, 'Specificity_best_train_afp_0'] = Specificity_best_train_afp_early
    result_0.loc[num, 'Sensitivity_adjust_train_afp_0'] = Sensitivity_adjust_train_afp_early
    result_0.loc[num, 'Specificity_adjust_train_afp_0'] = Specificity_adjust_train_afp_early
    result_0.loc[num, 'y_train_afp_0'] = ','.join(np.array(y_train_afp_2_early).astype('str').tolist())
    result_0.loc[num, 'predict_proba_train_afp_0'] = ','.join(np.array(pred_proba_train_afp_2_early[:, 1]).astype('str').tolist())
    result_0.loc[num, 'roc_auc_test_afp_0'] = roc_auc_test_afp_2_early
    result_0.loc[num,'Sensitivity_best_test_afp_0']=Sensitivity_best_test_afp_early
    result_0.loc[num, 'Specificity_best_test_afp_0'] = Specificity_best_test_afp_early
    result_0.loc[num, 'Sensitivity_adjust_test_afp_0'] = Sensitivity_adjust_test_afp_early
    result_0.loc[num, 'Specificity_adjust_test_afp_0'] = Specificity_adjust_test_afp_early
    result_0.loc[num, 'y_test_afp_0'] = ','.join(np.array(y_test_afp_2_early).astype('str').tolist())
    result_0.loc[num, 'predict_proba_test_afp_0'] = ','.join(np.array(pred_proba_test_afp_2_early[:, 1]).astype('str').tolist())

    result_0.loc[num, 'roc_auc_train_afp_HCC_HD_0'] = roc_auc_train_afp_2_HCC_HD_0
    result_0.loc[num,'Sensitivity_best_train_afp_HCC_HD_0']=Sensitivity_best_train_afp_HCC_HD_0
    result_0.loc[num, 'Specificity_best_train_afp_HCC_HD_0'] = Specificity_best_train_afp_HCC_HD_0
    result_0.loc[num, 'Sensitivity_adjust_train_afp_HCC_HD_0'] = Sensitivity_adjust_train_afp_HCC_HD_0
    result_0.loc[num, 'Specificity_adjust_train_afp_HCC_HD_0'] = Specificity_adjust_train_afp_HCC_HD_0
    result_0.loc[num, 'y_train_afp_HCC_HD_0'] = ','.join(np.array(y_train_afp_2_HCC_HD_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_train_afp_HCC_HD_0'] = ','.join(np.array(pred_proba_train_afp_2_HCC_HD_0[:, 1]).astype('str').tolist())
    result_0.loc[num, 'roc_auc_test_afp_HCC_HD_0'] = roc_auc_test_afp_2_HCC_HD_0
    result_0.loc[num,'Sensitivity_best_test_afp_HCC_HD_0']=Sensitivity_best_test_afp_HCC_HD_0
    result_0.loc[num, 'Specificity_best_test_afp_HCC_HD_0'] = Specificity_best_test_afp_HCC_HD_0
    result_0.loc[num, 'Sensitivity_adjust_test_afp_HCC_HD_0'] = Sensitivity_adjust_test_afp_HCC_HD_0
    result_0.loc[num, 'Specificity_adjust_test_afp_HCC_HD_0'] = Specificity_adjust_test_afp_HCC_HD_0
    result_0.loc[num, 'y_test_afp_HCC_HD_0'] = ','.join(np.array(y_test_afp_2_HCC_HD_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_test_afp_HCC_HD_0'] = ','.join(np.array(pred_proba_test_afp_2_HCC_HD_0[:, 1]).astype('str').tolist())

    result_0.loc[num, 'roc_auc_train_afp_HCC_HBV_0'] = roc_auc_train_afp_2_HCC_HBV_0
    result_0.loc[num,'Sensitivity_best_train_afp_HCC_HBV_0']=Sensitivity_best_train_afp_HCC_HBV_0
    result_0.loc[num, 'Specificity_best_train_afp_HCC_HBV_0'] = Specificity_best_train_afp_HCC_HBV_0
    result_0.loc[num, 'Sensitivity_adjust_train_afp_HCC_HBV_0'] = Sensitivity_adjust_train_afp_HCC_HBV_0
    result_0.loc[num, 'Specificity_adjust_train_afp_HCC_HBV_0'] = Specificity_adjust_train_afp_HCC_HBV_0
    result_0.loc[num, 'y_train_afp_HCC_HBV_0'] = ','.join(np.array(y_train_afp_2_HCC_HBV_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_train_afp_HCC_HBV_0'] = ','.join(np.array(pred_proba_train_afp_2_HCC_HBV_0[:, 1]).astype('str').tolist())
    result_0.loc[num, 'roc_auc_test_afp_HCC_HBV_0'] = roc_auc_test_afp_2_HCC_HBV_0
    result_0.loc[num,'Sensitivity_best_test_afp_HCC_HBV_0']=Sensitivity_best_test_afp_HCC_HBV_0
    result_0.loc[num, 'Specificity_best_test_afp_HCC_HBV_0'] = Specificity_best_test_afp_HCC_HBV_0
    result_0.loc[num, 'Sensitivity_adjust_test_afp_HCC_HBV_0'] = Sensitivity_adjust_test_afp_HCC_HBV_0
    result_0.loc[num, 'Specificity_adjust_test_afp_HCC_HBV_0'] = Specificity_adjust_test_afp_HCC_HBV_0
    result_0.loc[num, 'y_test_afp_HCC_HBV_0'] = ','.join(np.array(y_test_afp_2_HCC_HBV_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_test_afp_HCC_HBV_0'] = ','.join(np.array(pred_proba_test_afp_2_HCC_HBV_0[:, 1]).astype('str').tolist())

    result_0.loc[num, 'roc_auc_train_afp_HCC_LC_0'] = roc_auc_train_afp_2_HCC_LC_0
    result_0.loc[num,'Sensitivity_best_train_afp_HCC_LC_0']=Sensitivity_best_train_afp_HCC_LC_0
    result_0.loc[num, 'Specificity_best_train_afp_HCC_LC_0'] = Specificity_best_train_afp_HCC_LC_0
    result_0.loc[num, 'Sensitivity_adjust_train_afp_HCC_LC_0'] = Sensitivity_adjust_train_afp_HCC_LC_0
    result_0.loc[num, 'Specificity_adjust_train_afp_HCC_LC_0'] = Specificity_adjust_train_afp_HCC_LC_0
    result_0.loc[num, 'y_train_afp_HCC_LC_0'] = ','.join(np.array(y_train_afp_2_HCC_LC_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_train_afp_HCC_LC_0'] = ','.join(np.array(pred_proba_train_afp_2_HCC_LC_0[:, 1]).astype('str').tolist())
    result_0.loc[num, 'roc_auc_test_afp_HCC_LC_0'] = roc_auc_test_afp_2_HCC_LC_0
    result_0.loc[num,'Sensitivity_best_test_afp_HCC_LC_0']=Sensitivity_best_test_afp_HCC_LC_0
    result_0.loc[num, 'Specificity_best_test_afp_HCC_LC_0'] = Specificity_best_test_afp_HCC_LC_0
    result_0.loc[num, 'Sensitivity_adjust_test_afp_HCC_LC_0'] = Sensitivity_adjust_test_afp_HCC_LC_0
    result_0.loc[num, 'Specificity_adjust_test_afp_HCC_LC_0'] = Specificity_adjust_test_afp_HCC_LC_0
    result_0.loc[num, 'y_test_afp_HCC_LC_0'] = ','.join(np.array(y_test_afp_2_HCC_LC_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_test_afp_HCC_LC_0'] = ','.join(np.array(pred_proba_test_afp_2_HCC_LC_0[:, 1]).astype('str').tolist())

    result_0.loc[num, 'roc_auc_train_afp_HCC_HBV_LC_0'] = roc_auc_train_afp_2_HCC_HBV_LC_0
    result_0.loc[num,'Sensitivity_best_train_afp_HCC_HBV_LC_0']=Sensitivity_best_train_afp_HCC_HBV_LC_0
    result_0.loc[num, 'Specificity_best_train_afp_HCC_HBV_LC_0'] = Specificity_best_train_afp_HCC_HBV_LC_0
    result_0.loc[num, 'Sensitivity_adjust_train_afp_HCC_HBV_LC_0'] = Sensitivity_adjust_train_afp_HCC_HBV_LC_0
    result_0.loc[num, 'Specificity_adjust_train_afp_HCC_HBV_LC_0'] = Specificity_adjust_train_afp_HCC_HBV_LC_0
    result_0.loc[num, 'y_train_afp_HCC_HBV_LC_0'] = ','.join(np.array(y_train_afp_2_HCC_HBV_LC_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_train_afp_HCC_HBV_LC_0'] = ','.join(np.array(pred_proba_train_afp_2_HCC_HBV_LC_0[:, 1]).astype('str').tolist())
    result_0.loc[num, 'roc_auc_test_afp_HCC_HBV_LC_0'] = roc_auc_test_afp_2_HCC_HBV_LC_0
    result_0.loc[num,'Sensitivity_best_test_afp_HCC_HBV_LC_0']=Sensitivity_best_test_afp_HCC_HBV_LC_0
    result_0.loc[num, 'Specificity_best_test_afp_HCC_HBV_LC_0'] = Specificity_best_test_afp_HCC_HBV_LC_0
    result_0.loc[num, 'Sensitivity_adjust_test_afp_HCC_HBV_LC_0'] = Sensitivity_adjust_test_afp_HCC_HBV_LC_0
    result_0.loc[num, 'Specificity_adjust_test_afp_HCC_HBV_LC_0'] = Specificity_adjust_test_afp_HCC_HBV_LC_0
    result_0.loc[num, 'y_test_afp_HCC_HBV_LC_0'] = ','.join(np.array(y_test_afp_2_HCC_HBV_LC_0).astype('str').tolist())
    result_0.loc[num, 'predict_proba_test_afp_HCC_HBV_LC_0'] = ','.join(np.array(pred_proba_test_afp_2_HCC_HBV_LC_0[:, 1]).astype('str').tolist())

    result_0.loc[num, 'bs_train_afp_0'] = bs_train_afp_2_early
    result_0.loc[num, 'bs_test_afp_0'] = bs_test_afp_2_early
    result_0.loc[num, 'bs_train_afp_HCC_HD_0'] = bs_train_afp_2_HCC_HD_0
    result_0.loc[num, 'bs_test_afp_HCC_HD_0'] = bs_test_afp_2_HCC_HD_0
    result_0.loc[num, 'bs_train_afp_HCC_HBV_0'] = bs_train_afp_2_HCC_HBV_0
    result_0.loc[num, 'bs_test_afp_HCC_HBV_0'] = bs_test_afp_2_HCC_HBV_0
    result_0.loc[num, 'bs_train_afp_HCC_LC_0'] = bs_train_afp_2_HCC_LC_0
    result_0.loc[num, 'bs_test_afp_HCC_LC_0'] = bs_test_afp_2_HCC_LC_0
    result_0.loc[num, 'bs_train_afp_HCC_HBV_LC_0'] = bs_train_afp_2_HCC_HBV_LC_0
    result_0.loc[num, 'bs_test_afp_HCC_HBV_LC_0'] = bs_test_afp_2_HCC_HBV_LC_0
    result_0.loc[num, 'hl_train_afp_0'] = hl_train_afp_2_early
    result_0.loc[num, 'hl_test_afp_0'] = hl_test_afp_2_early
    result_0.loc[num, 'hl_train_afp_HCC_HD_0'] = hl_train_afp_2_HCC_HD_0
    result_0.loc[num, 'hl_test_afp_HCC_HD_0'] = hl_test_afp_2_HCC_HD_0
    result_0.loc[num, 'hl_train_afp_HCC_HBV_0'] = hl_train_afp_2_HCC_HBV_0
    result_0.loc[num, 'hl_test_afp_HCC_HBV_0'] = hl_test_afp_2_HCC_HBV_0
    result_0.loc[num, 'hl_train_afp_HCC_LC_0'] = hl_train_afp_2_HCC_LC_0
    result_0.loc[num, 'hl_test_afp_HCC_LC_0'] = hl_test_afp_2_HCC_LC_0
    result_0.loc[num, 'hl_train_afp_HCC_HBV_LC_0'] = hl_train_afp_2_HCC_HBV_LC_0
    result_0.loc[num, 'hl_test_afp_HCC_HBV_LC_0'] = hl_test_afp_2_HCC_HBV_LC_0
    
    result.to_csv(save_path+'result_'+str(save_name)+'.txt',sep='\t',index=False)
    result_0.to_csv(save_path+'result_0_'+str(save_name)+'.txt',sep='\t',index=False)


def main(args):
    get_result(args.save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--save_name', type=int, required=True,
                        help='save_name', dest='save_name')
    args = parser.parse_args()
    main(args)