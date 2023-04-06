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
from pycaleva import CalibrationEvaluator



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



seed=133
model_type='RF'
sample_weight = 'balanced'
cancer = ['HCC', 'HD', 'HBV', 'LC']
gene_list=['CYTOR','WDR74','GGA2-S','miR.21.5p','RN7SL1.S.fragment','SNORD89']
data_train=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/tmp/HCC/code_230323/GitHub/data_discovery.txt',sep='\t')
data_test=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/tmp/HCC/code_230323/GitHub/data_validation.txt',sep='\t')
data_all=pd.concat([data_train,data_test])
save_path='/apps/home/lulab_liuxiaofan/qhsky/project/tmp/HCC/result_230323/onesplit/'

sample_HCC=list(data_all.loc[(data_all['Group'] == 'HCC'), 'ID_new'])
sample_HCC_0 = list(data_all.loc[data_all.stage.isin(['A4', 'A1', 'A3', '0', 'A', 'A2']) & (data_all['Group'] == 'HCC'), 'ID_new'])
sample_control_all=list(data_all.loc[(data_all['Group'] != 'HCC'), 'ID_new'])
sample_control_HD=list(data_all.loc[(data_all['Group'] == 'HD'), 'ID_new'])
sample_control_HBV=list(data_all.loc[(data_all['Group'] == 'HBV'), 'ID_new'])
sample_control_LC=list(data_all.loc[(data_all['Group'] == 'LC'), 'ID_new'])

sample_all=sample_HCC+sample_control_all
sample_all_0=sample_HCC_0+sample_control_all
sample_all_late=list(set(sample_HCC)-set(sample_HCC_0))+sample_control_all
sample_HCC_HD=sample_HCC+sample_control_HD
sample_HCC_HD_0=sample_HCC_0+sample_control_HD
sample_HCC_HBV=sample_HCC+sample_control_HBV
sample_HCC_HBV_0=sample_HCC_0+sample_control_HBV
sample_HCC_LC=sample_HCC+sample_control_LC
sample_HCC_LC_0=sample_HCC_0+sample_control_LC
sample_HCC_HBV_LC=sample_HCC+sample_control_LC+sample_control_HBV
sample_HCC_HBV_LC_0=sample_HCC_0+sample_control_LC+sample_control_HBV


# 预测
data_train_afp = data_train.loc[pd.notnull(data_train['AFP final']), :]
data_test_afp = data_test.loc[pd.notnull(data_test['AFP final']), :]
data_train['AFP final']=data_train['AFP final'].fillna(-999)
data_test['AFP final']=data_test['AFP final'].fillna(-999)
data_train, data_test = imputation(data_train, data_test, gene_list, type='all', low=-15)
x_train_2 = np.array(data_train[gene_list + ['AFP final']])
x_test_2 = np.array(data_test[gene_list + ['AFP final']])
y_train_2 = np.array(data_train['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
y_test_2 = np.array(data_test['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

if sample_weight == 'balanced':
    sample_weight_ = compute_sample_weight(class_weight='balanced', y=y_train_2)
clf_2 = clf_select(model_type, seed)
clf_2.fit(x_train_2, y_train_2, sample_weight_)

#全部
pred_proba_train_2 = clf_2.predict_proba(x_train_2)
fpr_train, tpr_train, thresholds = roc_curve(y_train_2, pred_proba_train_2[:, 1])
Sensitivity_best_train, Specificity_best_train, Sensitivity_adjust_train, Specificity_adjust_train, Sensitivity_train, Specificity_train=find_metrics_best_for_shuffle(fpr_train, tpr_train,cut_spe=0.95)
roc_auc_train_2 = auc(fpr_train, tpr_train)
bs_train_2 = brier_score_loss(y_train_2, pred_proba_train_2[:, 1])
ce = CalibrationEvaluator(y_train_2, pred_proba_train_2[:, 1], outsample=True, n_groups=2)
hl_train_2 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_2, pred_proba_train_2[:, 1],np.array(data_train['ID_new'])]).T).to_csv(save_path+'result_train.txt',sep='\t',index=False,header=False)
pred_proba_test_2 = clf_2.predict_proba(x_test_2)
fpr_test, tpr_test, thresholds = roc_curve(y_test_2, pred_proba_test_2[:, 1])
Sensitivity_best_test, Specificity_best_test, Sensitivity_adjust_test, Specificity_adjust_test, Sensitivity_test, Specificity_test=find_metrics_best_for_shuffle(fpr_test, tpr_test,cut_spe=0.95)
roc_auc_test_2 = auc(fpr_test, tpr_test)
bs_test_2 = brier_score_loss(y_test_2, pred_proba_test_2[:, 1])
ce = CalibrationEvaluator(y_test_2, pred_proba_test_2[:, 1], outsample=True, n_groups=2)
hl_test_2 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_2, pred_proba_test_2[:, 1],np.array(data_test['ID_new'])]).T).to_csv(save_path+'result_test.txt',sep='\t',index=False,header=False)

#全部早期
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
bs_train_2_early = brier_score_loss(y_train_2_early, pred_proba_train_2_early[:, 1])
ce = CalibrationEvaluator(y_train_2_early, pred_proba_train_2_early[:, 1], outsample=True, n_groups=2)
hl_train_2_early = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_2_early, pred_proba_train_2_early[:, 1],np.array(data_train_early['ID_new'])]).T).to_csv(save_path+'result_train_0.txt',sep='\t',index=False,header=False)
pred_proba_test_2_early = clf_2.predict_proba(x_test_2_early)
fpr_test_early, tpr_test_early, thresholds = roc_curve(y_test_2_early, pred_proba_test_2_early[:, 1])
Sensitivity_best_test_early, Specificity_best_test_early, Sensitivity_adjust_test_early, Specificity_adjust_test_early, Sensitivity_test_early, Specificity_test_early=find_metrics_best_for_shuffle(fpr_test_early, tpr_test_early,cut_spe=0.95)
roc_auc_test_2_early = auc(fpr_test_early, tpr_test_early)
bs_test_2_early = brier_score_loss(y_test_2_early, pred_proba_test_2_early[:, 1])
ce = CalibrationEvaluator(y_test_2_early, pred_proba_test_2_early[:, 1], outsample=True, n_groups=2)
hl_test_2_early = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_2_early, pred_proba_test_2_early[:, 1],np.array(data_test_early['ID_new'])]).T).to_csv(save_path+'result_test_0.txt',sep='\t',index=False,header=False)

#全部晚期
data_train_late=data_train.loc[data_train.ID_new.isin(sample_all_late),:]
data_test_late = data_test.loc[data_test.ID_new.isin(sample_all_late), :]
x_train_2_late = np.array(data_train_late[gene_list + ['AFP final']])
x_test_2_late = np.array(data_test_late[gene_list + ['AFP final']])
y_train_2_late = np.array(data_train_late['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
y_test_2_late = np.array(data_test_late['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

pred_proba_train_2_late = clf_2.predict_proba(x_train_2_late)
fpr_train_late, tpr_train_late, thresholds = roc_curve(y_train_2_late, pred_proba_train_2_late[:, 1])
Sensitivity_best_train_late, Specificity_best_train_late, Sensitivity_adjust_train_late, Specificity_adjust_train_late, Sensitivity_train_late, Specificity_train_late=find_metrics_best_for_shuffle(fpr_train_late, tpr_train_late,cut_spe=0.95)
roc_auc_train_2_late = auc(fpr_train_late, tpr_train_late)
bs_train_2_late = brier_score_loss(y_train_2_late, pred_proba_train_2_late[:, 1])
ce = CalibrationEvaluator(y_train_2_late, pred_proba_train_2_late[:, 1], outsample=True, n_groups=2)
hl_train_2_late = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_2_late, pred_proba_train_2_late[:, 1],np.array(data_train_late['ID_new'])]).T).to_csv(save_path+'result_train_late.txt',sep='\t',index=False,header=False)
pred_proba_test_2_late = clf_2.predict_proba(x_test_2_late)
fpr_test_late, tpr_test_late, thresholds = roc_curve(y_test_2_late, pred_proba_test_2_late[:, 1])
Sensitivity_best_test_late, Specificity_best_test_late, Sensitivity_adjust_test_late, Specificity_adjust_test_late, Sensitivity_test_late, Specificity_test_late=find_metrics_best_for_shuffle(fpr_test_late, tpr_test_late,cut_spe=0.95)
roc_auc_test_2_late = auc(fpr_test_late, tpr_test_late)
bs_test_2_late = brier_score_loss(y_test_2_late, pred_proba_test_2_late[:, 1])
ce = CalibrationEvaluator(y_test_2_late, pred_proba_test_2_late[:, 1], outsample=True, n_groups=2)
hl_test_2_late = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_2_late, pred_proba_test_2_late[:, 1],np.array(data_test_late['ID_new'])]).T).to_csv(save_path+'result_test_late.txt',sep='\t',index=False,header=False)

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
bs_train_2_HCC_HD = brier_score_loss(y_train_2_HCC_HD, pred_proba_train_2_HCC_HD[:, 1])
ce = CalibrationEvaluator(y_train_2_HCC_HD, pred_proba_train_2_HCC_HD[:, 1], outsample=True, n_groups=2)
hl_train_2_HCC_HD = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_2_HCC_HD, pred_proba_train_2_HCC_HD[:, 1],np.array(data_train_HCC_HD['ID_new'])]).T).to_csv(save_path+'result_train_HCC_HD.txt',sep='\t',index=False,header=False)
pred_proba_test_2_HCC_HD= clf_2.predict_proba(x_test_2_HCC_HD)
fpr_test_HCC_HD, tpr_test_HCC_HD, thresholds = roc_curve(y_test_2_HCC_HD, pred_proba_test_2_HCC_HD[:, 1])
Sensitivity_best_test_HCC_HD, Specificity_best_test_HCC_HD, Sensitivity_adjust_test_HCC_HD, Specificity_adjust_test_HCC_HD, Sensitivity_test_HCC_HD, Specificity_test_HCC_HD=find_metrics_best_for_shuffle(fpr_test_HCC_HD, tpr_test_HCC_HD,cut_spe=0.95)
roc_auc_test_2_HCC_HD = auc(fpr_test_HCC_HD, tpr_test_HCC_HD)
bs_test_2_HCC_HD = brier_score_loss(y_test_2_HCC_HD, pred_proba_test_2_HCC_HD[:, 1])
ce = CalibrationEvaluator(y_test_2_HCC_HD, pred_proba_test_2_HCC_HD[:, 1], outsample=True, n_groups=2)
hl_test_2_HCC_HD = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_2_HCC_HD, pred_proba_test_2_HCC_HD[:, 1],np.array(data_test_HCC_HD['ID_new'])]).T).to_csv(save_path+'result_test_HCC_HD.txt',sep='\t',index=False,header=False)

#HCC VS HD 早期
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
bs_train_2_HCC_HD_0 = brier_score_loss(y_train_2_HCC_HD_0, pred_proba_train_2_HCC_HD_0[:, 1])
ce = CalibrationEvaluator(y_train_2_HCC_HD_0, pred_proba_train_2_HCC_HD_0[:, 1], outsample=True, n_groups=2)
hl_train_2_HCC_HD_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_2_HCC_HD_0, pred_proba_train_2_HCC_HD_0[:, 1],np.array(data_train_HCC_HD_0['ID_new'])]).T).to_csv(save_path+'result_train_HCC_HD_0.txt',sep='\t',index=False,header=False)

pred_proba_test_2_HCC_HD_0= clf_2.predict_proba(x_test_2_HCC_HD_0)
fpr_test_HCC_HD_0, tpr_test_HCC_HD_0, thresholds = roc_curve(y_test_2_HCC_HD_0, pred_proba_test_2_HCC_HD_0[:, 1])
Sensitivity_best_test_HCC_HD_0, Specificity_best_test_HCC_HD_0, Sensitivity_adjust_test_HCC_HD_0, Specificity_adjust_test_HCC_HD_0, Sensitivity_test_HCC_HD_0, Specificity_test_HCC_HD_0=find_metrics_best_for_shuffle(fpr_test_HCC_HD_0, tpr_test_HCC_HD_0,cut_spe=0.95)
roc_auc_test_2_HCC_HD_0 = auc(fpr_test_HCC_HD_0, tpr_test_HCC_HD_0)
bs_test_2_HCC_HD_0 = brier_score_loss(y_test_2_HCC_HD_0, pred_proba_test_2_HCC_HD_0[:, 1])
ce = CalibrationEvaluator(y_test_2_HCC_HD_0, pred_proba_test_2_HCC_HD_0[:, 1], outsample=True, n_groups=2)
hl_test_2_HCC_HD_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_2_HCC_HD_0, pred_proba_test_2_HCC_HD_0[:, 1],np.array(data_test_HCC_HD_0['ID_new'])]).T).to_csv(save_path+'result_test_HCC_HD_0.txt',sep='\t',index=False,header=False)

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
bs_train_2_HCC_HBV = brier_score_loss(y_train_2_HCC_HBV, pred_proba_train_2_HCC_HBV[:, 1])
ce = CalibrationEvaluator(y_train_2_HCC_HBV, pred_proba_train_2_HCC_HBV[:, 1], outsample=True, n_groups=2)
hl_train_2_HCC_HBV = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_2_HCC_HBV, pred_proba_train_2_HCC_HBV[:, 1],np.array(data_train_HCC_HBV['ID_new'])]).T).to_csv(save_path+'result_train_HCC_HBV.txt',sep='\t',index=False,header=False)

pred_proba_test_2_HCC_HBV= clf_2.predict_proba(x_test_2_HCC_HBV)
fpr_test_HCC_HBV, tpr_test_HCC_HBV, thresholds = roc_curve(y_test_2_HCC_HBV, pred_proba_test_2_HCC_HBV[:, 1])
Sensitivity_best_test_HCC_HBV, Specificity_best_test_HCC_HBV, Sensitivity_adjust_test_HCC_HBV, Specificity_adjust_test_HCC_HBV, Sensitivity_test_HCC_HBV, Specificity_test_HCC_HBV=find_metrics_best_for_shuffle(fpr_test_HCC_HBV, tpr_test_HCC_HBV,cut_spe=0.95)
roc_auc_test_2_HCC_HBV = auc(fpr_test_HCC_HBV, tpr_test_HCC_HBV)
bs_test_2_HCC_HBV = brier_score_loss(y_test_2_HCC_HBV, pred_proba_test_2_HCC_HBV[:, 1])
ce = CalibrationEvaluator(y_test_2_HCC_HBV, pred_proba_test_2_HCC_HBV[:, 1], outsample=True, n_groups=2)
hl_test_2_HCC_HBV = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_2_HCC_HBV, pred_proba_test_2_HCC_HBV[:, 1],np.array(data_test_HCC_HBV['ID_new'])]).T).to_csv(save_path+'result_test_HCC_HBV.txt',sep='\t',index=False,header=False)

#HCC VS HBV 早期
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
bs_train_2_HCC_HBV_0 = brier_score_loss(y_train_2_HCC_HBV_0, pred_proba_train_2_HCC_HBV_0[:, 1])
ce = CalibrationEvaluator(y_train_2_HCC_HBV_0, pred_proba_train_2_HCC_HBV_0[:, 1], outsample=True, n_groups=2)
hl_train_2_HCC_HBV_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_2_HCC_HBV_0, pred_proba_train_2_HCC_HBV_0[:, 1],np.array(data_train_HCC_HBV_0['ID_new'])]).T).to_csv(save_path+'result_train_HCC_HBV_0.txt',sep='\t',index=False,header=False)

pred_proba_test_2_HCC_HBV_0= clf_2.predict_proba(x_test_2_HCC_HBV_0)
fpr_test_HCC_HBV_0, tpr_test_HCC_HBV_0, thresholds = roc_curve(y_test_2_HCC_HBV_0, pred_proba_test_2_HCC_HBV_0[:, 1])
Sensitivity_best_test_HCC_HBV_0, Specificity_best_test_HCC_HBV_0, Sensitivity_adjust_test_HCC_HBV_0, Specificity_adjust_test_HCC_HBV_0, Sensitivity_test_HCC_HBV_0, Specificity_test_HCC_HBV_0=find_metrics_best_for_shuffle(fpr_test_HCC_HBV_0, tpr_test_HCC_HBV_0,cut_spe=0.95)
roc_auc_test_2_HCC_HBV_0 = auc(fpr_test_HCC_HBV_0, tpr_test_HCC_HBV_0)
bs_test_2_HCC_HBV_0 = brier_score_loss(y_test_2_HCC_HBV_0, pred_proba_test_2_HCC_HBV_0[:, 1])
ce = CalibrationEvaluator(y_test_2_HCC_HBV_0, pred_proba_test_2_HCC_HBV_0[:, 1], outsample=True, n_groups=2)
hl_test_2_HCC_HBV_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_2_HCC_HBV_0, pred_proba_test_2_HCC_HBV_0[:, 1],np.array(data_test_HCC_HBV_0['ID_new'])]).T).to_csv(save_path+'result_test_HCC_HBV_0.txt',sep='\t',index=False,header=False)

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
bs_train_2_HCC_LC = brier_score_loss(y_train_2_HCC_LC, pred_proba_train_2_HCC_LC[:, 1])
ce = CalibrationEvaluator(y_train_2_HCC_LC, pred_proba_train_2_HCC_LC[:, 1], outsample=True, n_groups=2)
hl_train_2_HCC_LC = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_2_HCC_LC, pred_proba_train_2_HCC_LC[:, 1],np.array(data_train_HCC_LC['ID_new'])]).T).to_csv(save_path+'result_train_HCC_LC.txt',sep='\t',index=False,header=False)

pred_proba_test_2_HCC_LC= clf_2.predict_proba(x_test_2_HCC_LC)
fpr_test_HCC_LC, tpr_test_HCC_LC, thresholds = roc_curve(y_test_2_HCC_LC, pred_proba_test_2_HCC_LC[:, 1])
Sensitivity_best_test_HCC_LC, Specificity_best_test_HCC_LC, Sensitivity_adjust_test_HCC_LC, Specificity_adjust_test_HCC_LC, Sensitivity_test_HCC_LC, Specificity_test_HCC_LC=find_metrics_best_for_shuffle(fpr_test_HCC_LC, tpr_test_HCC_LC,cut_spe=0.95)
roc_auc_test_2_HCC_LC = auc(fpr_test_HCC_LC, tpr_test_HCC_LC)
bs_test_2_HCC_LC = brier_score_loss(y_test_2_HCC_LC, pred_proba_test_2_HCC_LC[:, 1])
ce = CalibrationEvaluator(y_test_2_HCC_LC, pred_proba_test_2_HCC_LC[:, 1], outsample=True, n_groups=2)
hl_test_2_HCC_LC = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_2_HCC_LC, pred_proba_test_2_HCC_LC[:, 1],np.array(data_test_HCC_LC['ID_new'])]).T).to_csv(save_path+'result_test_HCC_LC.txt',sep='\t',index=False,header=False)

#HCC VS LC 早期
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
bs_train_2_HCC_LC_0 = brier_score_loss(y_train_2_HCC_LC_0, pred_proba_train_2_HCC_LC_0[:, 1])
ce = CalibrationEvaluator(y_train_2_HCC_LC_0, pred_proba_train_2_HCC_LC_0[:, 1], outsample=True, n_groups=2)
hl_train_2_HCC_LC_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_2_HCC_LC_0, pred_proba_train_2_HCC_LC_0[:, 1],np.array(data_train_HCC_LC_0['ID_new'])]).T).to_csv(save_path+'result_train_HCC_LC_0.txt',sep='\t',index=False,header=False)

pred_proba_test_2_HCC_LC_0= clf_2.predict_proba(x_test_2_HCC_LC_0)
fpr_test_HCC_LC_0, tpr_test_HCC_LC_0, thresholds = roc_curve(y_test_2_HCC_LC_0, pred_proba_test_2_HCC_LC_0[:, 1])
Sensitivity_best_test_HCC_LC_0, Specificity_best_test_HCC_LC_0, Sensitivity_adjust_test_HCC_LC_0, Specificity_adjust_test_HCC_LC_0, Sensitivity_test_HCC_LC_0, Specificity_test_HCC_LC_0=find_metrics_best_for_shuffle(fpr_test_HCC_LC_0, tpr_test_HCC_LC_0,cut_spe=0.95)
roc_auc_test_2_HCC_LC_0 = auc(fpr_test_HCC_LC_0, tpr_test_HCC_LC_0)
bs_test_2_HCC_LC_0 = brier_score_loss(y_test_2_HCC_LC_0, pred_proba_test_2_HCC_LC_0[:, 1])
ce = CalibrationEvaluator(y_test_2_HCC_LC_0, pred_proba_test_2_HCC_LC_0[:, 1], outsample=True, n_groups=2)
hl_test_2_HCC_LC_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_2_HCC_LC_0, pred_proba_test_2_HCC_LC_0[:, 1],np.array(data_test_HCC_LC_0['ID_new'])]).T).to_csv(save_path+'result_test_HCC_LC_0.txt',sep='\t',index=False,header=False)

#HCC VS LC+HBV
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
bs_train_2_HCC_HBV_LC = brier_score_loss(y_train_2_HCC_HBV_LC, pred_proba_train_2_HCC_HBV_LC[:, 1])
ce = CalibrationEvaluator(y_train_2_HCC_HBV_LC, pred_proba_train_2_HCC_HBV_LC[:, 1], outsample=True, n_groups=2)
hl_train_2_HCC_HBV_LC = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_2_HCC_HBV_LC, pred_proba_train_2_HCC_HBV_LC[:, 1],np.array(data_train_HCC_HBV_LC['ID_new'])]).T).to_csv(save_path+'result_train_HCC_HBV_LC.txt',sep='\t',index=False,header=False)

pred_proba_test_2_HCC_HBV_LC= clf_2.predict_proba(x_test_2_HCC_HBV_LC)
fpr_test_HCC_HBV_LC, tpr_test_HCC_HBV_LC, thresholds = roc_curve(y_test_2_HCC_HBV_LC, pred_proba_test_2_HCC_HBV_LC[:, 1])
Sensitivity_best_test_HCC_HBV_LC, Specificity_best_test_HCC_HBV_LC, Sensitivity_adjust_test_HCC_HBV_LC, Specificity_adjust_test_HCC_HBV_LC, Sensitivity_test_HCC_HBV_LC, Specificity_test_HCC_HBV_LC=find_metrics_best_for_shuffle(fpr_test_HCC_HBV_LC, tpr_test_HCC_HBV_LC,cut_spe=0.95)
roc_auc_test_2_HCC_HBV_LC = auc(fpr_test_HCC_HBV_LC, tpr_test_HCC_HBV_LC)
bs_test_2_HCC_HBV_LC = brier_score_loss(y_test_2_HCC_HBV_LC, pred_proba_test_2_HCC_HBV_LC[:, 1])
ce = CalibrationEvaluator(y_test_2_HCC_HBV_LC, pred_proba_test_2_HCC_HBV_LC[:, 1], outsample=True, n_groups=2)
hl_test_2_HCC_HBV_LC = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_2_HCC_HBV_LC, pred_proba_test_2_HCC_HBV_LC[:, 1],np.array(data_test_HCC_HBV_LC['ID_new'])]).T).to_csv(save_path+'result_test_HCC_HBV_LC.txt',sep='\t',index=False,header=False)

#HCC VS LC+HBV 早期
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
bs_train_2_HCC_HBV_LC_0 = brier_score_loss(y_train_2_HCC_HBV_LC_0, pred_proba_train_2_HCC_HBV_LC_0[:, 1])
ce = CalibrationEvaluator(y_train_2_HCC_HBV_LC_0, pred_proba_train_2_HCC_HBV_LC_0[:, 1], outsample=True, n_groups=2)
hl_train_2_HCC_HBV_LC_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_2_HCC_HBV_LC_0, pred_proba_train_2_HCC_HBV_LC_0[:, 1],np.array(data_train_HCC_HBV_LC_0['ID_new'])]).T).to_csv(save_path+'result_train_HCC_HBV_LC_0.txt',sep='\t',index=False,header=False)

pred_proba_test_2_HCC_HBV_LC_0= clf_2.predict_proba(x_test_2_HCC_HBV_LC_0)
fpr_test_HCC_HBV_LC_0, tpr_test_HCC_HBV_LC_0, thresholds = roc_curve(y_test_2_HCC_HBV_LC_0, pred_proba_test_2_HCC_HBV_LC_0[:, 1])
Sensitivity_best_test_HCC_HBV_LC_0, Specificity_best_test_HCC_HBV_LC_0, Sensitivity_adjust_test_HCC_HBV_LC_0, Specificity_adjust_test_HCC_HBV_LC_0, Sensitivity_test_HCC_HBV_LC_0, Specificity_test_HCC_HBV_LC_0=find_metrics_best_for_shuffle(fpr_test_HCC_HBV_LC_0, tpr_test_HCC_HBV_LC_0,cut_spe=0.95)
roc_auc_test_2_HCC_HBV_LC_0 = auc(fpr_test_HCC_HBV_LC_0, tpr_test_HCC_HBV_LC_0)
bs_test_2_HCC_HBV_LC_0 = brier_score_loss(y_test_2_HCC_HBV_LC_0, pred_proba_test_2_HCC_HBV_LC_0[:, 1])
ce = CalibrationEvaluator(y_test_2_HCC_HBV_LC_0, pred_proba_test_2_HCC_HBV_LC_0[:, 1], outsample=True, n_groups=2)
hl_test_2_HCC_HBV_LC_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_2_HCC_HBV_LC_0, pred_proba_test_2_HCC_HBV_LC_0[:, 1],np.array(data_test_HCC_HBV_LC_0['ID_new'])]).T).to_csv(save_path+'result_test_HCC_HBV_LC_0.txt',sep='\t',index=False,header=False)

#####################AFP######################

gene_list=[]
x_train_afp_2 = np.array(data_train_afp[gene_list + ['AFP final']])
x_test_afp_2 = np.array(data_test_afp[gene_list + ['AFP final']])
y_train_afp_2 = np.array(data_train_afp['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
y_test_afp_2 = np.array(data_test_afp['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

if sample_weight == 'balanced':
    sample_weight_ = compute_sample_weight(class_weight='balanced', y=y_train_afp_2)
clf_2_afp = clf_select(model_type, seed)
clf_2_afp.fit(x_train_afp_2, y_train_afp_2, sample_weight_)

#全部
pred_proba_train_afp_2 = clf_2_afp.predict_proba(x_train_afp_2)
fpr_train_afp, tpr_train_afp, thresholds = roc_curve(y_train_afp_2, pred_proba_train_afp_2[:, 1])
Sensitivity_best_train_afp, Specificity_best_train_afp, Sensitivity_adjust_train_afp, Specificity_adjust_train_afp, Sensitivity_train_afp, Specificity_train_afp=find_metrics_best_for_shuffle(fpr_train_afp, tpr_train_afp,cut_spe=0.95)
roc_auc_train_afp_2 = auc(fpr_train_afp, tpr_train_afp)
bs_train_afp_2 = brier_score_loss(y_train_afp_2, pred_proba_train_afp_2[:, 1])
ce = CalibrationEvaluator(y_train_afp_2, pred_proba_train_afp_2[:, 1], outsample=True, n_groups=2)
hl_train_afp_2 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_afp_2, pred_proba_train_afp_2[:, 1],np.array(data_train_afp['ID_new'])]).T).to_csv(save_path+'result_train_afp.txt',sep='\t',index=False,header=False)

pred_proba_test_afp_2 = clf_2_afp.predict_proba(x_test_afp_2)
fpr_test_afp, tpr_test_afp, thresholds = roc_curve(y_test_afp_2, pred_proba_test_afp_2[:, 1])
Sensitivity_best_test_afp, Specificity_best_test_afp, Sensitivity_adjust_test_afp, Specificity_adjust_test_afp, Sensitivity_test_afp, Specificity_test_afp=find_metrics_best_for_shuffle(fpr_test_afp, tpr_test_afp,cut_spe=0.95)
roc_auc_test_afp_2 = auc(fpr_test_afp, tpr_test_afp)
bs_test_afp_2 = brier_score_loss(y_test_afp_2, pred_proba_test_afp_2[:, 1])
ce = CalibrationEvaluator(y_test_afp_2, pred_proba_test_afp_2[:, 1], outsample=True, n_groups=2)
hl_test_afp_2 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_afp_2, pred_proba_test_afp_2[:, 1],np.array(data_test_afp['ID_new'])]).T).to_csv(save_path+'result_test_afp.txt',sep='\t',index=False,header=False)

#全部早期
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
bs_train_afp_2_early = brier_score_loss(y_train_afp_2_early, pred_proba_train_afp_2_early[:, 1])
ce = CalibrationEvaluator(y_train_afp_2_early, pred_proba_train_afp_2_early[:, 1], outsample=True, n_groups=2)
hl_train_afp_2_early = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_afp_2_early, pred_proba_train_afp_2_early[:, 1],np.array(data_train_afp_early['ID_new'])]).T).to_csv(save_path+'result_train_afp_0.txt',sep='\t',index=False,header=False)

pred_proba_test_afp_2_early = clf_2_afp.predict_proba(x_test_afp_2_early)
fpr_test_afp_early, tpr_test_afp_early, thresholds = roc_curve(y_test_afp_2_early, pred_proba_test_afp_2_early[:, 1])
Sensitivity_best_test_afp_early, Specificity_best_test_afp_early, Sensitivity_adjust_test_afp_early, Specificity_adjust_test_afp_early, Sensitivity_test_afp_early, Specificity_test_afp_early=find_metrics_best_for_shuffle(fpr_test_afp_early, tpr_test_afp_early,cut_spe=0.95)
roc_auc_test_afp_2_early = auc(fpr_test_afp_early, tpr_test_afp_early)
bs_test_afp_2_early = brier_score_loss(y_test_afp_2_early, pred_proba_test_afp_2_early[:, 1])
ce = CalibrationEvaluator(y_test_afp_2_early, pred_proba_test_afp_2_early[:, 1], outsample=True, n_groups=2)
hl_test_afp_2_early = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_afp_2_early, pred_proba_test_afp_2_early[:, 1],np.array(data_test_afp_early['ID_new'])]).T).to_csv(save_path+'result_test_afp_0.txt',sep='\t',index=False,header=False)

####全部晚期
data_train_afp_late=data_train_afp.loc[data_train_afp.ID_new.isin(sample_all_late),:]
data_test_afp_late = data_test_afp.loc[data_test_afp.ID_new.isin(sample_all_late), :]
x_train_afp_2_late = np.array(data_train_afp_late[gene_list + ['AFP final']])
x_test_afp_2_late = np.array(data_test_afp_late[gene_list + ['AFP final']])
y_train_afp_2_late = np.array(data_train_afp_late['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))
y_test_afp_2_late = np.array(data_test_afp_late['Group'].replace('HCC', 1).replace('HD', 0).replace('HBV', 0).replace('LC', 0))

pred_proba_train_afp_2_late = clf_2_afp.predict_proba(x_train_afp_2_late)
fpr_train_afp_late, tpr_train_afp_late, thresholds = roc_curve(y_train_afp_2_late, pred_proba_train_afp_2_late[:, 1])
Sensitivity_best_train_afp_late, Specificity_best_train_afp_late, Sensitivity_adjust_train_afp_late, Specificity_adjust_train_afp_late, Sensitivity_train_afp_late, Specificity_train_afp_late=find_metrics_best_for_shuffle(fpr_train_afp_late, tpr_train_afp_late,cut_spe=0.95)
roc_auc_train_afp_2_late = auc(fpr_train_afp_late, tpr_train_afp_late)
bs_train_afp_2_late = brier_score_loss(y_train_afp_2_late, pred_proba_train_afp_2_late[:, 1])
ce = CalibrationEvaluator(y_train_afp_2_late, pred_proba_train_afp_2_late[:, 1], outsample=True, n_groups=2)
hl_train_afp_2_late = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_afp_2_late, pred_proba_train_afp_2_late[:, 1],np.array(data_train_afp_late['ID_new'])]).T).to_csv(save_path+'result_train_afp_late.txt',sep='\t',index=False,header=False)

pred_proba_test_afp_2_late = clf_2_afp.predict_proba(x_test_afp_2_late)
fpr_test_afp_late, tpr_test_afp_late, thresholds = roc_curve(y_test_afp_2_late, pred_proba_test_afp_2_late[:, 1])
Sensitivity_best_test_afp_late, Specificity_best_test_afp_late, Sensitivity_adjust_test_afp_late, Specificity_adjust_test_afp_late, Sensitivity_test_afp_late, Specificity_test_afp_late=find_metrics_best_for_shuffle(fpr_test_afp_late, tpr_test_afp_late,cut_spe=0.95)
roc_auc_test_afp_2_late = auc(fpr_test_afp_late, tpr_test_afp_late)
bs_test_afp_2_late = brier_score_loss(y_test_afp_2_late, pred_proba_test_afp_2_late[:, 1])
ce = CalibrationEvaluator(y_test_afp_2_late, pred_proba_test_afp_2_late[:, 1], outsample=True, n_groups=2)
hl_test_afp_2_late = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_afp_2_late, pred_proba_test_afp_2_late[:, 1],np.array(data_test_afp_late['ID_new'])]).T).to_csv(save_path+'result_test_afp_late.txt',sep='\t',index=False,header=False)

#HCC vs HD
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
bs_train_afp_2_HCC_HD = brier_score_loss(y_train_afp_2_HCC_HD, pred_proba_train_afp_2_HCC_HD[:, 1])
ce = CalibrationEvaluator(y_train_afp_2_HCC_HD, pred_proba_train_afp_2_HCC_HD[:, 1], outsample=True, n_groups=2)
hl_train_afp_2_HCC_HD = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_afp_2_HCC_HD, pred_proba_train_afp_2_HCC_HD[:, 1],np.array(data_train_afp_HCC_HD['ID_new'])]).T).to_csv(save_path+'result_train_afp_HCC_HD.txt',sep='\t',index=False,header=False)

pred_proba_test_afp_2_HCC_HD= clf_2_afp.predict_proba(x_test_afp_2_HCC_HD)
fpr_test_afp_HCC_HD, tpr_test_afp_HCC_HD, thresholds = roc_curve(y_test_afp_2_HCC_HD, pred_proba_test_afp_2_HCC_HD[:, 1])
Sensitivity_best_test_afp_HCC_HD, Specificity_best_test_afp_HCC_HD, Sensitivity_adjust_test_afp_HCC_HD, Specificity_adjust_test_afp_HCC_HD, Sensitivity_test_afp_HCC_HD, Specificity_test_afp_HCC_HD=find_metrics_best_for_shuffle(fpr_test_afp_HCC_HD, tpr_test_afp_HCC_HD,cut_spe=0.95)
roc_auc_test_afp_2_HCC_HD = auc(fpr_test_afp_HCC_HD, tpr_test_afp_HCC_HD)
bs_test_afp_2_HCC_HD = brier_score_loss(y_test_afp_2_HCC_HD, pred_proba_test_afp_2_HCC_HD[:, 1])
ce = CalibrationEvaluator(y_test_afp_2_HCC_HD, pred_proba_test_afp_2_HCC_HD[:, 1], outsample=True, n_groups=2)
hl_test_afp_2_HCC_HD = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_afp_2_HCC_HD, pred_proba_test_afp_2_HCC_HD[:, 1],np.array(data_test_afp_HCC_HD['ID_new'])]).T).to_csv(save_path+'result_test_afp_HCC_HD.txt',sep='\t',index=False,header=False)

#HCC vs HD 早期
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
bs_train_afp_2_HCC_HD_0 = brier_score_loss(y_train_afp_2_HCC_HD_0, pred_proba_train_afp_2_HCC_HD_0[:, 1])
ce = CalibrationEvaluator(y_train_afp_2_HCC_HD_0, pred_proba_train_afp_2_HCC_HD_0[:, 1], outsample=True, n_groups=2)
hl_train_afp_2_HCC_HD_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_afp_2_HCC_HD_0, pred_proba_train_afp_2_HCC_HD_0[:, 1],np.array(data_train_afp_HCC_HD_0['ID_new'])]).T).to_csv(save_path+'result_train_afp_HCC_HD_0.txt',sep='\t',index=False,header=False)

pred_proba_test_afp_2_HCC_HD_0= clf_2_afp.predict_proba(x_test_afp_2_HCC_HD_0)
fpr_test_afp_HCC_HD_0, tpr_test_afp_HCC_HD_0, thresholds = roc_curve(y_test_afp_2_HCC_HD_0, pred_proba_test_afp_2_HCC_HD_0[:, 1])
Sensitivity_best_test_afp_HCC_HD_0, Specificity_best_test_afp_HCC_HD_0, Sensitivity_adjust_test_afp_HCC_HD_0, Specificity_adjust_test_afp_HCC_HD_0, Sensitivity_test_afp_HCC_HD_0, Specificity_test_afp_HCC_HD_0=find_metrics_best_for_shuffle(fpr_test_afp_HCC_HD_0, tpr_test_afp_HCC_HD_0,cut_spe=0.95)
roc_auc_test_afp_2_HCC_HD_0 = auc(fpr_test_afp_HCC_HD_0, tpr_test_afp_HCC_HD_0)
bs_test_afp_2_HCC_HD_0 = brier_score_loss(y_test_afp_2_HCC_HD_0, pred_proba_test_afp_2_HCC_HD_0[:, 1])
ce = CalibrationEvaluator(y_test_afp_2_HCC_HD_0, pred_proba_test_afp_2_HCC_HD_0[:, 1], outsample=True, n_groups=2)
hl_test_afp_2_HCC_HD_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_afp_2_HCC_HD_0, pred_proba_test_afp_2_HCC_HD_0[:, 1],np.array(data_test_afp_HCC_HD_0['ID_new'])]).T).to_csv(save_path+'result_test_afp_HCC_HD_0.txt',sep='\t',index=False,header=False)


#HCC vs HBV
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
bs_train_afp_2_HCC_HBV = brier_score_loss(y_train_afp_2_HCC_HBV, pred_proba_train_afp_2_HCC_HBV[:, 1])
ce = CalibrationEvaluator(y_train_afp_2_HCC_HBV, pred_proba_train_afp_2_HCC_HBV[:, 1], outsample=True, n_groups=2)
hl_train_afp_2_HCC_HBV = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_afp_2_HCC_HBV, pred_proba_train_afp_2_HCC_HBV[:, 1],np.array(data_train_afp_HCC_HBV['ID_new'])]).T).to_csv(save_path+'result_train_afp_HCC_HBV.txt',sep='\t',index=False,header=False)

pred_proba_test_afp_2_HCC_HBV= clf_2_afp.predict_proba(x_test_afp_2_HCC_HBV)
fpr_test_afp_HCC_HBV, tpr_test_afp_HCC_HBV, thresholds = roc_curve(y_test_afp_2_HCC_HBV, pred_proba_test_afp_2_HCC_HBV[:, 1])
Sensitivity_best_test_afp_HCC_HBV, Specificity_best_test_afp_HCC_HBV, Sensitivity_adjust_test_afp_HCC_HBV, Specificity_adjust_test_afp_HCC_HBV, Sensitivity_test_afp_HCC_HBV, Specificity_test_afp_HCC_HBV=find_metrics_best_for_shuffle(fpr_test_afp_HCC_HBV, tpr_test_afp_HCC_HBV,cut_spe=0.95)
roc_auc_test_afp_2_HCC_HBV = auc(fpr_test_afp_HCC_HBV, tpr_test_afp_HCC_HBV)
bs_test_afp_2_HCC_HBV = brier_score_loss(y_test_afp_2_HCC_HBV, pred_proba_test_afp_2_HCC_HBV[:, 1])
ce = CalibrationEvaluator(y_test_afp_2_HCC_HBV, pred_proba_test_afp_2_HCC_HBV[:, 1], outsample=True, n_groups=2)
hl_test_afp_2_HCC_HBV = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_afp_2_HCC_HBV, pred_proba_test_afp_2_HCC_HBV[:, 1],np.array(data_test_afp_HCC_HBV['ID_new'])]).T).to_csv(save_path+'result_test_afp_HCC_HBV.txt',sep='\t',index=False,header=False)

#HCC vs HBV早期
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
bs_train_afp_2_HCC_HBV_0 = brier_score_loss(y_train_afp_2_HCC_HBV_0, pred_proba_train_afp_2_HCC_HBV_0[:, 1])
ce = CalibrationEvaluator(y_train_afp_2_HCC_HBV_0, pred_proba_train_afp_2_HCC_HBV_0[:, 1], outsample=True, n_groups=2)
hl_train_afp_2_HCC_HBV_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_afp_2_HCC_HBV_0, pred_proba_train_afp_2_HCC_HBV_0[:, 1],np.array(data_train_afp_HCC_HBV_0['ID_new'])]).T).to_csv(save_path+'result_train_afp_HCC_HBV_0.txt',sep='\t',index=False,header=False)

pred_proba_test_afp_2_HCC_HBV_0= clf_2_afp.predict_proba(x_test_afp_2_HCC_HBV_0)
fpr_test_afp_HCC_HBV_0, tpr_test_afp_HCC_HBV_0, thresholds = roc_curve(y_test_afp_2_HCC_HBV_0, pred_proba_test_afp_2_HCC_HBV_0[:, 1])
Sensitivity_best_test_afp_HCC_HBV_0, Specificity_best_test_afp_HCC_HBV_0, Sensitivity_adjust_test_afp_HCC_HBV_0, Specificity_adjust_test_afp_HCC_HBV_0, Sensitivity_test_afp_HCC_HBV_0, Specificity_test_afp_HCC_HBV_0=find_metrics_best_for_shuffle(fpr_test_afp_HCC_HBV_0, tpr_test_afp_HCC_HBV_0,cut_spe=0.95)
roc_auc_test_afp_2_HCC_HBV_0 = auc(fpr_test_afp_HCC_HBV_0, tpr_test_afp_HCC_HBV_0)
bs_test_afp_2_HCC_HBV_0 = brier_score_loss(y_test_afp_2_HCC_HBV_0, pred_proba_test_afp_2_HCC_HBV_0[:, 1])
ce = CalibrationEvaluator(y_test_afp_2_HCC_HBV_0, pred_proba_test_afp_2_HCC_HBV_0[:, 1], outsample=True, n_groups=2)
hl_test_afp_2_HCC_HBV_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_afp_2_HCC_HBV_0, pred_proba_test_afp_2_HCC_HBV_0[:, 1],np.array(data_test_afp_HCC_HBV_0['ID_new'])]).T).to_csv(save_path+'result_test_afp_HCC_HBV_0.txt',sep='\t',index=False,header=False)


#HCC vs LC
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
bs_train_afp_2_HCC_LC = brier_score_loss(y_train_afp_2_HCC_LC, pred_proba_train_afp_2_HCC_LC[:, 1])
ce = CalibrationEvaluator(y_train_afp_2_HCC_LC, pred_proba_train_afp_2_HCC_LC[:, 1], outsample=True, n_groups=2)
hl_train_afp_2_HCC_LC = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_afp_2_HCC_LC, pred_proba_train_afp_2_HCC_LC[:, 1],np.array(data_train_afp_HCC_LC['ID_new'])]).T).to_csv(save_path+'result_train_afp_HCC_LC.txt',sep='\t',index=False,header=False)

pred_proba_test_afp_2_HCC_LC= clf_2_afp.predict_proba(x_test_afp_2_HCC_LC)
fpr_test_afp_HCC_LC, tpr_test_afp_HCC_LC, thresholds = roc_curve(y_test_afp_2_HCC_LC, pred_proba_test_afp_2_HCC_LC[:, 1])
Sensitivity_best_test_afp_HCC_LC, Specificity_best_test_afp_HCC_LC, Sensitivity_adjust_test_afp_HCC_LC, Specificity_adjust_test_afp_HCC_LC, Sensitivity_test_afp_HCC_LC, Specificity_test_afp_HCC_LC=find_metrics_best_for_shuffle(fpr_test_afp_HCC_LC, tpr_test_afp_HCC_LC,cut_spe=0.95)
roc_auc_test_afp_2_HCC_LC = auc(fpr_test_afp_HCC_LC, tpr_test_afp_HCC_LC)
bs_test_afp_2_HCC_LC = brier_score_loss(y_test_afp_2_HCC_LC, pred_proba_test_afp_2_HCC_LC[:, 1])
ce = CalibrationEvaluator(y_test_afp_2_HCC_LC, pred_proba_test_afp_2_HCC_LC[:, 1], outsample=True, n_groups=2)
hl_test_afp_2_HCC_LC = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_afp_2_HCC_LC, pred_proba_test_afp_2_HCC_LC[:, 1],np.array(data_test_afp_HCC_LC['ID_new'])]).T).to_csv(save_path+'result_test_afp_HCC_LC.txt',sep='\t',index=False,header=False)


#HCC vs LC早期
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
bs_train_afp_2_HCC_LC_0 = brier_score_loss(y_train_afp_2_HCC_LC_0, pred_proba_train_afp_2_HCC_LC_0[:, 1])
ce = CalibrationEvaluator(y_train_afp_2_HCC_LC_0, pred_proba_train_afp_2_HCC_LC_0[:, 1], outsample=True, n_groups=2)
hl_train_afp_2_HCC_LC_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_afp_2_HCC_LC_0, pred_proba_train_afp_2_HCC_LC_0[:, 1],np.array(data_train_afp_HCC_LC_0['ID_new'])]).T).to_csv(save_path+'result_train_afp_HCC_LC_0.txt',sep='\t',index=False,header=False)

pred_proba_test_afp_2_HCC_LC_0= clf_2_afp.predict_proba(x_test_afp_2_HCC_LC_0)
fpr_test_afp_HCC_LC_0, tpr_test_afp_HCC_LC_0, thresholds = roc_curve(y_test_afp_2_HCC_LC_0, pred_proba_test_afp_2_HCC_LC_0[:, 1])
Sensitivity_best_test_afp_HCC_LC_0, Specificity_best_test_afp_HCC_LC_0, Sensitivity_adjust_test_afp_HCC_LC_0, Specificity_adjust_test_afp_HCC_LC_0, Sensitivity_test_afp_HCC_LC_0, Specificity_test_afp_HCC_LC_0=find_metrics_best_for_shuffle(fpr_test_afp_HCC_LC_0, tpr_test_afp_HCC_LC_0,cut_spe=0.95)
roc_auc_test_afp_2_HCC_LC_0 = auc(fpr_test_afp_HCC_LC_0, tpr_test_afp_HCC_LC_0)
bs_test_afp_2_HCC_LC_0 = brier_score_loss(y_test_afp_2_HCC_LC_0, pred_proba_test_afp_2_HCC_LC_0[:, 1])
ce = CalibrationEvaluator(y_test_afp_2_HCC_LC_0, pred_proba_test_afp_2_HCC_LC_0[:, 1], outsample=True, n_groups=2)
hl_test_afp_2_HCC_LC_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_afp_2_HCC_LC_0, pred_proba_test_afp_2_HCC_LC_0[:, 1],np.array(data_test_afp_HCC_LC_0['ID_new'])]).T).to_csv(save_path+'result_test_afp_HCC_LC_0.txt',sep='\t',index=False,header=False)

#HCC vs LC+HBV
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
bs_train_afp_2_HCC_HBV_LC = brier_score_loss(y_train_afp_2_HCC_HBV_LC, pred_proba_train_afp_2_HCC_HBV_LC[:, 1])
ce = CalibrationEvaluator(y_train_afp_2_HCC_HBV_LC, pred_proba_train_afp_2_HCC_HBV_LC[:, 1], outsample=True, n_groups=2)
hl_train_afp_2_HCC_HBV_LC = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_afp_2_HCC_HBV_LC, pred_proba_train_afp_2_HCC_HBV_LC[:, 1],np.array(data_train_afp_HCC_HBV_LC['ID_new'])]).T).to_csv(save_path+'result_train_afp_HCC_HBV_LC.txt',sep='\t',index=False,header=False)

pred_proba_test_afp_2_HCC_HBV_LC= clf_2_afp.predict_proba(x_test_afp_2_HCC_HBV_LC)
fpr_test_afp_HCC_HBV_LC, tpr_test_afp_HCC_HBV_LC, thresholds = roc_curve(y_test_afp_2_HCC_HBV_LC, pred_proba_test_afp_2_HCC_HBV_LC[:, 1])
Sensitivity_best_test_afp_HCC_HBV_LC, Specificity_best_test_afp_HCC_HBV_LC, Sensitivity_adjust_test_afp_HCC_HBV_LC, Specificity_adjust_test_afp_HCC_HBV_LC, Sensitivity_test_afp_HCC_HBV_LC, Specificity_test_afp_HCC_HBV_LC=find_metrics_best_for_shuffle(fpr_test_afp_HCC_HBV_LC, tpr_test_afp_HCC_HBV_LC,cut_spe=0.95)
roc_auc_test_afp_2_HCC_HBV_LC = auc(fpr_test_afp_HCC_HBV_LC, tpr_test_afp_HCC_HBV_LC)
bs_test_afp_2_HCC_HBV_LC = brier_score_loss(y_test_afp_2_HCC_HBV_LC, pred_proba_test_afp_2_HCC_HBV_LC[:, 1])
ce = CalibrationEvaluator(y_test_afp_2_HCC_HBV_LC, pred_proba_test_afp_2_HCC_HBV_LC[:, 1], outsample=True, n_groups=2)
hl_test_afp_2_HCC_HBV_LC = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_afp_2_HCC_HBV_LC, pred_proba_test_afp_2_HCC_HBV_LC[:, 1],np.array(data_test_afp_HCC_HBV_LC['ID_new'])]).T).to_csv(save_path+'result_test_afp_HCC_HBV_LC.txt',sep='\t',index=False,header=False)


#HCC vs LC+HBV早期
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
bs_train_afp_2_HCC_HBV_LC_0 = brier_score_loss(y_train_afp_2_HCC_HBV_LC_0, pred_proba_train_afp_2_HCC_HBV_LC_0[:, 1])
ce = CalibrationEvaluator(y_train_afp_2_HCC_HBV_LC_0, pred_proba_train_afp_2_HCC_HBV_LC_0[:, 1], outsample=True, n_groups=2)
hl_train_afp_2_HCC_HBV_LC_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_train_afp_2_HCC_HBV_LC_0, pred_proba_train_afp_2_HCC_HBV_LC_0[:, 1],np.array(data_train_afp_HCC_HBV_LC_0['ID_new'])]).T).to_csv(save_path+'result_train_afp_HCC_HBV_LC_0.txt',sep='\t',index=False,header=False)

pred_proba_test_afp_2_HCC_HBV_LC_0= clf_2_afp.predict_proba(x_test_afp_2_HCC_HBV_LC_0)
fpr_test_afp_HCC_HBV_LC_0, tpr_test_afp_HCC_HBV_LC_0, thresholds = roc_curve(y_test_afp_2_HCC_HBV_LC_0, pred_proba_test_afp_2_HCC_HBV_LC_0[:, 1])
Sensitivity_best_test_afp_HCC_HBV_LC_0, Specificity_best_test_afp_HCC_HBV_LC_0, Sensitivity_adjust_test_afp_HCC_HBV_LC_0, Specificity_adjust_test_afp_HCC_HBV_LC_0, Sensitivity_test_afp_HCC_HBV_LC_0, Specificity_test_afp_HCC_HBV_LC_0=find_metrics_best_for_shuffle(fpr_test_afp_HCC_HBV_LC_0, tpr_test_afp_HCC_HBV_LC_0,cut_spe=0.95)
roc_auc_test_afp_2_HCC_HBV_LC_0 = auc(fpr_test_afp_HCC_HBV_LC_0, tpr_test_afp_HCC_HBV_LC_0)
bs_test_afp_2_HCC_HBV_LC_0 = brier_score_loss(y_test_afp_2_HCC_HBV_LC_0, pred_proba_test_afp_2_HCC_HBV_LC_0[:, 1])
ce = CalibrationEvaluator(y_test_afp_2_HCC_HBV_LC_0, pred_proba_test_afp_2_HCC_HBV_LC_0[:, 1], outsample=True, n_groups=2)
hl_test_afp_2_HCC_HBV_LC_0 = ce.hosmerlemeshow().pvalue
pd.DataFrame(np.vstack([y_test_afp_2_HCC_HBV_LC_0, pred_proba_test_afp_2_HCC_HBV_LC_0[:, 1],np.array(data_test_afp_HCC_HBV_LC_0['ID_new'])]).T).to_csv(save_path+'result_test_afp_HCC_HBV_LC_0.txt',sep='\t',index=False,header=False)

# plt.figure(figsize=(5,4.5))
# sns.set_context("talk", font_scale=1, rc={"lines.linewidth": 2.5})
# plt.plot(fpr_train,tpr_train,'-',label='HCCMDP(HCC vs all control):%0.3f' % roc_auc_train_2,color='r', lw=2.5)
# plt.plot(fpr_train_afp,tpr_train_afp,'-',label='AFP(HCC vs all control):%0.3f' % roc_auc_train_afp_2,color='b', lw=2.5)
# plt.plot(fpr_train_HCC_HD,tpr_train_HCC_HD,'-',label='HCCMDP(HCC vs HD):%0.3f' % roc_auc_train_2_HCC_HD,color='g', lw=2.5)
# plt.plot(fpr_train_afp_HCC_HD,tpr_train_afp_HCC_HD,'-',label='AFP(HCC vs HD):%0.3f' % roc_auc_train_afp_2_HCC_HD,color='y', lw=2.5)
# plt.plot(fpr_train_HCC_HBV,tpr_train_HCC_HBV,'-',label='HCCMDP(HCC vs HBV):%0.3f' % roc_auc_train_2_HCC_HBV,color='m', lw=2.5)
# plt.plot(fpr_train_afp_HCC_HBV,tpr_train_afp_HCC_HBV,'-',label='AFP(HCC vs HBV):%0.3f' % roc_auc_train_afp_2_HCC_HBV,color='c', lw=2.5)
# plt.plot(fpr_train_HCC_LC,tpr_train_HCC_LC,'-',label='HCCMDP(HCC vs LC):%0.3f' % roc_auc_train_2_HCC_LC,color='k', lw=2.5)
# plt.plot(fpr_train_afp_HCC_LC,tpr_train_afp_HCC_LC,'-',label='AFP(HCC vs LC):%0.3f' % roc_auc_train_afp_2_HCC_LC,color='#A52A2A', lw=2.5)
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
# plt.yticks(fontproperties ='Arial', size = 15,fontweight='bold')
# plt.xticks(fontproperties ='Arial', size = 15,fontweight='bold')
# plt.ylabel('Sensitivity',fontproperties ='Arial', size = 18,fontweight='bold')
# plt.xlabel('1-Specificity',fontproperties ='Arial', size = 18,fontweight='bold')
# plt.legend(loc=4,borderaxespad = 0.,title=None,frameon=False,prop={'family':'Arial','size':9,'weight':'bold'})
# plt.ylim([0,1.01])
# plt.xlim([0,1])
# plt.tight_layout()
# plt.show()
# plt.close()
#
# plt.figure(figsize=(5,4.5))
# sns.set_context("talk", font_scale=1, rc={"lines.linewidth": 2.5})
# plt.plot(fpr_train_early,tpr_train_early,'-',label='HCCMDP(HCC vs all control):%0.3f' % roc_auc_train_2_early,color='r', lw=2.5)
# plt.plot(fpr_train_afp_early,tpr_train_afp_early,'-',label='AFP(HCC vs all control):%0.3f' % roc_auc_train_afp_2_early,color='b', lw=2.5)
# plt.plot(fpr_train_HCC_HD_0,tpr_train_HCC_HD_0,'-',label='HCCMDP(HCC vs HD):%0.3f' % roc_auc_train_2_HCC_HD_0,color='g', lw=2.5)
# plt.plot(fpr_train_afp_HCC_HD_0,tpr_train_afp_HCC_HD_0,'-',label='AFP(HCC vs HD):%0.3f' % roc_auc_train_afp_2_HCC_HD_0,color='y', lw=2.5)
# plt.plot(fpr_train_HCC_HBV_0,tpr_train_HCC_HBV_0,'-',label='HCCMDP(HCC vs HBV):%0.3f' % roc_auc_train_2_HCC_HBV_0,color='m', lw=2.5)
# plt.plot(fpr_train_afp_HCC_HBV_0,tpr_train_afp_HCC_HBV_0,'-',label='AFP(HCC vs HBV):%0.3f' % roc_auc_train_afp_2_HCC_HBV_0,color='c', lw=2.5)
# plt.plot(fpr_train_HCC_LC_0,tpr_train_HCC_LC_0,'-',label='HCCMDP(HCC vs LC):%0.3f' % roc_auc_train_2_HCC_LC_0,color='k', lw=2.5)
# plt.plot(fpr_train_afp_HCC_LC_0,tpr_train_afp_HCC_LC_0,'-',label='AFP(HCC vs LC):%0.3f' % roc_auc_train_afp_2_HCC_LC_0,color='#A52A2A', lw=2.5)
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
# plt.yticks(fontproperties ='Arial', size = 15,fontweight='bold')
# plt.xticks(fontproperties ='Arial', size = 15,fontweight='bold')
# plt.ylabel('Sensitivity',fontproperties ='Arial', size = 18,fontweight='bold')
# plt.xlabel('1-Specificity',fontproperties ='Arial', size = 18,fontweight='bold')
# plt.legend(loc=4,borderaxespad = 0.,title=None,frameon=False,prop={'family':'Arial','size':9,'weight':'bold'})
# plt.ylim([0,1.01])
# plt.xlim([0,1])
# plt.tight_layout()
# plt.show()
# plt.close()

# plt.figure(figsize=(5,4.5))
# sns.set_context("talk", font_scale=1, rc={"lines.linewidth": 2.5})
# plt.plot(fpr_test,tpr_test,'-',label='HCCMDP(HCC vs all control):%0.3f' % roc_auc_test_2,color='r', lw=2.5)
# plt.plot(fpr_test_afp,tpr_test_afp,'-',label='AFP(HCC vs all control):%0.3f' % roc_auc_test_afp_2,color='b', lw=2.5)
# plt.plot(fpr_test_HCC_HBV_LC,tpr_test_HCC_HBV_LC,'-',label='HCCMDP(HCC vs CHB and LC):%0.3f' % roc_auc_test_2_HCC_HBV_LC,color='#A52A2A', lw=2.5)
# plt.plot(fpr_test_afp_HCC_HBV_LC,tpr_test_afp_HCC_HBV_LC,'-',label='AFP(HCC vs CHB and LC):%0.3f' % roc_auc_test_afp_2_HCC_HBV_LC,color='y', lw=2.5)
# plt.plot(fpr_test_HCC_HBV,tpr_test_HCC_HBV,'-',label='HCCMDP(HCC vs CHB):%0.3f' % roc_auc_test_2_HCC_HBV,color='g', lw=2.5)
# plt.plot(fpr_test_afp_HCC_HBV,tpr_test_afp_HCC_HBV,'-',label='AFP(HCC vs CHB):%0.3f' % roc_auc_test_afp_2_HCC_HBV,color='orange', lw=2.5)
# plt.plot(fpr_test_HCC_LC,tpr_test_HCC_LC,'-',label='HCCMDP(HCC vs LC):%0.3f' % roc_auc_test_2_HCC_LC,color='m', lw=2.5)
# plt.plot(fpr_test_afp_HCC_LC,tpr_test_afp_HCC_LC,'-',label='AFP(HCC vs LC):%0.3f' % roc_auc_test_afp_2_HCC_LC,color='c', lw=2.5)
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
# plt.yticks(fontproperties ='Arial', size = 15,fontweight='bold')
# plt.xticks(fontproperties ='Arial', size = 15,fontweight='bold')
# plt.ylabel('Sensitivity',fontproperties ='Arial', size = 18,fontweight='bold')
# plt.xlabel('1-Specificity',fontproperties ='Arial', size = 18,fontweight='bold')
# plt.legend(loc=4,borderaxespad = 0.,title=None,frameon=False,prop={'family':'Arial','size':9,'weight':'bold'})
# plt.ylim([0,1.01])
# plt.xlim([0,1])
# plt.tight_layout()
# plt.show()
# plt.close()
#
# plt.figure(figsize=(5,4.5))
# sns.set_context("talk", font_scale=1, rc={"lines.linewidth": 2.5})
# plt.plot(fpr_test_early,tpr_test_early,'-',label='HCCMDP(HCC vs all control):%0.3f' % roc_auc_test_2_early,color='r', lw=2.5)
# plt.plot(fpr_test_afp_early,tpr_test_afp_early,'-',label='AFP(HCC vs all control):%0.3f' % roc_auc_test_afp_2_early,color='b', lw=2.5)
# plt.plot(fpr_test_HCC_HBV_LC_0,tpr_test_HCC_HBV_LC_0,'-',label='HCCMDP(HCC vs CHB and LC):%0.3f' % roc_auc_test_2_HCC_HBV_LC_0,color='#A52A2A', lw=2.5)
# plt.plot(fpr_test_afp_HCC_HBV_LC_0,tpr_test_afp_HCC_HBV_LC_0,'-',label='AFP(HCC vs CHB and LC):%0.3f' % roc_auc_test_afp_2_HCC_HBV_LC_0,color='y', lw=2.5)
# plt.plot(fpr_test_HCC_HBV_0,tpr_test_HCC_HBV_0,'-',label='HCCMDP(HCC vs CHB):%0.3f' % roc_auc_test_2_HCC_HBV_0,color='g', lw=2.5)
# plt.plot(fpr_test_afp_HCC_HBV_0,tpr_test_afp_HCC_HBV_0,'-',label='AFP(HCC vs CHB):%0.3f' % roc_auc_test_afp_2_HCC_HBV_0,color='orange', lw=2.5)
# plt.plot(fpr_test_HCC_LC_0,tpr_test_HCC_LC_0,'-',label='HCCMDP(HCC vs LC):%0.3f' % roc_auc_test_2_HCC_LC_0,color='m', lw=2.5)
# plt.plot(fpr_test_afp_HCC_LC_0,tpr_test_afp_HCC_LC_0,'-',label='AFP(HCC vs LC):%0.3f' % roc_auc_test_afp_2_HCC_LC_0,color='c', lw=2.5)
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
# plt.yticks(fontproperties ='Arial', size = 15,fontweight='bold')
# plt.xticks(fontproperties ='Arial', size = 15,fontweight='bold')
# plt.ylabel('Sensitivity',fontproperties ='Arial', size = 18,fontweight='bold')
# plt.xlabel('1-Specificity',fontproperties ='Arial', size = 18,fontweight='bold')
# plt.legend(loc=4,borderaxespad = 0.,title=None,frameon=False,prop={'family':'Arial','size':9,'weight':'bold'})
# plt.ylim([0,1.01])
# plt.xlim([0,1])
# plt.tight_layout()
# plt.show()
# plt.close()


plt.figure(figsize=(8,4.5))
sns.set_context("talk", font_scale=1, rc={"lines.linewidth": 2.5})
plt.plot(fpr_test,tpr_test,'-',label='HCCMDP(all control)',color='r', lw=2.5)
plt.plot(fpr_test_afp,tpr_test_afp,'-',label='AFP(all control)',color='b', lw=2.5)
plt.plot(fpr_test_HCC_HBV_LC,tpr_test_HCC_HBV_LC,'-',label='HCCMDP(CHB and LC)',color='#A52A2A', lw=2.5)
plt.plot(fpr_test_afp_HCC_HBV_LC,tpr_test_afp_HCC_HBV_LC,'-',label='AFP(CHB and LC)' ,color='y', lw=2.5)
plt.plot(fpr_test_HCC_HBV,tpr_test_HCC_HBV,'-',label='HCCMDP(CHB)',color='g', lw=2.5)
plt.plot(fpr_test_afp_HCC_HBV,tpr_test_afp_HCC_HBV,'-',label='AFP(CHB)',color='orange', lw=2.5)
plt.plot(fpr_test_HCC_LC,tpr_test_HCC_LC,'-',label='HCCMDP(LC)',color='m', lw=2.5)
plt.plot(fpr_test_afp_HCC_LC,tpr_test_afp_HCC_LC,'-',label='AFP(LC)',color='c', lw=2.5)
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
plt.yticks(fontproperties ='Arial', size = 15,fontweight='bold')
plt.xticks(fontproperties ='Arial', size = 15,fontweight='bold')
plt.ylabel('Sensitivity',fontproperties ='Arial', size = 18,fontweight='bold')
plt.xlabel('1-Specificity',fontproperties ='Arial', size = 18,fontweight='bold')
plt.legend(loc=2, bbox_to_anchor=(1.2, 0.8),borderaxespad = 0.,title=None,frameon=False,prop={'family':'Arial','size':9,'weight':'bold'})
plt.ylim([0,1.01])
plt.xlim([0,1])
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(8,4.5))
sns.set_context("talk", font_scale=1, rc={"lines.linewidth": 2.5})
plt.plot(fpr_test_early,tpr_test_early,'-',label='HCCMDP(all control)',color='r', lw=2.5)
plt.plot(fpr_test_afp_early,tpr_test_afp_early,'-',label='AFP(all control)' ,color='b', lw=2.5)
plt.plot(fpr_test_HCC_HBV_LC_0,tpr_test_HCC_HBV_LC_0,'-',label='HCCMDP(CHB and LC)' ,color='#A52A2A', lw=2.5)
plt.plot(fpr_test_afp_HCC_HBV_LC_0,tpr_test_afp_HCC_HBV_LC_0,'-',label='AFP(CHB and LC)' ,color='y', lw=2.5)
plt.plot(fpr_test_HCC_HBV_0,tpr_test_HCC_HBV_0,'-',label='HCCMDP(CHB)' ,color='g', lw=2.5)
plt.plot(fpr_test_afp_HCC_HBV_0,tpr_test_afp_HCC_HBV_0,'-',label='AFP(CHB)' ,color='orange', lw=2.5)
plt.plot(fpr_test_HCC_LC_0,tpr_test_HCC_LC_0,'-',label='HCCMDP(LC)' ,color='m', lw=2.5)
plt.plot(fpr_test_afp_HCC_LC_0,tpr_test_afp_HCC_LC_0,'-',label='AFP(LC)',color='c', lw=2.5)
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
plt.yticks(fontproperties ='Arial', size = 15,fontweight='bold')
plt.xticks(fontproperties ='Arial', size = 15,fontweight='bold')
plt.ylabel('Sensitivity',fontproperties ='Arial', size = 18,fontweight='bold')
plt.xlabel('1-Specificity',fontproperties ='Arial', size = 18,fontweight='bold')
plt.legend(loc=2, bbox_to_anchor=(1.2, 0.8),borderaxespad = 0.,title=None,frameon=False,prop={'family':'Arial','size':9,'weight':'bold'})
plt.ylim([0,1.01])
plt.xlim([0,1])
plt.tight_layout()
plt.show()
plt.close()
