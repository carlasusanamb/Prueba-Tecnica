# -*- coding: utf-8 -*-
"""
Código para determinar los posibles buenos pagadores de una entidad bancaria X
Se debe ejecutar or partes para garantizar su buen funcionamiento
"""
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy import stats
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import pickle
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json
import unicodedata

#CARGO DATA

data = pd.read_csv('C:/Users/Carla/Desktop/prueba_platzi/BD.csv', encoding= 'unicode_escape',sep=",")

columnas = data.columns
data.head()

data = data.rename(columns={'DEFAULT_A_30 DIAS'  : 'Default'})

#VEO EXISTENCIA DE NAN
data.isna().sum()


#CAMBIO ORDEN EN DEFAULT
data['Default'].value_counts()
#

#CAMBIO ORDEN EN DEFAULT
data["Default"] = 1 - data["Default"]
data["Default"].value_counts()



#ELIMINO VARIABLES
data = data.drop(["Unnamed: 0","...1","...2","Applicant_ID","Tipo_producto","Fecha_de_operacion","Fecha de nacimiento","Estado_residencia",
                  "Nombre_empresa","Ciudad",
"Primer_uso_TDC","Primera_Morosidad30","Primera_Morosidad60","Primera_Morosidad90",
"Mora_Max_12Meses","Primera_Morosidad30_","DIFERENCIA Primera_Morosidad30 Y Fecha_de_operacion",
"OCURRENCIA A 30 DIAS","Primera_Morosidad60_",
"DIFERENCIA Primera_Morosidad60 Y Fecha_de_operacion",
"OCURRENCIA A 60 DIAS","DEFAULT_A_60_DIAS","Primera_Morosidad90_",
"DIFERENCIA Primera_Morosidad90 Y Fecha_de_operacion","OCURRENCIA 90 DIAS",
"DEFAULT_A_90_DIAS","Unnamed: 0","ESTADO CUENTA"  ], axis=1)


#VEO CORRELACION ENTRE VARIABLES
corrMatrix = data.corr()

##############
##############
#CALCULAR IV
def iv_woe(data, target, bins=10, show_woe=False):
    
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF

#APPLY FUNCTION
iv, woe = iv_woe(data = data, target = 'Default', bins=10, show_woe = True)
print(iv)
#print(woe)




categorical_cols = ['ESTADO CIVIL', 'Genero', 'Profesion', 'Nivel educativo']

for col in categorical_cols:
    data[col] = data[col].astype('category')

data= pd.get_dummies(data, prefix_sep='__', columns=categorical_cols)


data.head()

#VEO CORRELACION ENTRE VARIABLES
corrMatrix = data.corr()



dat = data



#
X = data.drop(['Default'],axis=1)
Y = data['Default']

X.isnull().sum()
Y.isnull().sum()

np.isnan(X).any()
np.isinf(X).any()

np.isfinite(X).any()

# Asumiendo X_cleaned y y_cleaned datasets
from sklearn.model_selection import train_test_split

X_train, X_test_val, y_train, y_test_val = train_test_split(X, Y, test_size=0.30, random_state=42, stratify=Y) 
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.40, random_state=42, stratify=y_test_val)


#
pbounds = {
    'learning_rate': (0.005, 0.8),
    'n_estimators': (100, 500),
    'max_depth': (5,15),
    'scale_pos_weight': (0.08,0.1),
    'subsample': (0.3, 0.3),
    'colsample': (0.5, 0.5),  
    'gamma': (0.1, 5)
}

def xgboost_hyper_param(learning_rate,
                        n_estimators,
                        max_depth,
                        subsample,
                        colsample,
                        gamma,
                        #colsample_bytree
                        #eta,
                        #min_child_weight,
                        scale_pos_weight
                        #reg_alpha,
                        #reg_lambda
                        ):
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)
    clf = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        gamma=gamma,
        #colsample_bytree=colsample_bytree,
        #eta=eta,
        #min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
        #reg_lambda=reg_lambda,
        #reg_alpha=reg_alpha,
        objetive='binary:logistic',
        )
    return np.mean(cross_val_score(clf, X_train, y_train, cv=3, scoring='balanced_accuracy'))
optimizer = BayesianOptimization(
    f=xgboost_hyper_param,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(init_points=5, n_iter=5)

#PARAMETROS OPTIMIZADOS
optimizer.max['params']

params = optimizer.max['params']
params['max_depth'] = np.int(params['max_depth'])
params['n_estimators'] = np.int(params['n_estimators'])

params["objective"] = 'binary:logistic'

model = XGBClassifier(**params,random_state=42)
model.fit(X_train, y_train)


#
y_train_pred = model.predict(X_train)

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_train, y_train_pred))


y_pred = model.predict_proba(X_train)
y_score = [y[1] for y in y_pred]

model_performance = pd.DataFrame({"accuracy":[],"precision":[],"recall":[],"f1":[],"cut":[]})
for x in range(0,100):
    pred = [1 if y > (x*0.01) else 0 for y in y_score]
    acc = (y_train == pred).mean()
    prec = sum(np.multiply(y_train,pred)) / (sum( np.multiply([ 1 - pr for pr in y_train], pred)) + sum(np.multiply(y_train,pred)))
    reca = sum(np.multiply(y_train,pred)) / (sum( np.multiply([ 1 - pr for pr in pred], y_train)) + sum(np.multiply(y_train,pred)))
    f_1 = 2 * prec * reca/ (prec + reca)
    model_performance = model_performance.append({
        "accuracy": acc,
        "precision": prec,
        "recall": reca,
        "f1": f_1,
        "cut": x
    }, ignore_index=True)
model_performance

model_performance.iloc[model_performance['f1'].idxmax()]

threshold = 0.13


#################
predConThres = [1 if y > threshold else 0 for y in y_score]

d = y_train
d = d.reset_index()

d1 = pd.DataFrame(y_score)
d1 = d1.rename(columns={ 0 : 'Score'})
d1["Default"] = d["Default"]

ks = stats.ks_2samp(d1[d1["Default"]==0]['Score'], d1[d1["Default"]==1]['Score']).statistic
ks



#%matplotlib inline
def plot_roc_curve(y, y_pred, gini, ks):
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, 'b--', label='%s AUC = %0.4f, GINI = %0.2f, KS = %s' % ('Model: ', roc_auc, gini, ks))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0, fontsize='small')
    plt.show()
    
gini = 2 * roc_auc_score(y_train, y_score) - 1


print('GINI = %s, KS = %s' % (gini, ks))
plot_roc_curve(y_train, y_score, gini, ks)

#MATRIZ DE CONFUSION
print(accuracy_score(y_train, predConThres))
print(f1_score(y_train, predConThres))
confusion_matrix(y_train, predConThres)



##########
#VER METRICAS SOBRE DATA DE VALIDACION 
y_pred = model.predict_proba(X_val)
y_score = [y[1] for y in y_pred]

threshold = 0.13

#CALCULO METRICAS SOBRE DATA DE ENTRENAMIENTO
predConThres = [1 if y > threshold else 0 for y in y_score]

d = y_val
d = d.reset_index()

d1 = pd.DataFrame(y_score)
d1 = d1.rename(columns={ 0 : 'Score'})
d1["Default"] = d["Default"]

ks = stats.ks_2samp(d1[d1["Default"]==0]['Score'], d1[d1["Default"]==1]['Score']).statistic
ks


#%matplotlib inline
def plot_roc_curve(y, y_pred, gini, ks):
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, 'b--', label='%s AUC = %0.4f, GINI = %0.2f, KS = %s' % ('Model: ', roc_auc, gini, ks))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0, fontsize='small')
    plt.show()
    
    
gini = 2 * roc_auc_score(y_val, y_score) - 1



print('GINI = %s, KS = %s' % (gini, ks))
plot_roc_curve(y_val, y_score, gini, ks)

#MATRIZ DE CONFUSION
print(accuracy_score(y_val, predConThres))
print(f1_score(y_val, predConThres))
confusion_matrix(y_val, predConThres)


##########
#VER METRICAS SOBRE DATA DE TEST

y_pred = model.predict_proba(X_test)
y_score = [y[1] for y in y_pred]

threshold = 0.13

#################
predConThres = [1 if y > threshold else 0 for y in y_score]


d = y_test
d = d.reset_index()

d1 = pd.DataFrame(y_score)
d1 = d1.rename(columns={ 0 : 'Score'})
d1["Default"] = d["Default"]

ks = stats.ks_2samp(d1[d1["Default"]==0]['Score'], d1[d1["Default"]==1]['Score']).statistic
ks


#%matplotlib inline
def plot_roc_curve(y, y_pred, gini, ks):
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, 'b--', label='%s AUC = %0.4f, GINI = %0.2f, KS = %s' % ('Model: ', roc_auc, gini, ks))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0, fontsize='small')
    plt.show()
    
    
gini = 2 * roc_auc_score(y_test, y_score) - 1



print('GINI = %s, KS = %s' % (gini, ks))
plot_roc_curve(y_test, y_score, gini, ks)

#MATRIZ DE CONFUSION
print(accuracy_score(y_test, predConThres))
print(f1_score(y_test, predConThres))
confusion_matrix(y_test, predConThres)



#EXPORTO DATA APROBADOS
#APLICO MODELO A TODA LA DATA
#desde aqui
y_pred_T = model.predict_proba(X)
y_score_T = [y[1] for y in y_pred_T] #OJO CON EL CAMBIO



threshold
    


predConThres_T = [1 if y > threshold else 0 for y in y_score_T]

pred_T = pd.DataFrame(predConThres_T)
pred_T = pred_T.rename(columns={ 0 : 'Default'})

pred_T['Default'].value_counts()


#EXPORTO SCORE DATA COMPLETA


 y_score_T = pd.DataFrame(y_score_T)
 y_score_T = y_score_T.rename(columns={ 0 : 'Score'})
 y_score_T["Default"] = Y
 
 
 y_score_T.to_csv("C:/Users/Carla/Desktop/prueba_platzi/data_aprobados.csv",index=False)



#EXPORTO DATA TEST
#APLICO MODELO A TODA LA DATA
y_pred_T = model.predict_proba(X_test)
y_score_T = [y[1] for y in y_pred_T] #OJO CON EL CAMBIO




threshold
    


predConThres_T = [1 if y > threshold else 0 for y in y_score_T]

pred_T = pd.DataFrame(predConThres_T)
pred_T = pred_T.rename(columns={ 0 : 'Default'})

pred_T['Default'].value_counts()


#EXPORTO SCORE DATA COMPLETA
y_test = y_test.reset_index()

 y_score_T = pd.DataFrame(y_score_T)
 y_score_T = y_score_T.rename(columns={ 0 : 'Score'})
 y_score_T["Default"] = y_test['Default']

 y_score_T.to_csv("C:/Users/Carla/Desktop/prueba_platzi/data_test_2.csv",index=False)







