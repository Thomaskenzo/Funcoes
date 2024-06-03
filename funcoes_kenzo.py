import sys
import time
import pandas as pd
from time import perf_counter
import chime
chime.theme(chime.themes()[2])

def tictoc(func):
    def wrapper():
        t1 = time.perf_counter()
        func()
        print(f'{func.__name__} : {round(time.perf_counter() - t1,1)} seconds')
    return wrapper


def sucess():
    for x in range(0,3):
        chime.success()
        time.sleep(0.6)
    return

def error():
    chime.error()
    time.sleep(4)
    return

def timeit():
    return perf_counter()

def delta_time(start, end):
    chime.success()
    time.sleep(0.6)
    print(f'Tempo de execucao Start - End: {round((end - start)/60,2)} Minutos')
    return 

def agrup(group, dic):
    '''dic = {'Nome_do_agrupamento': ('Coluna_a_ser_agrupada', 'funcao')}\n
Agrupamento = dataframe.groupby(by=['Coluna_Chave']).apply(fk.agrup,dic)\n\n
mean: Compute mean of groups\n
sum: Compute sum of group values\n
size: Compute group sizes\n
count: Compute count of group\n
std: Standard deviation of groups\n
var: Compute variance of groups\n
sem: Standard error of the mean of groups\n
min: Compute min of group values\n
max: Compute max of group values'''
    agrupado = {}
    for key in dic:
        agrupado[key] = group[dic[key][0]].agg(dic[key][1])
    return pd.Series(agrupado)

def grid_search(clf, x_train, y_train, params, scores, cv):
    grid = GridSearchCV(clf, params, scoring=score, cv=cv, return_train_score=True, njobs=-1, verbose=2)
    grid_fitted = grid.fit(x_train,np.ravel(y_train))
    
    print(f'Best Score: {round(grid_fitted.best_score_,4)}')
    print(f'Best parameters: {grid_fitted.best_params_}\n')
    return grid_fitted.best_score_, grid_fitted.best_params_, grid_fitted.best_estimator_

def select_estimator(X_train, y_train, cv, scoring, models, params):
    '''models = {'LogisticRegression':LogisticRegression(random_state=42),
'LightGBM': lightgbm.LGBMClassifier(random_state=42,class_weight='balanced')}\n
params = {'LogisticRegression':{'penalty':['l1','l2'], 'C':[10, 1, 0.1, 0.01]},
          'LightGBM':{'n_estimator':[16,32,64,128],'learning_rate':[0.5,0.3,0.1],'max_depth':[3,4,5]}}\n
    df_res = select_estimator(x_train, y_train['target'], KFold(4,shuffle=True), scoring='precision', models=models,params=params)\n
    best_clf = df_res.iloc[0][3]
    '''
    estimators = []
    b_scores = []
    b_params = []
    b_estimator = []
    keys = models.keys()
    
    for key in keys:
        print(f'Running Gridsearch for {key}')
        estimators.append(key)
        score, param, esti = grid_search(models[key], X_train, y_train, params[key], scoring, cv)
        b_scores.append(score)
        b_params.append(param)
        b_estimator.append(esti) 
    return pd.DataFrame.from_dict({
            'estimator':estimators,
            'best_score':b_scores,
            'best_params':b_params,
            'model':b_estimator}).sort_values(by=['best_score'],
                                              ascending=False).reset_index(drop=True)
def evaluate_auc(model, test_features, test_labels):
    prop_1_test = test_labels.sum()/test_labels.shape[0]
    pred_proba = model.predict_proba(test_features)[:,1]
    predictions = model.predict(test_features)
    
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    auc = roc_auc_score(test_labels, pred_proba)
    
    print(f'Model Performance: ')
    print(f'Proporção de 1 no teste: {round(prop_1_test,2)}')
    print(f'Accuracy: {round(accuracy,2)}')
    print(f'Precision: {round(precision,2)}')
    print(f'Recall: {round(recall,2)}')
    print(f'F1_Score: {round(f1,2)}')
    print(f'AUC: {round(auc,2)}')
    
    return auc

def evaluate_model_thresh(model, test_features, test_labels, threshold):
    prop_1_test = test_labels.sum()/test_labels.shape[0]
    pred_proba = model.predict_proba(test_features)[:,1]
    predictions = np.where(pred_proba > threshold,1,0)
    
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    auc = roc_auc_score(test_labels, pred_proba)
    
    print(f'Model Performance: ')
    print(f'Proporção de 1 no teste: {round(prop_1_test,2)}')
    print(f'Accuracy: {round(accuracy,2)}')
    print(f'Precision: {round(precision,2)}')
    
    print(f'Recall: {round(recall,2)}')
    print(f'F1_Score: {round(f1,2)}')
    print(f'AUC: {round(auc,2)}')
    return auc                                         

def tabela_percentis_recall_precision(data, name_prob,name_true,quantiles):
    data.sort_values(by=[name_prob],ascending=False).reset_index(inplace=True,drop=True)
    first=True
    for i, q in enumerate(quantiles):
        tam1=round(data.shape[0]*q)
        tam0 = data.shape[0] - tam1
        pred = np.concatenate((np.repeat(1,tam1), np.repeat(0,tam0)),axis=1)
        precision = precision_score(data[name_true], pred)*100
        recall = recall_score(data[name_true],pred)*100
        f1 = f1_score(data[name_true], pred)*100
        accuracy = accuracy_score(data[name_true],pred)*100
        info = {'percentil': 100*q,
                'quantidade':tam1,
                'ponto de corte':min(data.loc[0:tam1,name_prob]),
                'recall':f'{round(recall,1)}',
                'precision':f'{round(precision,1)}',
                'f1':f'{round(f1,1)}',
                'accuracy':f'{round(accuracy,1)}'}
        resultados_aux = pd.DataFrame(info, index=[str(i)])
        if first:
            results = resultados_aux
            first = False
        else:
            results = pd.concat([results,resultados_aux],axis=0)
    results = results[['percentil','quantidade','ponto de corte','recall','precision','f1','accuracy']].astype(float)
    return results

def list_chunker(lst, n):
    """
    chunker = list_chunker(uma_lista,um_valor_n)\n
    for i,u in enumerate(chunker):
    """
    for i in range(0,len(lst), n):
        yield lst[i:i+n]
    return

def help():
    from inspect import getmembers, isfunction
    print(f'Lista de Funções do modulo: ')
    explicacao = {'sucess':'Executa Som de Sucesso.',
                  'error':'Executa Som de Erro.',
                  'timeit':'Retorna momento de execucao.',
                  'delta_time':'Printa tempo de execução. (start, end).',
                  'help':'Printa possíveis funções do modulo.'}
    remove_func = ['']
    for x in getmembers(sys.modules[__name__], isfunction):
        if x[0] not in remove_func:
            print(f'Função {x[0]}: {explicacao[x[0]]}')
    return
