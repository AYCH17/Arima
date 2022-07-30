#%%
## For data
import pandas as pd
import numpy as np
## For plotting
import matplotlib.pyplot as plt
## For Arima
import pmdarima
from pmdarima.utils.array import _diff_inv_vector

import statsmodels.tsa.api as sm
from statsmodels.tsa.arima_model import ARIMA


df = pd.read_csv('us-counties.csv')

df.head()

df.tail()
# %%
###preparation des donnees

#additionner les valeurs (cas et deces)de chaque date
df = df.groupby('date').sum()
# eliminer la colonne fips
df = df.drop("fips", axis= 1)
#Renommer les colonnes
df.columns = ['nouveaux_cas', 'décès']
#afficher les 5 premieres valeurs
df.head()
#afficher les 5 dernieres valeurs
df.tail()
# %%
df.query("décès == '' ")
#separer les cas et les deces
#%%
df_cas = pd.DataFrame(df['nouveaux_cas'])
df_deces = pd.DataFrame(df['décès'])

df_cas.head()
df_deces.head()


# %%
def split_train_test(df, test=0.20, plot=True, figsize=(15,5)):
    ## define splitting point
    if type(test) is float:
        split = int(len(df)*(1-test))
        perc = test
    elif type(test) is str:
        split = df.reset_index()[ 
                      df.reset_index().iloc[:,0]==test].index[0]
        perc = round(len(df[split:])/len(df), 2)
    else:
        split = test
        perc = round(len(df[split:])/len(df), 2)
    
    print("--- splitting at index: ", split, "|", 
          df.index[split], "| test size:", perc, " ---")
    
    ## split df
    df_train = df.head(split)
    df_test = df.tail(len(df)-split)
    if plot is True:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, 
                               sharey=True, figsize=figsize)
        df_train.plot(ax=ax[0], grid=True, title="Train", 
                      color="black")
        df_test.plot(ax=ax[1], grid=True, title="Test", 
                     color="black")
        ax[0].set(xlabel=None)
        ax[1].set(xlabel=None)
        plt.show()
        
    return df_train, df_test

#%%
def show_results(df_train, df_test,atr : str, plot= True, figsize=(20,15)):

    #residus et erreurs sur le modele et la prediction
    df_train['residus'] = df_train[atr] - df_train['modele']
    df_test['erreur'] = df_test[atr][10:] - df_test['prediction'][10:]
    df_test["erreur_pct"] = df_test["erreur"] / df_test[atr][10:]

    df_train.describe()
    df_test.describe()

    if plot==True:
            fig = plt.figure(figsize=figsize)
            #fig.suptitle(atr, fontsize=20)   
            ax1 = fig.add_subplot(2,2, 1)
            ax2 = fig.add_subplot(2,2, 2)
            ax3 = fig.add_subplot(2,2, 3)
            fig2 = plt.figure(figsize=figsize)

            ax4 = fig2.add_subplot(2,2, 1, sharey=ax1)
            ax5 = fig2.add_subplot(2,2, 2, sharey=ax1)
            ax6 = fig2.add_subplot(2,2, 3)
            ax7 = fig2.add_subplot(2,2, 4)



            df_train.plot(ax = ax1, title ='Tout',color = ['black','blue','red'])
            df_train.plot(ax = ax2, title = 'Modele', y = 'modele', color = 'blue')
            df_train.plot(ax = ax3, title = 'Residus',y = 'residus', color = 'red')

            df_test.plot(ax = ax4, title = 'Reel', y = atr, color ='black')
            df_test.plot(ax = ax5, title = 'Prediction', y = 'prediction', color = 'green')
            df_test.plot(ax = ax6, title = 'Erreur', y = 'erreur', color = 'red')
            df_test.plot(ax = ax7, title = 'Pct Erreur', y = 'erreur_pct', color = 'pink')

    return df_train[[atr,"modele","residus"]],  df_test[[atr, "prediction","erreur"]]

    

#%%#
def apply_arima(df_train, df_test,atr, order, seasonal_order, figsize = (20,15)):
    
    ## train
    model = sm.SARIMAX(df_train, order=order, seasonal_order=seasonal_order).fit()
    

    df_train["modele"] = model.fittedvalues
    
    ## test
    #predict = pd.Series(model.predict(start = len(df_train), end= len(df_train)+len(df_test)-1)).to_frame()

    #predict = pd.Series(model.predict(start = len(df_train), end= len(df_train)+len(df_test)-1)[1])
    #predict.rename('prediction')
    #predict.columns = ['prediction']
    #predict.index.rename('date', inplace=True)
    print(df_test,len(df_test))
    #print(predict, len(predict))
    #df_test = pd.concat([df_test, predict], axis =0)
    predict = model.predict(start = len(df_train), end= len(df_train)+len(df_test)-1).to_list()

    print('pred',predict)
    df_test['prediction']=predict

    print(df_test)
    ## evaluer
    #df = df_train.append(df_test)
    title = "ARIMA "+str(order) 
    title = "S"+title+" x "+str(seasonal_order) if np.sum(seasonal_order) > 0 else title
    
    df_train, df_test = show_results(df_train, df_test, atr, figsize=figsize)
    
    return df_train, df_test, model



'''
best_model_cas = pmdarima.auto_arima(df_cas,                                   
                                 seasonal=True, stationary=False, 
                                 m=7, information_criterion='aic', 
                                 max_order=20,                                     
                                 max_p=10, max_d=3, max_q=10,                                     
                                 max_P=10, max_D=3, max_Q=10,                                   
                                 error_action='ignore')
print(f"best model (nouveaux cas) --> (p, d, q): {best_model_cas.order}  and  (P, D, Q, s):{best_model_cas.seasonal_order}")


best_model_deces = pmdarima.auto_arima(df_deces, 
                                 seasonal=True, stationary=False, 
                                 m=7, information_criterion='aic', 
                                 max_order=20,                                     
                                 max_p=10, max_d=3, max_q=10,                                     
                                 max_P=10, max_D=3, max_Q=10,                                   
                                 error_action='ignore')
print(f"best model (deces) --> (p, d, q): {best_model_deces.order}  and  (P, D, Q, s):{best_model_deces.seasonal_order}")
'''

#%%
df_train_cas, df_test_cas = split_train_test(df_cas)
df_train_deces, df_test_deces = split_train_test(df_deces)

#%%
df_train_deces,df_test_deces, model_deces = apply_arima(df_train_deces, df_test_deces,atr='décès', order=(3, 2, 2), 
                         seasonal_order=(1,0,1,7))


df_train_cas, df_test_cas, model_cas = apply_arima(df_train_cas, df_test_cas,atr='nouveaux_cas', order=(3, 2, 2), 
                         seasonal_order=(1,0,1,7))

#%%