#%%
## For data
import pandas as pd
import numpy as np
## For plotting
import matplotlib.pyplot as plt
## For Arima
import pmdarima

import statsmodels.tsa.api as sm

## For Lstm
#from tensorflow.keras import models, layers, preprocessing as kprocessing

#lecture des donnees
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

#fonction pour l'affichage de l'analyse des resultats avec des graphes
#%%
def utils_evaluate_forecast(dtf,f , title, plot=True, figsize=(20,13)):
  
        
        ## residuals
        dtf["residuals"] = dtf[f] - dtf["model"]
        dtf["error"] = dtf[f] - dtf["forecast"]
        dtf["error_pct"] = dtf["error"] / dtf[f]
        
        ## kpi
        residuals_mean = dtf["residuals"].mean()
        residuals_std = dtf["residuals"].std()
        error_mean = dtf["error"].mean()
        error_std = dtf["error"].std()
        mae = dtf["error"].apply(lambda x: np.abs(x)).mean()
        mape = dtf["error_pct"].apply(lambda x: np.abs(x)).mean()  
        mse = dtf["error"].apply(lambda x: x**2).mean()
        rmse = np.sqrt(mse)  #root mean squared error
        
        ## intervals
        dtf["conf_int_low"] = dtf["forecast"] - 1.96*residuals_std
        dtf["conf_int_up"] = dtf["forecast"] + 1.96*residuals_std
        dtf["pred_int_low"] = dtf["forecast"] - 1.96*error_std
        dtf["pred_int_up"] = dtf["forecast"] + 1.96*error_std
        
        ## plot
        if plot==True:
            fig = plt.figure(figsize=figsize)
            fig.suptitle(title, fontsize=20)   
            ax1 = fig.add_subplot(2,2, 1)
            ax2 = fig.add_subplot(2,2, 2, sharey=ax1)
            ax3 = fig.add_subplot(2,2, 3)
            ax4 = fig.add_subplot(2,2, 4)
            ### training
            dtf[pd.notnull(dtf["model"])][[f,"model"]].plot(color=["black","green"], title="Model", grid=True, ax=ax1)      
            ax1.set(xlabel=None)
            ### test
            dtf[pd.isnull(dtf["model"])][[f,"forecast"]].plot(color=["black","red"], title="Forecast", grid=True, ax=ax2)
            ax2.fill_between(x=dtf.index, y1=dtf['pred_int_low'], y2=dtf['pred_int_up'], color='b', alpha=0.2)
            ax2.fill_between(x=dtf.index, y1=dtf['conf_int_low'], y2=dtf['conf_int_up'], color='b', alpha=0.3)     
            ax2.set(xlabel=None)
            ### residuals
            dtf[["residuals","error"]].plot(ax=ax3, color=["green","red"], title="Residuals", grid=True)
            ax3.set(xlabel=None)
            ### residuals distribution
            dtf[["residuals","error"]].plot(ax=ax4, color=["green","red"], title="Residuals Distribution", grid=True)
            ax4.set(ylabel=None)
            plt.show()
            print("Training --> Residuals mean:", np.round(residuals_mean), " | std:", np.round(residuals_std))
            print("Test --> Error mean:", np.round(error_mean), " | std:", np.round(error_std),
                  " | mae:",np.round(mae), " | mape:",np.round(mape*100), "%  | mse:",np.round(mse), " | rmse:",np.round(rmse))
        
        return dtf[[f,"model","residuals","conf_int_low","conf_int_up", 
                    "forecast","error","pred_int_low","pred_int_up"]]
    
   

#fonction pour la modelisation avec ARIMAX
#%%#
def fit_sarimax(df_train, df_test,f, order=(1,0,1), 
                seasonal_order=(0,0,0,0), exog_train=None, 
                exog_test=None, figsize=(15,10)):
    ## train
    model = sm.SARIMAX(df_train, order=order, 
                        seasonal_order=seasonal_order, 
                        exog=exog_train, enforce_stationarity=False, 
                        enforce_invertibility=False).fit()
    #dtf_train = df_train.to_frame(name="ts")
    df_train["model"] = model.fittedvalues
    
    ## test
    #dtf_test = df_test.to_frame(name="ts")
    df_test["forecast"] = model.predict(start=len(df_train), 
                            end=len(df_train)+len(df_test)-1, 
                            exog=exog_test)
    
    ## evaluate
    dtf = df_train.append(df_test)
    title = "ARIMA "+str(order) if exog_train is None else "ARIMAX "+str(order)
    title = "S"+title+" x "+str(seasonal_order) if np.sum(seasonal_order) > 0 else title
    dtf = utils_evaluate_forecast(dtf,f, figsize=figsize, title=title)
    return dtf, model

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
df_deces, model_deces = fit_sarimax(df_train_deces, df_test_deces,f='décès', order=(3, 2, 2), 
                         seasonal_order=(1,0,1,7))

df_cas, model_cas = fit_sarimax(df_train_cas, df_test_cas,f='nouveaux_cas', order=(3, 2, 2), 
                         seasonal_order=(1,0,1,7))

#%%
