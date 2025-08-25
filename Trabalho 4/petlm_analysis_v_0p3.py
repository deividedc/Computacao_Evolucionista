#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import pandas as pd
import math
import matplotlib.pyplot as pl
import scipy as sp
import glob
import seaborn as sns
import re
import os, sys
import itertools
import hydroeval as he
from scipy import stats
import pickle 
from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error

sys.path.append(os.getcwd())
from util.metrics import rrmse, agreementindex,  lognashsutcliffe,  nashsutcliffe

import skill_metrics as sm

#%%-----------------------------------------------------------------------------
pd.options.display.float_format = '{:.3f}'.format
palette_color="Set1"#"Blues_r"

def fmt(x): 
    if (type(x) == str or type(x) == tuple or type(x) == list):
        return str(x)
    else:
      if (abs(x)>0.001 and abs(x)<1e0):
        return '%1.3f' % x   
      else:
        return '%1.3f' % x #return '%1.3f' % x
  
def fstat(x):
  #m,s= '{:1.4g}'.format(np.mean(x)), '{:1.4g}'.format(np.std(x))
  #m,s, md= fmt(np.mean(x)), fmt(np.std(x)), fmt(np.median(x)) 
  m,s, md= np.mean(x), np.std(x), np.median(x) 
  #text=str(m)+'$\pm$'+str(s)
  s = '--' if s<1e-8 else s
  text=fmt(m)+' ('+fmt(s)+')'#+' ['+str(md)+']'
  return text
  
def mean_percentual_error(y_true, y_pred):    
  y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
  return np.mean(np.abs(y_pred - y_true)/np.abs(y_true))*100

def VAF(y_true, y_pred):    
  y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
  return ( 1 - np.var(y_true - y_pred)/np.var(y_true) )*100

def accuracy_log(y_true, y_pred):
    y_true=np.abs(np.array(y_true))
    y_pred=np.abs(np.array(y_pred))
    return (np.abs(np.log10(y_true/y_pred))<0.3).sum()/len(y_true)*100
    

# http://www.jesshamrick.com/2016/04/13/reproducible-plots/
def set_style():
    # This sets reasonable defaults for size for
    # a figure that will go in a paper
    sns.set_context("paper")
    #pl.style.use(['seaborn-white', 'seaborn-paper'])
    #matplotlib.rc("font", family="Times New Roman")
    #(_palette("Greys", 1, 0.99, )
    #sns.set_palette("Blues_r", 1, 0.99, )
    sns.set_palette(palette_color, )
    sns.set_context("paper", font_scale=1.8, 
        rc={"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
            'xtick.labelsize':16,'ytick.labelsize':16,
            'font.family':"Times New Roman", }
        ) 
    # Set the font to be serif, rather than sans
    #sns.set(font='serif', font_scale=1.4,)
    
    # Make the background white, and specify the
    # specific font family
    sns.set_style(style="white", rc={
        #"font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })
    
    #os.system('rm -rf ~/.cache/matplotlib/tex.cache/')
    pl.rc('text', usetex=True)
    #pl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    pl.rc('font', family='serif',  serif='Times')

#sns.set(style="ticks", palette="Set1", color_codes=True, font_scale=1.4,)
#%%-----------------------------------------------------------------------------
#fn='./data/data_ldc_vijay/sahay_2011.csv'
#A = pd.read_csv(fn, delimiter=';')
#B = A.drop(labels=['Number', 'Stream', 'Observed',], axis=1)
#y_test = A[['Observed']].values
#for c in B:
#    y_pred=B[[c]].values
#    
#    rmse, r2 = mean_squared_error(y_test, y_pred)**.5, r2_score(y_test, y_pred)
#    r=stats.pearsonr(y_test.ravel(), y_pred.ravel())[0]
#    acc=accuracy_log(y_test.ravel(), y_pred.ravel())
#    #rmslkx=rmse_lower(y_test.ravel(), y_pred.ravel(), typ='Kx')
#    #rmslq=rmse_lower(y_test.ravel(), y_pred.ravel(), typ='Q')
#    print("%12s \t %8.2f %8.2f %8.2f %8.2f" % (c,rmse, acc, r2,r))
    
#%%-----------------------------------------------------------------------------
    
set_style()    
    
basename='eml__'


#from pandas.compat.pickle_compat import _class_locations_map
#
#_class_locations_map.update({
#    ('pandas.core.internals.managers', 'BlockManager'): ('pandas.core.internals', 'BlockManager')
#}) 

# path='./pkl_irati*'
pkl_list  = []
# for (k,p) in enumerate(glob.glob(path)):
#     pkl_list += glob.glob(p+'/'+'*.pkl')

#pkl_list   += glob.glob('./pkl_yudongnan/naula*.pkl')

#pkl_list   += glob.glob('./pkl_petlm*/*.pkl')

#pkl_list   += glob.glob('./old_pkl_petlm*/*.pkl')
pkl_list   += glob.glob('./pkl_petlm*/*.pkl')

#pkl_list   += glob.glob('./pkl_yudongnan/3fold*.pkl')
#pkl_list   += glob.glob('./pkl_yudongnan/5fold*.pkl')
#pkl_list   += glob.glob('./pkl_yudongnan/10fold*.pkl')

#
pkl_list.sort()
#
# leitura dos dados
#
A=[]
for pkl in pkl_list:
    #print(pkl)
    df = pd.read_pickle(pkl)       
    A.append(df)
    #with open(pkl, 'rb') as f:
    #    x = pickle.load(f)
#
A = pd.concat(A, sort=False)

#%%
# remove/collect information

#A=A[A['ALGO']=='CMA-ES: Covariance Matrix Adaptation Evolutionary Strategy']
#A=A[A['RUN']<=70]
#A=A[A['RUN']>0]


#%%

models_to_remove = [
                    #'MARS', 
                    #'XGB', 'RF', 'AB', 'RBFNN', 
                    #'GPR', 
                    #'SVR', 
                    #'ANN','MLP',
                    #'GPR-FS', 'SVR-FS', 
                    #'XGB-FS'
                    #'SVM', 'SVR-L', 'LSSVR'
                    ]
#models_to_remove = []
for m in models_to_remove:
    A = A[A['EST_NAME'] != m]    

A['ALGO'] = [i.split(':')[0] for i in A['ALGO']]

algorithms_to_remove = [
                    #'xNES',                    
                    ]
for m in algorithms_to_remove:
    A = A[A['ALGO'] != m]    


#datasets_to_remove = ['LDC case 8', 'LDC case 9',]
datasets_to_remove = []
#datasets_to_remove = ['LDC case '+str(i) for i in range(8)]
for m in datasets_to_remove:
    A = A[A['DATASET_NAME'] != m]    

# Deixar comentadas as linhas abaixo
#if A['DATASET_NAME'].unique()[0] == 'Energy Efficiency':
#    A['DATASET_NAME'] = A['OUTPUT']; A['OUTPUT']='Load'         
#A['DATASET_NAME'] = 'Irati'

A['DATASET_NAME'] = [x.split('-')[0] for x in A['DATASET_NAME']]
#%%
steps=['TRAIN', 'TEST'] if 'Y_TEST_PRED' in A.columns else ['TRAIN']

C = []
for step in steps:
    for k in range(len(A)):
        df=A.iloc[k]
        y_true = pd.DataFrame(df['Y_'+step+'_TRUE'], columns=[df['OUTPUT']])#['0'])
        y_pred = pd.DataFrame(df['Y_'+step+'_PRED'], columns=[df['OUTPUT']])#['0'])
        #print (k, df['EST_PARAMS'])
        
        run = df['RUN']
        av = df['ACTIVE_VAR']
        ds_name = df['DATASET_NAME']
        s0 = ''.join([str(i) for i in av])
        s1 = ' '.join(['x_'+str(i) for i in av])
        s2 = '|'.join(['$x_'+str(i)+'$' for i in av])
        var_names = y_true.columns
        
        df['EST_PARAMS']['scaler']=df['SCALER']
        
        if len(y_true)>0:
            for v in var_names:
                _mape    = abs((y_true[v] - y_pred[v])/y_true[v]).mean()*100
                _vaf     = VAF(y_true[v], y_pred[v])
                _r2      = r2_score(y_true[v], y_pred[v])
                _mae     = mean_absolute_error(y_true[v], y_pred[v])
                _mse     = mean_squared_error(y_true[v], y_pred[v])
                _rrmse   = rrmse(y_true[v], y_pred[v])
                _wi      = agreementindex(y_true[v], y_pred[v])
                _r       = stats.pearsonr(y_true[v], y_pred[v])[0]
                #_nse     = he.nse(y_true.values, y_pred.values)[0]
                _nse     = nashsutcliffe(y_true.values, y_pred.values)
                #_lnse    = lognashsutcliffe(y_true.values, y_pred.values)
                _rmse    = he.rmse(y_true.values, y_pred.values)[0]
                #_rmsekx  = rmse_lower(y_true.values, y_pred.values, 'Kx')
                #_rmseq   = rmse_lower(y_true.values, y_pred.values, 'Q')
                _kge     = he.kge(y_true.values, y_pred.values)[0][0]
                _mare    = he.mare(y_true.values, y_pred.values)[0]
                dic     = {'Run':run, 'Output':v, 'MAPE':_mape, 'R$^2$':_r2, 'MSE':_mse,
                           'Active Features':s2, 'Seed':df['SEED'], 
                          'Dataset':ds_name, 'Phase':step, '':None,
                          'NSE': _nse, 'MARE': _mare, 'MAE': _mae, 'VAF': _vaf, 
                          'Active Variables': ', '.join(df['ACTIVE_VAR_NAMES']),
                          #'RMSELKX':rmsekx, 'RMSELQ':rmseq, 
                          'Scaler': df['SCALER'], 'KGE': _kge,
                          'RMSE':_rmse, 'R':_r, 'Parameters':df['EST_PARAMS'],
                          'NDEI':_rmse/np.std(y_true.values),
                          'WI':_wi, 'RRMSE':_rrmse,
                          'y_true':y_true.values.ravel(), 
                          'y_pred':y_pred.values.ravel(),
                          'n_folds':df['N_SPLITS'],
                           #'Optimizer':df['ALGO'].split(':')[0], #A['ALGO'].iloc[0].split(':')[0],
                          'Optimizer':df['ALGO'], #A['ALGO'].iloc[0].split(':')[0],
                          #'Accuracy':accuracy_log(y_true.values.ravel(), y_pred.values.ravel()),
                          'Estimator':df['EST_NAME']}
                C.append(dic)
    
#        if step=='TEST':
#            pl.plot(y_true.values,y_true.values,'r-',y_true.values, y_pred.values, 'b.', )
#            t=ds_name+' - '+df['EST_NAME']+': '+step+': '+str(fmt(r2))
#            pl.title(t)
#            pl.show()


#            
#df              = pd.read_csv('./references/reference_tayfur.csv', delimiter=';')
#ref_estimator   = df['Estimator'].unique()[0]
#df['Run']       = 30
#
#for i in range(len(df)):
#    aux = dic.copy()
#    for c in df:
#        aux[c] =  df.iloc[i][c]
#    
#    C.append(aux)
        
C = pd.DataFrame(C)
C = C.reindex(sorted(C.columns), axis=1)

#C[C['Run']>=20]
#C['Output']='$K_x$(m$^2$/s)'

#C=C[C['Run']> 29]
#C=C[C['Run']>=6]; C=C[C['Run']< 36  ]
#C=C[C['Run']>=12]; C=C[C['Run']< 42]
#C['Dataset'] = [i.replace('Naula model','Case') for i in C['Dataset']]
#C['Output']='$Q_t$'


#C=C[C['Optimizer']=='SGA']
#C=C[C['Optimizer']=='PSO']
#C=C[C['Optimizer']=='DE']
#%%
print(C.groupby(['Optimizer', 'Estimator']).agg(len))

#%%
metrics=[
        'R', 
        # 'WI',
        'R$^2$', 
        # 'RRMSE',
        #'RMSELKX', 'RMSELQ', 
        #'RMSE$(K_x<100)$', 'RMSE$(B/H<50)$', 
        'RMSE', 
        # 'NDEI', 
        'MAE', #'Accuracy', 
        'MAPE',
        # 'NSE', #'LNSE', 
        # 'KGE',
        # 'MARE', 
        'MSE',
        # 'VAF', 
        #'MAE (MJ/m$^2$)', 'R',  'RMSE (MJ/m$^2$)',
        ]
    
metrics_max =  ['NSE', 'VAF', 'R', 'Accuracy','R$^2$', 'KGE', 'WI']    
#%%
#aux=A.iloc[0]
#D1=pd.read_csv('./references/deng_2002.csv', delimiter=';')
#D2=pd.read_csv('./data/data_ldc_vijay/tayfur_2005.csv', delimiter=';')
#
#D1.sort_values(by=['Measured'], axis=0, inplace=True)
#D2.sort_values(by=['Kx(m2/s)'], axis=0, inplace=True)

#for x,df in C.groupby('Estimator'): print(x,df[['R$^2$']].plot(kind='hist', title=x))
#%%
#idx_drop = C[['Dataset','Estimator', 'Run', 'Phase', 'Output']].drop_duplicates().index
#C=C.iloc[idx_drop]

#C1 = pd.read_csv('./references/reference_zaher_elm.csv')
#C1['Estimator']=ref_estimator
#C1 = C1.reindex(sorted(C1.columns), axis=1)

#C.sort_index(axis=0, level=['Dataset','Estimator'], inplace=True)
#C=C.append(C1,)# sort=True)
#C=C.append(C1, sort=True)
#%%
#S=[]   
#for (i,j), df in C.groupby(['Dataset','Active Variables']): 
#    S.append({' Dataset':i, 'Active Variables':j})
#    print('\t',i,'\t','\t',j)

#S=pd.DataFrame(S)
#print(S)
#sns.catplot(x='Dataset', y='R$^2$', data=C, hue='Phase', kind='bar', col='Estimator')
#%%
#for (p,e,o), df in C.groupby(['Phase','Estimator', 'Output']):
# if p=='TEST':
#  #if e!= ref_estimator:  
#    print ('='*80+'\n'+' - '+e+' - '+str(o)+'\n'+'='*80+'\n')
#       
#    df1 = df.groupby(['Active Variables'])
#    
#    active_variables=pd.DataFrame()
#    for m in metrics: 
#        grp  = list(df1.groups.keys())
#        mean = df1[m].agg(np.mean).values
#        std  = df1[m].agg(np.std).values
#        v    = [fmt(i)+' ('+fmt(j)+')' for (i,j) in zip(mean,std)]
#        
#        active_variables['Set']=grp
#        active_variables[m]=v
#    
#    active_variables.sort_values(by=['Accuracy'], axis=0, inplace=True, ascending=False)    
#    
#    print(active_variables)Blues

#%%
# ANOVA test
latex_dir = 'latex'

# Cria a pasta 'latex' se não existir
os.makedirs(latex_dir, exist_ok=True)

anova=[]
for (f,e),df in C.groupby(['Phase', 'Estimator']):
    #df1 = df[df['Estimator']!=ref_estimator]
    df1=df
    for (d,o),df2 in df1.groupby(['Dataset','Output', ]):    
            if f=='TRAIN':
                print('\n'+'='*80+'\n'+str(d)+' '+str(f)+' '+str(o)+' '+str(e)+'\n'+'='*80)
                nam = 'Optimizer'
                groups=df2.groupby(nam,)
                print(df2[nam].unique())
                
                for m in metrics:
                    #-pl.figure()
                    dic={}
                    for g, dg in groups: 
                        #-h=sns.distplot(dg[m].values, label=g)
                        dic[g]=dg[m].values
                        
                    #f_, p_ = stats.f_oneway(*dic.values())
                    f_, p_ = stats.kruskal(*dic.values())
                    #-h.legend(); h.set_xlabel(m); 
                    #-h.set_title('Dataset: '+d+'\n F-statistic = '+fmt(f_)+', '+'$p$-value = '+fmt(p_));
                    #-h.set_title('Dataset: '+d+' ($p = $'+fmt(p_)+')');
                    #-pl.ylabel(m)
                    #-pl.show()     
                    anova.append({ #m:fmt(p_),
                                  'Phase':f, 
                                  'Estimator':e,
                                  'Output':o,'Metric':m, 
                                  'F-value':fmt(f_), 
                                  'p-value':fmt(p_),  
                                  'Dataset':d,
                                  },)


anova=pd.DataFrame(anova)
groups=anova.groupby(['Dataset', 'Output', 'Estimator'])
p_value_table=[]
for g, dg in groups:
    dic = dict(zip(dg['Metric'],dg['p-value']))
    dic[' Dataset'] = g[0]
    dic[' Output'] = g[1]
    dic[' Estimator'] = g[2]
    p_value_table.append(dic)

p_value_table = pd.DataFrame(p_value_table)    

fn = basename+'_cmp_estimators'+'_table.tex'
fn = re.sub('-','_', re.sub('\/','',fn)).lower()
p_value_table = p_value_table.reindex(sorted(p_value_table.columns), axis=1)
fn_path = os.path.join(latex_dir, fn)
p_value_table.to_latex(buf=fn_path, index=False)
print(p_value_table)    
#%%       
# # ANOVA test
# anova=[]
# for (f,e),df in C.groupby(['Phase', 'Optimizer']):
#     #df1 = df[df['Estimator']!=ref_estimator]
#     df1=df
#     for (d,o),df2 in df1.groupby(['Dataset','Output', ]):    
#             if f=='TRAIN':
#                 print('\n'+'='*80+'\n'+str(d)+' '+str(f)+' '+str(o)+' '+str(e)+'\n'+'='*80)
#                 nam = 'Estimator'
#                 groups=df2.groupby(nam,)
#                 print(df2[nam].unique())
                
#                 for m in metrics:
#                     #-pl.figure()
#                     dic={}
#                     for g, dg in groups: 
#                         #-h=sns.distplot(dg[m].values, label=g)
#                         dic[g]=dg[m].values
                        
#                     #f_, p_ = stats.f_oneway(*dic.values())
#                     f_, p_ = stats.kruskal(*dic.values())
#                     #-h.legend(); h.set_xlabel(m); 
#                     #-h.set_title('Dataset: '+d+'\n F-statistic = '+fmt(f_)+', '+'$p$-value = '+fmt(p_));
#                     #-h.set_title('Dataset: '+d+' ($p = $'+fmt(p_)+')');
#                     #-pl.ylabel(m)
#                     #-pl.show()     
#                     anova.append({ #m:fmt(p_),
#                                   'Phase':f, 
#                                   'Optimizer':e,
#                                   'Output':o,'Metric':m, 
#                                   'F-value':fmt(f_), 
#                                   'p-value':fmt(p_),  
#                                   'Dataset':d,
#                                   },)


# anova=pd.DataFrame(anova)
# groups=anova.groupby(['Dataset', 'Output', 'Optimizer'])
# p_value_table=[]
# for g, dg in groups:
#     dic = dict(zip(dg['Metric'],dg['p-value']))
#     dic[' Dataset'] = g[0]
#     dic[' Output'] = g[1]
#     dic[' Optimizer'] = g[2]
#     p_value_table.append(dic)

# p_value_table = pd.DataFrame(p_value_table)    

# fn = basename+'_cmp_optimizers'+'_table.tex'
# fn = re.sub('-','_', re.sub('\/','',fn)).lower()
# p_value_table = p_value_table.reindex(sorted(p_value_table.columns), axis=1)
# p_value_table.to_latex(buf=fn, index=False)
# print(p_value_table)    
#%%       


plot_base_dir = "plot"  
plot_type = "heatmap"   

# cria a pasta base e a subpasta do tipo de plot
plot_dir = os.path.join(plot_base_dir, plot_type)
os.makedirs(plot_dir, exist_ok=True)


aux=[]
for a,df1 in C.groupby(['Optimizer',]):
    for (f,d,e,o,), df in df1.groupby(['Phase', 'Dataset', 'Estimator','Output',]):
        for m in metrics:
            dic={}
            dic['Dataset']=d
            dic['Phase']=f
            dic['Optimizer']=a
            dic['Output']=o
            dic['Estimator']=e
            dic['Metric']=m
            dic['value']= df[m].quantile(0.50)            
            aux.append(dic)
            #print(a,d,f,e,o,len(df), m, df[m].mean())
        
heatmap = pd.DataFrame(aux)
heatmap = heatmap[heatmap['Phase']!='TEST']


for (f,d,o,m), df_heat in heatmap.groupby(['Phase', 'Dataset', 'Output','Metric']):
    print(f,d,o,m)
    row,col,val=['Optimizer','Estimator', 'value']
    df2=df_heat[[row,col,val]]

    n_row, n_col = len(df2[row].unique()),len(df2[col].unique())
    values = df2[val].values.reshape(n_row,n_col)
    h = pd.DataFrame(values, columns=df2[col].unique(), index=df2[row].unique())
    #h.style.background_gradient(cmap='Blues')
    #g=sns.heatmap(h.T, cmap='RdYlGn_r', linewidths=0.5, annot=True, fmt='.3g',)
    cmap_choice = 'Blues_r' if (m not in metrics_max) else 'Blues'
    #cmap_choice = 'GnBu' if (m not in metrics_max) else 'BuGn'
    pl.figure(figsize=(7,6))
    g=sns.heatmap(h.T, 
                  #cmap='YlGnBu', 
                  #cmap='tab20b', 
                  #cmap='gist_ncar', 
                  #cmap='viridis', 
                  #cmap='Blues_r', 
                  cmap=cmap_choice, robust=True,
                  linewidths=0.5, alpha=0.85, annot=True, fmt='.3f',)    
    pl.title('Averaged '+m)
    
    
    fn = basename+'300dpi_cross_table_'+str(f)+'_'+str(d)+'_'+str(o)+'_'+str(m)+'.svg'
    fn = re.sub('\^','', re.sub('\$','',fn))
    fn = re.sub('\(','', re.sub('\)','',fn))
    fn = re.sub(' ','_', re.sub('\/','',fn))
    fn = re.sub('-','_', re.sub('\/','',fn)).lower()
    fn_path = os.path.join(plot_dir, fn)
    pl.savefig(fn_path, bbox_inches='tight', dpi=1200)

    
    pl.show()
    #print(h.mean(axis=0))
#    for r in df2[row]:
#        for c in df2[col]:
#            print(r,c)
   
#%%       
aux=[]
for a,df1 in C.groupby(['Optimizer',]):
    for (f,d,e,o,), df in df1.groupby(['Phase', 'Dataset', 'Estimator','Output',]):
        #print(a,d,f,e,o,len(df))
        dic={}
        dic['Dataset']=d
        dic['Phase']=f
        dic['Optimizer']=a
        dic['Output']=o
        dic['Estimator']=e
        for f in metrics:
            dic[f]= fstat(df[f])
        
        aux.append(dic)
        
tbl = pd.DataFrame(aux)
tbl = tbl[tbl['Phase']!='TEST']
           
#%%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.width=140
df_table=[]



for (a,f,d,o), df in tbl.groupby(['Optimizer','Phase', 'Dataset','Output']):
    #for (d,f), df in tbl.groupby(['Dataset','Phase', ]):
    for m in metrics:
        x, s = [], []
        for v in df[m].values:
            x_,s_ = v.split(' ')
            x_    = float(x_)   
            x.append(x_)
            s.append(s_)
        
        x_idx     = np.argmax(x) if m in metrics_max else np.argmin(x)
        x         =[fmt(i) for i in x]
        x[x_idx]  = '{ \\bf '+x[x_idx]+'}'
        
        df[m]     = [ i+' '+j for (i,j) in zip(x,s)]
        
        
    fn = basename+'_comparison_datasets'+'_table_'+a.lower()+'_'+d.lower()+'_'+f.lower()+'.tex'
    fn = re.sub('\^','', re.sub('\$','',fn))
    fn = re.sub('\(','', re.sub('\)','',fn))
    fn = re.sub(' ','_', re.sub('\/','',fn))
    
    print('\n'+'='*80+'\n'+str(a)+' '+str(d)+' '+str(f)+'\n'+'='*80)
    print(fn)
    df['Modeling Phase']=df['Phase']
    df.drop(['Phase',], axis=1)
    #df1=df[['Modeling Phase', 'Dataset', 'Estimator', 'R', 'VAF', 'RMSE (MJ/m$^2$)', 'MAE (MJ/m$^2$)', 'NSE']]
    df1=df[['Modeling Phase', 'Dataset', 'Optimizer', 'Output', 'Estimator', ]+metrics]
    #df.drop(['Output', 'Dataset', 'Phase'], axis=1, inplace=True)
    print(df1)
    fn_path = os.path.join(latex_dir, fn)
    df1.to_latex(buf=fn_path, index=False, escape=False, label=fn, caption='', column_format='r'*df1.shape[1])
    df_table.append(df1)

df_table=pd.concat(df_table)

cpt = 'Caption to be inserted.'
fn = basename+'_comparison_datasets'+'_table'
fn = re.sub('-','_', re.sub('\/','',fn)).lower()
df_table.drop(labels=['Modeling Phase'], axis=1, inplace=True)
fn_path = os.path.join(latex_dir, fn)
df_table.to_latex(buf=fn_path+'.tex', index=False, escape=False, label=fn, caption=cpt, column_format='r'*df_table.shape[1])
#df_table.T.to_latex(buf=fn+'_transpose.tex', index=True, escape=False, label=fn, caption=cpt, column_format='r'*df_table.shape[1])
#print(df_table)

os.system('cp '+fn+'.tex'+' ./latex/tables/')

#%%    
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.width=140
df_table=[]

for (a,f,d,o), df in tbl.groupby(['Estimator','Phase', 'Dataset','Output']):
    #for (d,f), df in tbl.groupby(['Dataset','Phase', ]):
    for m in metrics:
        x, s = [], []
        for v in df[m].values:
            x_,s_ = v.split(' ')
            x_    = float(x_)   
            x.append(x_)
            s.append(s_)
        
        x_idx     = np.argmax(x) if m in metrics_max else np.argmin(x)
        x         =[fmt(i) for i in x]
        x[x_idx]  = '{ \\bf '+x[x_idx]+'}'
        
        df[m]     = [ i+' '+j for (i,j) in zip(x,s)]
        
        
    fn = basename+'_comparison_datasets'+'_table_'+a.lower()+'_'+d.lower()+'_'+f.lower()+'.tex'
    fn = re.sub('\^','', re.sub('\$','',fn))
    fn = re.sub('\(','', re.sub('\)','',fn))
    fn = re.sub(' ','_', re.sub('\/','',fn))
    
    print('\n'+'='*80+'\n'+str(a)+' '+str(d)+' '+str(f)+'\n'+'='*80)
    print(fn)
    df['Modeling Phase']=df['Phase']
    df.drop(['Phase',], axis=1)
    #df1=df[['Modeling Phase', 'Dataset', 'Estimator', 'R', 'VAF', 'RMSE (MJ/m$^2$)', 'MAE (MJ/m$^2$)', 'NSE']]
    df1=df[['Modeling Phase', 'Dataset', 'Optimizer', 'Output', 'Estimator', ]+metrics]
    #df.drop(['Output', 'Dataset', 'Phase'], axis=1, inplace=True)
    print(df1)
    fn_path = os.path.join(latex_dir, fn)
    df1.to_latex(buf=fn_path, index=False, escape=False, label=fn, caption='', column_format='r'*df1.shape[1])
    df_table.append(df1)

df_table=pd.concat(df_table)

cpt = 'Caption to be inserted.'
fn = basename+'_comparison_datasets'+'_table'
fn = re.sub('-','_', re.sub('\/','',fn)).lower()
df_table.drop(labels=['Modeling Phase'], axis=1, inplace=True)
fn_path = os.path.join(latex_dir, fn)
df_table.to_latex(buf=fn_path+'.tex', index=False, escape=False, label=fn, caption=cpt, column_format='r'*df_table.shape[1])
#df_table.T.to_latex(buf=fn+'_transpose.tex', index=True, escape=False, label=fn, caption=cpt, column_format='r'*df_table.shape[1])
#print(df_table)

os.system('cp '+fn+'.tex'+' ./latex/tables/')

#%%    
# radarchart

plot_type = "radarchart"   

# cria a pasta base e a subpasta do tipo de plot
plot_dir = os.path.join(plot_base_dir, plot_type)
os.makedirs(plot_dir, exist_ok=True)

for (f,a,d,o,), df in C.groupby(['Phase', 'Optimizer', 'Dataset', 'Output',]):
    if f!='TEST':
        #print(df[metrics].columns)
        print(f,a,d,o)
        df_estimator=df.groupby(['Estimator'])[metrics].agg(np.mean)
        df_estimator.index=df_estimator.index.values

        categories=df_estimator.columns
        N = len(categories)
        
        for i in df_estimator.columns: 
            if i in metrics_max:
                df_estimator[i]=df_estimator[i]/df_estimator[i].max()*100
            else:
                df_estimator[i]=df_estimator[i].min()/df_estimator[i]*100


        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

    
        # initialise the spider plot
        pl.figure(figsize=(5,5), )#dpi=72)
        
        ax = pl.subplot(111, polar=True)
        
        # if you want the first axis to be on top:
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw one axe per variable + add labels labels yet
        pl.xticks(angles[:-1], categories, color='k')
    
        # Draw ylabels
        ax.set_rlabel_position(0)
        pl.ylim(0,108)
        ax.tick_params(axis='y', colors='grey',  grid_linestyle='--', size=7)
        ax.tick_params(axis='x', colors='grey',  grid_linestyle='--', size=7)
        
        for i in range(len(df_estimator)):
            values=df_estimator.iloc[i].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', marker='o', label=df_estimator.index[i])
            ax.fill(angles, values,alpha=0.02)# 'w', alpha=0)

        # Add legend
        #pl.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        #pl.title(d, y=1.1,)
        #pl.legend(loc=0, bbox_to_anchor=(1.12, 0.7), title=r"\textbf{"+d+"}", fancybox=True)
        pl.legend(loc=0, bbox_to_anchor=(1.12, 1.0), 
                  #title=r"{"+d+' - '+o+"}", 
                  title=r"{"+a+"}", 
                  fancybox=True)

        fn = basename+'300dpi_radarchart_'+str(f)+'_'+str(d)+'_'+str(o)+'_'+str(a)+'.svg'
        fn = re.sub('\^','', re.sub('\$','',fn))
        fn = re.sub('\(','', re.sub('\)','',fn))
        fn = re.sub(' ','_', re.sub('\/','',fn))
        fn = re.sub('-','_', re.sub('\/','',fn)).lower()
        fn_path = os.path.join(plot_dir, fn)
        pl.savefig(fn_path,  bbox_inches='tight', dpi=1200)

        pl.show()
        
#sys.exit()
#%%    

plot_type = "boxplot"   

# cria a pasta base e a subpasta do tipo de plot
plot_dir = os.path.join(plot_base_dir, plot_type)
os.makedirs(plot_dir, exist_ok=True)

#for(d,opt),df in C.groupby(['Dataset', 'Optimizer']):
for d,df in C.groupby(['Dataset', ]): 
    # https://github.com/pog87/PtitPrince/blob/master/RainCloud_Plot.ipynb
    n_estimators = df['Estimator'].unique().shape[0]
    ds=df['Output'].unique(); ds.sort()
    hs=df['Estimator'].unique(); hs.sort(); #hs=np.concatenate([hs[hs!=ref_estimator],hs[hs==ref_estimator]])
    #sns.set(font_scale=2.5)
    for kind in [ 'box',]:# 'violin']:
    #for kind in ['bar', 'box', 'violin']:
        for m in metrics:
            kwargs={'edgecolor':"grey", 'capsize':0.05, 'alpha':0.95, 'ci':'sd', 'errwidth':1.0, 'dodge':True, 'aspect':2.4, 'legend':None, } if kind=='bar' else {'notch':0, 'ci':'sd','aspect':2,}
    #        sns.catplot(x='Dataset', y='R$^2$', data=C, hue='Phase', kind='bar', col='Estimator')
    #        g=sns.catplot(x='Dataset', y=m, col='Estimator', data=C, 
    #                       kind=kind, sharey=False, hue='Phase', 
    #                       **kwargs,);
    #        g=sns.catplot(col='Dataset', y=m, hue='Estimator', data=C, 
    #                       kind=kind, sharey=False, x='Phase', 
    #                       **kwargs,);
            if kind=='bar':
                g=sns.catplot(x='Optimizer', y=m, hue='Estimator', #row='Phase', 
                              data=df[df['Phase']!='TEST'], 
                              #order=ds, hue_order=hs, 
                              kind=kind, 
                              #sharey=False,  
                              #col_wrap=2, palette=palette_color_1,
                              **kwargs,)            
            elif kind=='box':
                g=sns.catplot(col='Optimizer', y=m, hue='Estimator', x='Phase', 
                              #order=ds, hue_order=hs,
                             data=df[df['Phase']!='TEST'],  legend=False,
                              kind=kind, sharey=True, col_wrap=2,  
                               #order=ds, 
                               hue_order=hs,
                              **kwargs,)
            elif kind=='violin':
                g=sns.catplot(x='Optimizer', y=m, hue='Estimator', col='Phase', 
                              data=df[df['Phase']!='TEST'], 
                              #order=ds, hue_order=hs,
                              scale="count", #inner="quartile", 
                              count=0, legend=False,
                              kind=kind, sharey=False,  #col_wrap=2,                       
                              **kwargs,)
            else:
                pass
            
            #g.despine(left=True)
            fmtx='%2.3f'
            for ax in g.axes.ravel():
                ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=00,)# fontdict={'fontsize':17})
                if kind=='bar':
                    ax.set_ylim([0, 1.15*ax.get_ylim()[1]])
                    ax.set_xlabel(None); #ax.set_ylabel(m);
                    _h=[]
                    for p in ax.patches:
                        _h.append(p.get_height())
                     
                    _h=np.array(_h)
                    _h=_h[~np.isnan(_h)]
                    _h_max = np.max(_h)
                    for p in ax.patches:
                        _h= 0 if np.isnan(p.get_height()) else p.get_height()
                        p.set_height(_h)                
                        ax.text(
                                x=p.get_x() + p.get_width()/2., 
                                #y=1.04*p.get_height(), 
                                y=0.02*_h_max+p.get_height(), 
                                s=fmtx % p.get_height() if p.get_height()>0 else None, 
                                #fontsize=16, 
                                color='black', ha='center', 
                                va='bottom', rotation=90, weight='bold',
                                )
                pl.legend(bbox_to_anchor=(1.20, 0.5), loc=10, borderaxespad=0.,
                          ncol=1, fontsize=14, 
                          #title=r"{"+d+' - '+o+"}", 
                          #title=r"{"+opt+' - '+o+"}", 
                          fancybox=True ) 
                #pl.legend(loc=0, ncol=n_estimators, borderaxespad=0.,)

            #fn = basename+'300dpi_comparison_'+opt+'_'+d+'_metric_'+m.lower()+'_'+kind+'.svg'
            fn = basename+'300dpi_cmp_'+'_'+d+'_metric_'+m.lower()+'_'+kind+'.svg'
            fn = re.sub('\^','', re.sub('\$','',fn))
            fn = re.sub('\(','', re.sub('\)','',fn))
            fn = re.sub(' ','_', re.sub('\/','',fn))
            fn = re.sub('-','_', re.sub('\<','_',fn)).lower()
            print(fn)
            fn_path = os.path.join(plot_dir, fn)
            pl.savefig(fn_path,  bbox_inches='tight', dpi=1200)
                    
            pl.show()

#sys.exit()
#%%
def replace_names(s):
    sv = [
            ('gamma', '$\gamma$'), ('l2_penalty','$C_2$'),
            ('squared_epsilon_insensitive','$L_2$'),
            ('epsilon_insensitive','$L_1$'),
            ('epsilon','$\\varepsilon$'), ('C', '$C$'),
            ('l1_ratio','$L_1$ ratio'), ('alpha','$\\alpha$'),            
            ('thin_plate','T. Plate'),('cubic','Cubic'),
            ('inverse','Inverse'),('quintic','Quintic'),('linear','Linear'),
            ('penalty','$\gamma$'),('max_degree','$q$'),
            ('hidden_layer_sizes', 'HL'),
            ('learning_rate_init', 'LR'),
            ('rbf_width', '$\gamma$'), 
            ('activation_func', '$G$'),
            ('activation', '$\\varphi$'),
            ('n_hidden', 'HL'),
            ('number_of_neurons', 'NNHL'),
            ('sigmoid', 'Sigmoid'),
            ('inv_multiquadric', 'Inv. Multiquadric'),
            ('multiquadric', 'Multiquadric'),
            ('hardlim', 'HardLim'),('softlim', 'SoftLim'),
            ('tanh', 'Hyp. Tangent'),
            ('gaussian', 'Gaussian'),
            ('identity', 'Identity'),
            ('swish', 'Swish'),
            ('relu', 'ReLU'),
            ('logistic', 'Logistic'),
            ('Kappa', '$\kappa$'),
            ('criterion','Criterion'),
            ('learning_rate','LR'),
            ('friedman_mse','MSE'),
            ('reg_lambda','$\lambda$'),
            ('max_depth','Max. Depth'),
            ('min_samples_leaf','Min. Samples Leaf'),
            ('min_samples_split','Min. Samples Split'),
            ('min_weight_fraction_leaf', 'Min. Weig. Fract. Leaf'),
            ('n_estimators', 'No Estimators'),
            ('presort', 'Presort'),
            ('subsample', 'Subsample'),
            ('n_neighbors','$K$'),
            ('positive','Positive Weights'),
            ('max_terms','Max. Terms'),
            ('max_iter','Max. Iter.'),
            ('min_child_weight','Min. Child Weight'),
            ('colsample_bytree','Col. Sample'),
            ('thin_plate', 'thin-plate'),            
            ('interaction_only','Interaction Only'), 
            ('k1','$k_0$'),
            ('sigma', '$\sigma$'), ('beta', '$\\beta$'),
            ('U/u*','$U/u^*$'), 
            ('B','$B$'),('H','$H$'),('U','$U$'),('u*','$u^*$'),
        ]  
    for s1,s2 in sv:
        r=s.replace(str(s1), s2)
        if(r!=s):
            #print r           
            return r
    return r    
        
#%%
#for (p,d,e,o), df in C.groupby(['Phase','Dataset','Estimator','Output']):
#    print (p+'\t'+d+'\t\t'+e+'\t'+str(len(df)))

#%%
parameters=pd.DataFrame()
for (p,d,a,e,o), df in C.groupby(['Phase','Dataset','Optimizer','Estimator','Output']):
  #if e!= ref_estimator:  
    print (p+'\t'+a+'\t'+d+'\t\t'+e+'\t'+str(len(df)))
    aux={}
    par = pd.DataFrame(list(df['Parameters']))

    if e=='RBFNN':
        #par['hidden_layer_sizes']=[len(j) for j in par['hidden_layer_sizes']]
        _t=['func',]
        for t in _t:
            par[t] = [replace_names(i) for i in par[t].values]
            
        #print(par); print('\n\n\n\n\n')

    if e=='ANN':
        par['hidden_layer_sizes']=[len(j) for j in par['hidden_layer_sizes']]
        _t=['activation',]
        for t in _t:
            par[t] = [replace_names(i) for i in par[t].values]
    
    if e=='MLP':
        par['number_of_neurons']=[j[0] for j in par['hidden_layer_sizes']]
        #par['no_layers']=[len(j) for j in par['hidden_layer_sizes']]
        par['hidden_layer_sizes']=[len(j) for j in par['hidden_layer_sizes']]
        __par=par
        #_t=['activation',]
        #for t in _t:
        #    par[t] = [replace_names(i) for i in par[t].values]
    
    if  e=='ELM':
        par.drop(['regressor'], axis=1, inplace=True)
        _t=['activation_func',]
        for t in _t:
            par[t] = [replace_names(i) for i in par[t].values]

    if  e=='SVR' or e=='SVR-L':
        #par['gamma'] = [0 if a=='scale' else a for a in par['gamma']]

        _t=['loss',]
        for t in _t:
            par[t] = [replace_names(i) for i in par[t].values]
        
        par__=par

    if  e=='XGB':
        #par.drop(['objective'], axis=1, inplace=True)
        print(par)

    #if  e=='SVR' or e=='SVR-FS' or e=='SVR-L':
    #    par['gamma'] = [0 if a=='scale' else a for a in par['gamma']]
    #    print(par)
        #sys.exit()

    if  e=='GPR' or e=='GPR-FS':
        par['$\\nu$'] = [float(str(a).split('nu=')[1].split(')')[0]) for a in par['kernel']]
        par['$l$'] = [float(str(a).split('length_scale=')[1].split(', nu')[0]) for a in par['kernel']]
        par.drop(labels=['kernel'], axis=1, inplace=True)
        print(par)
        #sys.exit()

    par=par.melt()
    par['Estimator']=e
    par['Dataset']=d
    par['Phase']=p
    par['Output']=o
    par['Optimizer']=a
    par['variable'] = [replace_names(i) for i in par['variable'].values]

    parameters = parameters.append(par, sort=True)
        
parameters['Parameter']=parameters['variable']
parameters=parameters[parameters['Parameter']!='regressor'] 

#%%

#for (p,d,a,e,o), df in C.groupby(['Phase','Dataset','Optimizer','Estimator','Output']):
for (p,a,e,t,d), df in parameters.groupby(['Phase','Optimizer','Estimator', 'Parameter','Dataset']):

 if p!='TEST':
  #if e!= ref_estimator:
   #if '-FS' in e:
    print ('='*80+'\n'+t+' - '+a+' - '+e+' - '+str(d)+'\n'+'='*80+'\n')
    pl.figure()
    if df['value'].unique().shape[0]<= 6:
        df['value']=df['value'].astype(int,errors='ignore',)
        kwargs={"linewidth": 1, 'edgecolor':None,}
        g = sns.catplot(x='value', col='Output', kind='count', data=df, 
                                                col_wrap=4,
                        aspect=0.618, palette=palette_color, **kwargs)
        fmtx='%3d'
        g.set_ylabels('Frequency')#(e+': Parameter '+t)            
        g.fig.tight_layout()
        
        for ax in g.axes.ravel():
            ax.axes.set_xlabel('('+a+') '+e+': Parameter '+t)
            ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=0, fontsize=16,)                       
            ax.set_ylim([0, 1.05*ax.get_ylim()[1]])            
            ylabels = ['%3d'% x for x  in ax.get_yticks()]
            ax.set_yticklabels(ylabels,)# fontsize=16,)
            #ax.set_xlabel(e+': '+t, )#fontsize=16,)

            #ax.set_xlabel('Day'); #ax.set_ylabel(m);

        for ax in g.axes.ravel():
            _h=[]
            for pat in ax.patches:
                _h.append(pat.get_height())
             
            _h=np.array(_h)
            _h=_h[~np.isnan(_h)]
            _h_max = np.max(_h)
            for pat in ax.patches:
                _h= 0 if np.isnan(pat.get_height()) else pat.get_height()
                pat.set_height(_h)
                ax.text(
                        x=pat.get_x() + pat.get_width()/2., 
                        #y=1.04*p.get_height(), 
                        y=0.05*_h_max+pat.get_height(), 
                        s=fmtx % pat.get_height(), 
                        #fontsize=16, 
                        color='black', ha='center', 
                        va='bottom', rotation=0, weight='bold',
                       )
        #pl.legend( loc=10, borderaxespad=0., fontsize=16, ) 
        #pl.show()
    else:
        df['value']=df['value'].astype(float,errors='ignore',)    
        kwargs={"linewidth": 1, 'aspect':0.4618,  'notch':1}
        #g = sns.catplot(x='value', y='Output', kind='box', data=df,  orient='h', palette=palette_color, **kwargs, )
        #xmin, xmax = g.ax.get_xlim()
        #g.ax.set_xlim(left=0, right=xmax)
        g = sns.catplot(y='value', x='Output', kind='box', data=df,  orient='v', palette=palette_color, **kwargs, )
        #g.ax.set_xlabel(d+' -- '+e+': Parameter '+t, fontsize=16,)
        #g.ax.set_xlabel(e+': Parameter '+t, )#fontsize=16,)
        g.ax.axes.set_xlabel('('+a+') '+e+': Parameter '+t)

        g.ax.set_ylabel(d, rotation=90)
        g.ax.set_ylabel(None)#fontsize=16,)
        g.fig.tight_layout()
        #g.fig.set_figheight(4.00)
        #pl.xticks(rotation=45)
        #g.ax.set_ylabel(e+': Parameter '+t)
        
#    min, xmax = g.ax.get_xlim()
#    g.ax.set_xlim(left=0, right=xmax)
#    g.fig.tight_layout()
#    g.fig.set_figheight(0.50)
#    pl.xticks(rotation=45)
    fn = basename+'300dpi_comparison_datasets'+'_parameters_'+'__'+a+'__'+e+'__'+t+'__'+p+'.svg'
    fn = re.sub('\^','', re.sub('\$','',fn))
    fn = re.sub('\(','', re.sub('\)','',fn))
    fn = re.sub(' ','_', re.sub('\/','',fn))
    fn = re.sub('\\\\','', re.sub('x.','x',fn))
    fn = re.sub('-','_', re.sub('\/','',fn)).lower()
    fn = fn.lower()
    #print(fn)
    pl.savefig(fn, transparent=True, optimize=True,
               bbox_inches='tight', 
               dpi=1200)
    pl.show()

#sys.exit()    
#%%

# sensitivity analysis
# https://github.com/SALib/SALib/blob/master/examples/morris/morris.py

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR, LinearSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import  XGBRegressor
from util.ELM import  ELMRegressor, ELMRegressor
#from util.MLP import MLPRegressor as MLPR
from util.RBFNN import RBFNNRegressor, RBFNN
from util.LSSVR import LSSVR
from pyearth import Earth as MARS
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

from read_data_zhao2016 import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import MaxAbsScaler

dataset=read_zhao2016()
set_style()

feature_names    = dataset['feature_names'   ]
X_train          = dataset['X_train'         ]
X_test           = dataset['X_test'          ]
y_train          = dataset['y_train'         ]
y_test           = dataset['y_test'          ]
n_features       = dataset['n_features'      ]


v_ref = 'RMSE'
v_aux = 'KGE'
k = -1
unc_tab=[]
#for (e,d,o,p,), df in C.groupby(['Estimator','Dataset','Output','Phase',]):
for (a,e,d,o,p,), df in C.groupby(['Optimizer', 'Estimator','Dataset','Output','Phase',]):
 if p=='TRAIN':
  #if e!= ref_estimator:  
   #if '-FS' in e:
    #print ('='*80+'\n'+p+' - '+d+' - '+e+' - '+str(o)+'\n'+'='*80+'\n')
    
    k = df[v_ref].idxmin()
    aux = df.loc[k] 
    param = aux['Parameters'].copy()
    
    X_train_ = pd.DataFrame(data=X_train, columns=feature_names)
    X_test_  = pd.DataFrame(data=X_test, columns=feature_names)
    if len(X_test_)==0:
        X_test_ = X_train_.copy()
        
    if e=='ELM':
        _alpha = param['l2_penalty']
        param.pop('l2_penalty')
        regressor = None if _alpha<1e-4 else Ridge(alpha=_alpha,random_state=aux['Seed'])
        param['regressor']=regressor
    
    estimators={
        'SVR':SVR(),
        'ELM':ELMRegressor(random_state=aux['Seed']),
        'EN':ElasticNet(),
        'RBFNN':RBFNNRegressor(),
        'LSSVR':LSSVR(),
        'XGB':XGBRegressor(),
        'GPR':GaussianProcessRegressor(random_state=aux['Seed'], optimizer=None, normalize_y=False),
        'MARS':MARS(),
        'SVR-L':LinearSVR(),
        'RF': RandomForestRegressor(random_state=aux['Seed'],),
        'AB': AdaBoostRegressor(random_state=aux['Seed'],),
        'ANN':MLPRegressor(random_state=aux['Seed'], warm_start=True, 
                      early_stopping=True, validation_fraction=0.20,
                      learning_rate='adaptive',  solver='adam',),
        'MLP':MLPRegressor(random_state=aux['Seed'], warm_start=True, 
                      early_stopping=True, validation_fraction=0.20,
                      learning_rate='adaptive',  solver='adam',),
        }
 
    active_features = aux['Active Variables']
    active_features = [s.replace(' ','') for s in active_features.split(',')]
    
    X_train_ = X_train_[active_features].values
    X_test_  = X_test_[active_features].values   
    n_features = X_train_.shape[1]

    #scaler=MaxAbsScaler()
    #scaler.fit(X_train_)    
    #X_train_ = scaler.transform(X_train_)
    #X_test_  = scaler.transform(X_test_)
    
    
    #reg = SVR() if 'SVR' in e else GaussianProcessRegressor(optimizer=None)
    reg=estimators[e]

    for pr in ['scaler', 'k1']:        
        if pr in param.keys():
            param.pop(pr) 
    
    reg.set_params(**param)    
    reg.fit(X_train_, y_train.T.ravel())
    
    n_outcomes=250000
    data=np.random.uniform( low=X_test_.min(axis=0), high=X_test_.max(axis=0), size=(n_outcomes, X_test_.shape[1]) )
    #data=np.random.normal( loc=X_test_.mean(axis=0), scale=X_test_.std(axis=0), size=(n_outcomes, X_test_.shape[1]) )
    predict = reg.predict(data)
    median = np.median(predict)
    mad=np.abs(predict - median).mean()
    uncertainty = 100*mad/median
    print(e,d, median, mad, n_features, uncertainty/n_features, uncertainty)
    dc={'Case':d, 'Optimizer':a,
        'Model':e, 'No. features':n_features, 'Median':median, 
        'MAD':mad, 'Uncertainty':uncertainty, v_ref:aux[v_ref]}
    unc_tab.append(dc)
    

unc_tab = pd.DataFrame(unc_tab)
fn='uncertainty_table__mc'
cpt='Caption to be inserted.'
fn_path = os.path.join(latex_dir, fn)
unc_tab.to_latex(buf=fn_path+'.tex', index=False, escape=False, label=fn, 
                 caption=cpt, column_format='r'*df_table.shape[1], 
                 float_format="%.4f")
print(unc_tab)

# unc_tab=unc_tab[unc_tab['Case']!='Case 1']
unc_tab.index=[i for i in range(unc_tab.shape[0])]

#%%
#unc_tab['case']=['FS' if 'FS' in t else 'C'+s.split(' ')[1] for s,t in zip(unc_tab['Case'],unc_tab['Model'])]
# #unc_tab['case']=['FS' if 'FS' in t else s for s,t in zip(unc_tab['Case'],unc_tab['Model'])]

pl.figure(figsize=(4,4))
p1=sns.relplot(x='Uncertainty', y=v_ref, hue='Model', size='No. features', 
                style='Optimizer',
                sizes=(600, 600),size_norm=(1,len(unc_tab['No. features'].unique())),
                data=unc_tab, alpha=0.9,
                )
# for line in range(0,unc_tab.shape[0]):
#       p1.ax.text(x=unc_tab['Uncertainty'][line]+0, y=unc_tab[v_ref][line], 
#               s=fmt(unc_tab['MAD'][line]),
#               horizontalalignment='left', size='large', color='black', 
#               weight='semibold', rotation=0)
     
fn = basename+'300dpi_comparison_uncertainty_rmse'+'.svg'
fn = re.sub('\^','', re.sub('\$','',fn))
fn = re.sub('\(','', re.sub('\)','',fn))
fn = re.sub(' ','_', re.sub('\/','',fn))
fn = re.sub('\\\\','', re.sub('x.','x',fn))
fn = re.sub('-','_', re.sub('\/','',fn)).lower()
fn = fn.lower()
#print(fn)
pl.savefig(fn, transparent=True, optimize=True,
            bbox_inches='tight', 
            dpi=1200)     
pl.show()


#sys.exit()    
#%%
v_ref = 'RMSE'
v_aux = 'KGE'
k = -1
unc_tab=[]
for (a,e,d,o,p,), df in C.groupby(['Optimizer', 'Estimator','Dataset','Output','Phase',]):
  if p=='TRAIN':
  #if e!= ref_estimator:  
    #if '-FS' in e:
    #print ('='*80+'\n'+p+' - '+d+' - '+e+' - '+str(o)+'\n'+'='*80+'\n')
    
    k = df[v_ref].idxmin()
    aux = df.loc[k] 
    
    ix=aux['y_pred']>0; unc_p,unc_t = aux['y_pred'][ix], aux['y_true'][ix]
    unc_e = (np.log10(unc_p) - np.log10(unc_t))
    unc_m=unc_e.mean()
    unc_s = np.sqrt(sum((unc_e - unc_m)**2)/(len(unc_e)-1))
    pei95=fmt(10**(-unc_m-1.96*unc_s))+' to '+fmt(10**(-unc_m+1.96*unc_s))
    #print(p+' - '+d+' - '+e+' - '+str(o), fmt(unc_m), fmt(unc_s), pei95 )
    sig = '+' if unc_m > 0 else ''
    dc={'Optimizer':a, 'Model':e, 'Case':d, 'MPE':sig+fmt(unc_m), 'WUB':'$\pm$'+fmt(unc_s), 'PEI95':pei95}
    unc_tab.append(dc)


unc_tab = pd.DataFrame(unc_tab)
fn='uncertainty_table__models'
cpt='Caption to be inserted.'
fn_path = os.path.join(latex_dir, fn)
unc_tab.to_latex(buf=fn_path+'.tex', index=False, escape=False, label=fn, caption=cpt, column_format='r'*df_table.shape[1])
print(unc_tab)
    
#sys.exit()
#%%
stations = C['Dataset'].unique()
stations.sort()
colors={}
kolors=['r', 'darkgreen', 'b', 'm', 'c', 'darkslategray', 'darkorange', 'purple','y', 'olive', 'brown' ]
for i, j in zip(stations,kolors): 
    colors[i]=j

for type_plot in ['target', 'taylor', ]:
    for (d,o,p,a), df in C.groupby(['Dataset','Output','Phase','Optimizer']):
      if p=='TRAIN':
      #if e!= ref_estimator:  
        #if '-FS' in e:
        print ('='*80+'\n'+p+' - '+d+' - '+str(o)+'\n'+'='*80+'\n')
        ref=df.iloc[0]['y_true']    
        est=df['Estimator'].unique(); label=dict(zip(est,kolors[:len(est)]))
        
        k=0
        pl.figure(figsize=[7.5,7.5])
        for e, df1 in df.groupby('Estimator'):    
            overlay='on' if k>0 else 'off'
            taylor_stats=[{'sdev':np.std(ref), 'crmsd':0, 'ccoef':1,  
                           'title':a,
                            'label':'Observation', 'bias':1, 'rmsd':0}]
            
            for i in range(len(df1)):
                pred=df1.iloc[i]['y_pred']
                ts=sm.taylor_statistics(pred,ref,'data')
                taylor_stats.append({'sdev':ts['sdev'][1], 'crmsd':ts['crmsd'][1], 
                                      'ccoef':ts['ccoef'][1],  'label':e, 
                                      'bias':sm.bias(pred, ref),
                                      'rmsd':sm.rmsd(pred, ref),                                 
                                      })
    
            taylor_stats = pd.DataFrame(taylor_stats)
            if type_plot=='taylor':
                sm.taylor_diagram(taylor_stats['sdev'].values, 
                              taylor_stats['crmsd'].values, 
                              taylor_stats['ccoef'].values,
                              markercolor =kolors[k], alpha = 0.00,
                              markerSize = 16, rmsLabelFormat='0:.2f',
                              colSTD='k', colRMS='k', colCOR='k',
                              overlay = overlay, 
                              markerLabel = label)
            elif type_plot=='target':
                sm.target_diagram(taylor_stats['bias'].values, 
                              taylor_stats['crmsd'].values, 
                              taylor_stats['rmsd'].values,
                              markercolor =kolors[k], alpha = 0.0,
                              markerSize = 4, circleLineSpec = 'k--',
                              #circles = [1000, 2000, 3000],
                              overlay = overlay, markerLabel = label)
            else:
                sys.exit('Plot type '+type_plot+' uNdefined')
                
            k+=1
        
    
        fn = basename+'300dpi_'+type_plot+'_diagram'+'__'+d+'__'+p+'__'+o+'__'+a+'.svg'
        fn = re.sub('\^','', re.sub('\$','',fn))
        fn = re.sub('\(','', re.sub('\)','',fn))
        fn = re.sub(' ','_', re.sub('\/','',fn))
        fn = re.sub('\\\\','', re.sub('x.','x',fn))
        fn = re.sub('-','_', re.sub('\/','',fn)).lower()
        fn = fn.lower()
        print(fn)
        pl.savefig(fn, transparent=True, optimize=True, bbox_inches='tight', dpi=1200)        
        pl.show()
    
# #sys.exit()
#%%
stations = C['Dataset'].unique()
estimators = C['Estimator'].unique()
optimizers = C['Optimizer'].unique()
stations.sort()
colors={}
markers={}
for i, j in zip(stations,['r', 'darkgreen', 'b', 'm', 'c','y', 'olive',  'darkorange', 'brown', 'darkslategray', ]): 
    colors[i]=j
    
for i, j in zip(optimizers,['r', 'darkgreen', 'b', 'm', 'c', 'darkslategray', 'darkorange', 'purple','y', 'olive', 'brown' ]): 
    colors[i]=j
    
for i, j in zip(estimators,['o','s','h','d','+','*' ]): 
    markers[i]=j
    
v_ref = 'RMSE'
v_aux = 'MAE'
k = -1
best_models=pd.DataFrame()
for (a,e,d,o,p,), df in C.groupby(['Optimizer', 'Estimator','Dataset','Output','Phase',]):
  if p=='TRAIN':
  #if e!= ref_estimator:  
    #if '-FS' in e:
    print ('='*80+'\n'+p+' - '+d+' - '+e+' - '+str(o)+'\n'+'='*80+'\n')
    
    k = df[v_ref].idxmin()
    aux = df.loc[k] 
    pl.figure(figsize=(4,4))
    ax = sns.regplot(x="y_true", y="y_pred", data=aux, ci=0.95, 
                      line_kws={'color':'black'}, 
                      marker=markers[e],
                      scatter_kws={'alpha':0.85, 'color':colors[a], 's':100},
                      #scatter_kws={'alpha':0.85, 'color':colors[d], 's':100},
                      #label='WI'+' = '+fmt(aux['WI']),
                      #label='R'+' = '+fmt(aux['R']),
                      label='R$^2$'+' = '+fmt(aux['R$^2$']),
                      )
    #ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(left    = aux['y_true'].min(), right    = 1.1*aux['y_true'].max() )
    ax.set_ylim(bottom  = aux['y_true'].min(), top      = 1.1*aux['y_true'].max() )
    ax.set_title(d+' -- '+e+' ({\\bf '+p+'}) '+'\n'+v_ref+' = '+fmt(df[v_ref][k]))
    ax.set_title(a+' -- '+e+' ('+p+') '+'\n'+v_ref+' = '+fmt(df[v_ref][k])+', '+v_aux+' = '+fmt(df[v_aux][k]))
    ax.set_xlabel('Measured   '+aux['Output'])
    ax.set_ylabel('Predicted  '+aux['Output'])
    ax.set_yticklabels(labels=ax.get_yticks(), rotation=0)
    ax.set_xticklabels(labels=ax.get_xticks(), rotation=0)
    ax.set_aspect(1)
    ax.legend(frameon=False, markerscale=0, loc=0)
    fn = basename+'300dpi_scatter'+'_best_model_'+'__'+a+'__'+e+'__'+d+'__'+p+'.svg'
    fn = re.sub('\^','', re.sub('\$','',fn))
    fn = re.sub('\(','', re.sub('\)','',fn)) 
    fn = re.sub(' ','_', re.sub('\/','',fn))
    fn = re.sub('\\\\','', re.sub('x.','x',fn))
    fn = re.sub('-','_', re.sub('\/','',fn)).lower()
    fn = fn.lower()
    #print(fn)
    pl.savefig(fn, transparent=True, optimize=True, bbox_inches='tight', dpi=1200)
    best_models['TOC']=aux['y_true']
    best_models[e]=aux['y_pred']
    
    
    pl.show()
#%%
# # variaton plots, hydrograms
# # https://seaborn.pydata.org/generated/seaborn.lineplot.html
# stations = C['Dataset'].unique()
# stations.sort()
# colors={}
# for i, j in zip(stations,['r', 'darkgreen', 'b', 'm', 'c','y', 'olive',  'darkorange', 'brown', 'darkslategray', ]): 
#     colors[i]=j
    
# v_ref = 'RMSE'
# v_aux = 'KGE'
# k = -1

# for (d,o,p,), df1 in C.groupby(['Dataset','Output','Phase',]):
#  if p!='TRAIN':
#   #if e!= ref_estimator:  
#    #if '-FS' in e:
#     print ('='*80+'\n'+p+' - '+d+' - '+' - '+str(o)+'\n'+'='*80+'\n')
    
#     tab_best_models=pd.DataFrame()
#     for e, df in df1.groupby('Estimator'):
#         k = df[v_ref].idxmin()
#         aux = df.loc[k] 
#         xrange=pd.date_range(start="2001-01-01", periods=len(aux['y_true']), freq="m", normalize=True)
#         aux['Month']    = xrange
#         aux['Observed'] = aux['y_true']
#         aux['Predicted']= aux['y_pred']
#         print(e)
#         #tab_best_models['Case']=d
#         #tab_best_models['Month']=aux['Month']
#         tab_best_models['Observed']=aux['y_true']
#         tab_best_models[e]=aux['y_pred']
#         tab_best_models.index=aux['Month']
        
#         fn='table_best_models_'+'__'+d+'__'+p+'.csv'; fn=re.sub(' ','_',fn).lower()
#         tab_best_models.to_csv(path_or_buf=fn, sep=';', index=False)
#         tab_best_models.to_excel(fn.replace('.csv', '.xlsx'), sheet_name='Station 2 best results', index=False)
        
#         aux3=pd.DataFrame(np.c_[aux['y_true'],aux['y_pred'],], columns=['Observed', e], 
#                      index=xrange)   
        
#         pl.figure(figsize=(12,4))
#         id_var='Year'
#         aux3[id_var]=aux3.index
#         target='$Q_t$'
    
#         aux4=aux3.melt(id_vars=id_var)
#         aux4['Estimator']=aux4['variable']
#         aux4[target]=aux4['value']
#         ax=sns.lineplot(x=id_var, y=target, hue='Estimator',style='Estimator', data=aux4, markers=True, dashes=False)
#         ax.legend(frameon=True, markerscale=0, loc=2, bbox_to_anchor=(1.01, 1.0))
#         ax.set_title(d+':'+e)
#         ax.set_title(d+' -- '+e+' ('+p+') '+' -- '+v_ref+' = '+fmt(df[v_ref][k])+', '+v_aux+' = '+fmt(df[v_aux][k]))
#         ax.grid()

#         fn = basename+'300dpi_variation_'+'_best_model_'+e+'__'+d+'__'+p+'.svg'
#         fn = re.sub('\^','', re.sub('\$','',fn))
#         fn = re.sub('\(','', re.sub('\)','',fn)) 
#         fn = re.sub(' ','_', re.sub('\/','',fn))
#         fn = re.sub('\\\\','', re.sub('x.','x',fn))
#         fn = re.sub('-','_', re.sub('\/','',fn)).lower()
#         fn = fn.lower()
#         #print(fn)
#         pl.savefig(fn, transparent=False, optimize=True, bbox_inches='tight', dpi=1200)
        
#         pl.show()
# #%%
# for (d,o,p,), df1 in C.groupby(['Dataset','Output','Phase',]):
#  if p!='TRAIN':
#   #if e!= ref_estimator:  
#    #if '-FS' in e:
#     print ('='*80+'\n'+p+' - '+d+' - '+' - '+str(o)+'\n'+'='*80+'\n')
    
#     for e, df in df1.groupby('Estimator'):
#         tab_model=pd.DataFrame()
#         for k in range(len(df)):
#             aux = df.iloc[k]
#             xrange=pd.date_range(start="2001-01-01", periods=len(aux['y_true']), freq="m", normalize=True)
#             aux['Month']    = xrange
#             aux['Observed'] = aux['y_true']
#             aux['Predicted']= aux['y_pred']
#             #tab_best_models['Case']=d
#             #tab_best_models['Month']=aux['Month']
#             tab_model['Observed']=aux['y_true']
#             tab_model['Run '+str(k)]=aux['y_pred']
#             tab_model.index=aux['Month']
            
#         fn='table_model_'+'__'+d+'__'+e+'.csv'; fn=re.sub(' ','_',fn).lower()
#         #tab_model.to_csv(path_or_buf=fn, sep=';', index=False)
#         tab_model.to_excel(fn.replace('.csv', '.xlsx'), sheet_name='Station 2 best results', index=False)
        
# #%% 
#     pl.figure(figsize=(12,4))
#     id_var='Year'

#     tab_best_models[id_var]=tab_best_models.index
#     target='$Q_t$'
    
#     aux2=tab_best_models.melt(id_vars=id_var)
#     aux2['Estimator']=aux2['variable']
#     aux2[target]=aux2['value']
#     ax=sns.lineplot(x=id_var, y=target, hue='Estimator', data=aux2,)
#     #ax.set_xlim(bottom  = aux2[id_var].min(), top      = 1.0*aux2[id_var].max() )
#     #ax.set_ylim(bottom  = aux2[target].min(), top      = 1.0*aux2[target].max() )
#     ax.legend(frameon=True, markerscale=0, loc=2, bbox_to_anchor=(1.01, 1.0))
#     ax.set_title(d)
#     ax.grid()

#     fn = basename+'300dpi_variation_comparison_'+'_best_model_'+'__'+d+'__'+p+'.svg'
#     fn = re.sub('\^','', re.sub('\$','',fn))
#     fn = re.sub('\(','', re.sub('\)','',fn)) 
#     fn = re.sub(' ','_', re.sub('\/','',fn))
#     fn = re.sub('\\\\','', re.sub('x.','x',fn))
#     fn = re.sub('-','_', re.sub('\/','',fn)).lower()
#     fn = fn.lower()
#     #print(fn)
#     pl.savefig(fn, transparent=False, optimize=True, bbox_inches='tight', dpi=1200)
    
#     pl.show()
# #%%
# #for (p,e,d,o), df in C.groupby(['Phase','Estimator','Dataset','Output']):
# # if p!='TRAIN':
# #  if e!= ref_estimator:  
# #    print ('='*80+'\n'+d+' - '+e+' - '+str(o)+'\n'+'='*80+'\n')
# #    aux={}
# #    par = pd.DataFrame(list(df['Parameters']))
# #    if e=='ANN':
# #        par['Layer Sizes']=[len(j) for j in par['hidden_layer_sizes']]
# #        #g=sns.catplot(hue='activation', x='Layer Sizes', data=par, kind='count', aspect=0.618)
# #        g=sns.catplot(x='activation', hue='Layer Sizes', data=par, kind='count', aspect=0.618)
# #        for p in g.ax.patches:
# #                g.ax.annotate('{:.0f}'.format(p.get_height()),
# #                            (p.get_x()*1.0, p.get_height()+.1), fontsize=12)
# #        
# #    par.columns = [replace_names(i) for i in par.columns]
# #    if e!= 'ANN':
# #     for t in par: 
# #        print(d,e,o,t,type(par[t]), par[t].dtype)       
# #        if par[t].dtype=='float64' or par[t].dtype=='int64':
# #            #pl.figure(figsize=(1,4))
# #            g = sns.catplot(x=t, data=par, kind='box', orient='h', notch=0, )#palette='Blues_r', )# width=0.1)
# #            xmin, xmax = g.ax.get_xlim()
# #            g.ax.set_xlim(left=0, right=xmax)
# #            g.ax.set_xlabel(d+' -- '+e+': Parameter '+t)
# #            g.fig.tight_layout()
# #            g.fig.set_figheight(0.50)
# #            pl.xticks(rotation=45)
# #            #g.ax.set_title(d+' - '+e)
# #            #xlabels = ['{:,.2g}'.format(x) for x in g.ax.get_xticks()/1000]
# #            #g.set_xticklabels(xlabels)
# #            pl.show()
# #        if par[t].dtype=='int64':
# #            par[t] = [ str(i) if type(i)==list or type(i)==tuple or i==None else i for i in par[t] ]
# #            #par[t] = [replace_names(j) for j in par[t]]
# #            g = sns.catplot(x=t, data=par, kind='count', palette=palette_color, aspect=0.618)
# #            ymin, ymax = g.ax.get_ylim()
# #            g.ax.set_ylim(bottom=0, top=ymax*1.1)
# #            pl.ylabel(u'Frequency')
# #            #if t=='n_hidden' or 'activation_func':    
# #            pl.xticks(rotation=90)               
# #            for p in g.ax.patches:
# #                g.ax.annotate('{:.0f}'.format(p.get_height()),
# #                            (p.get_x()*1.0, p.get_height()+.1), fontsize=16)
# #                
# #            pl.show()
# #            
# ##        elif type(par[t].values[0])==str: 
# ##            par[t] = [ str(i) if type(i)==list or type(i)==tuple or i==None else i for i in par[t] ]
# ##            par[t] = [replace_names(j) for j in par[t]]
# ##            g = sns.catplot(x=t, data=par, kind='count', palette=palette_color, aspect=0.618)
# ##            ymin, ymax = g.ax.get_ylim()
# ##            g.ax.set_ylim(bottom=0, top=ymax*1.1)
# ##            pl.ylabel(u'Frequency')
# ##            #if t=='n_hidden' or 'activation_func':    
# ##            pl.xticks(rotation=90)               
# ##            for p in g.ax.patches:
# ##                g.ax.annotate('{:.0f}'.format(p.get_height()),
# ##                            (p.get_x()*1.0, p.get_height()+.1), fontsize=12)
# #        else:
# #            pass
# #
# #        #pl.xlabel('')
# #        #pl.title(e+''+': '+replace_names(t), fontsize=16)
# #       # pl.show()
            
# #%%
# #for (e,o), df in C.groupby(['Estimator','Output']):
# #  if e!=ref_estimator:  
# #    print ('='*80+'\n'+e+' - '+o+'\n'+'='*80+'\n')
# #    aux={}
# #    par = pd.DataFrame(list(df['Parameters']))
# #    par=par.melt()
# #    par['variable'] = [replace_names(i) for i in par['variable'].values]
# #    #print(par)     
# #    for p1, df5 in par.groupby('variable'):
# #        if type(df5['value'].values[0])!=str and type(df5['value'].values[0])!=bool:
# #            kwargs={'capsize':0.05, 'ci':'sd', 'errwidth':1, 'dodge':True, 'aspect':2.5}
# #            fig=sns.catplot(x='variable', y='value', data=df5, kind='bar', **kwargs)
# #            fmt='%1.0f' if type(df5['value'].values[0])==int else '%2.3f'
# #            #fmt='%1.0f' if p1=='HL' else fmt
# #            for ax in fig.axes.ravel():
# #                for p in ax.patches:
# #                    ax.set_ylabel(p1); ax.set_xlabel('Day')
# #                    ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=90)
# #                    ax.text(
# #                                p.get_x() + p.get_width()/3., 
# #                                1.001*p.get_height(), 
# #                                fmt % p.get_height(), 
# #                                fontsize=12, color='black', ha='center', 
# #                                va='bottom', rotation=90, #weight='bold',
# #                            )
# #        else:
# #            kwargs={'dodge':True, 'aspect':0.618}
# ##            fig=sns.catplot(data=df5,x='value', kind='count', **kwargs)   
# ##            for ax in fig.axes.ravel():
# ##                #ax.set_ylim([0, 1.06*ax.get_ylim()[1]])
# ##                #t=str(ax.get_title()); ax.set_ylabel(t)
# ##                #ax.set_title('')
# ##                s1,s2= ax.get_title().split('|')
# ##                ax.set_title(s2); ax.set_ylabel(s1) ; ax.set_xlabel('') 
# ##                for p in ax.patches:
# ##                    p.set_height( 0 if np.isnan(p.get_height()) else p.get_height() )
# ##                    ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=90)
# ##                    ax.text(
# ##                                p.get_x() + p.get_width()/2., 
# ##                                1.001*p.get_height(), 
# ##                                '%1.0f' % p.get_height(), 
# ##                                fontsize=16, color='black', ha='center', 
# ##                                va='bottom', rotation=0, #weight='bold',
# ##                            )
# #
# #
# ##        pl.xlabel('Day'); pl.ylabel(p1) 
# ##        pl.title(s)
# #        
# ##        fn = basename+'_parameter_'+str(p1)+'_estimator_'+reg.lower()+'_'+'_distribution'+'.svg'
# ##        #fig = ax.get_figure()
# ##        pl.savefig(re.sub('\\\\','',re.sub('\^','', re.sub('\$','',fn) ) ),  bbox_inches='tight', dpi=1200)
# ##
# ##        pl.show()
# #    
# #%%
# n=[]
# for a in C['Active Variables']:
#     b=a.replace(' ','').split(',')
#     b=[replace_names(i) for i in b]
#     n.append(', '.join(b))

# C['Active Variables']=n    
# #%%    
# #from itertools import combinations
# #kind='count'
# #for (p,d,e,o), df in C.groupby(['Phase','Dataset','Estimator','Output']):  
# #  if p!= 'TRAIN':  
# #    if e!= ref_estimator:
# #      if '-FS' in e:         
# #        print (p+'\t'+d+'\t\t'+e+'\t'+str(len(df)))
# #        print('Number of sets: ','for ', e, ' = ', df['Active Variables'].unique().shape)
# #        kwargs={'edgecolor':"k", 'alpha':0.95, 'dodge':True, 'aspect':1.618, 'legend':None, } if kind=='count' else None
# #        n=[]
# #        for a in df['Active Variables']:
# #            for i in a.replace(' ','').split(','):
# #                #v=i.replace(' ','')
# #                #n.append(replace_names(v))
# #                n.append(i)
# #        
# #        n=[]
# #        for a in df['Active Variables']:
# #            b=a.replace(' ','').split(',')
# #            #b=[replace_names(i) for i in b]
# #            for k in range(len(b)):
# #                for i in combinations(b,k+1):
# #                    n.append({'set':', '.join(i), 'order':k+1, 'count':1})
# #
# #        P=pd.DataFrame(data=n)
# #        Q=P.groupby(['set']).agg(np.sum)
# #        Q.sort_values(by='count', axis=0, ascending=False, inplace=True)
# #
# #        for order, dfo in P.groupby(['order']):
# #            g=sns.catplot(x='set', data=dfo, kind='count',
# #                          order = dfo['set'].value_counts().index,
# #                          aspect=3,
# #                          #**kwargs,
# #                          )
# #            g.ax.set_xticklabels(labels=g.ax.get_xticklabels(),rotation=90)
# #            g.ax.set_title(e+': Order = '+str(order))
# #            pl.show()
                    
# #%%
# kind='count'
# for (p,d,e,o), df in C.groupby(['Phase','Dataset','Estimator','Output']):  
#   if p!= 'TRAIN':  
#     #if e!= ref_estimator:
#       if '-FS' in e:
#         print (p+'\t'+d+'\t\t'+e+'\t'+str(len(df)))
#         kwargs={'edgecolor':"k", 'alpha':0.95, 'dodge':True, 'aspect':0.618, 'legend':None, } if kind=='count' else None
#         n=[]
#         for a in df['Active Variables']:
#             for i in a.replace(' ','').split(','):
#                 #v=i.replace(' ','')
#                 #n.append(replace_names(v))
#                 n.append(i)
                
#         #n = [replace_names(i) for i in n]
#         P=pd.DataFrame(data=n, columns=['Variable'])
#         P['count']=1
#         g=sns.catplot(x='Variable', data=P, kind='count',
#                       order = P['Variable'].value_counts().index,
#                       **kwargs,
#                       )
#         #g.set_xticklabels(labels=g.ax.get_xticklabels(), rotation=90)
#         g.ax.legend(title=e)#labels=[e])
#         fmtx='%d'
#         for ax in g.axes.ravel():
#             ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=90)
#             ax.set_ylabel('Count')
#             ax.set_xlabel('Feature')
#             if kind=='count':
#                 #ax.set_ylim([0, 1.21*ax.get_ylim()[1]])
#                 #ax.set_xlabel('Day'); #ax.set_ylabel(m);
#                 _h=[patch.get_height() for patch in ax.patches]
                 
#                 _h=np.array(_h)
#                 _h=_h[~np.isnan(_h)]
#                 _h_max = np.max(_h)
#                 for patch in ax.patches:
#                     _h= 0 if np.isnan(patch.get_height()) else patch.get_height()
#                     patch.set_height(_h)                
#                     ax.text(
#                             x=patch.get_x() + patch.get_width()/2., 
#                             #y=1.04*patch.get_height(), 
#                             y=0.02*_h_max+patch.get_height(), 
#                             s=fmtx % patch.get_height() if patch.get_height()>0 else None, 
#                             #fontsize=16, 
#                             color='black', ha='center', 
#                             va='bottom', rotation=90, weight='bold',
#                             )
            
#         fn = basename+'300dpi_active_features_distribution_'+e+'__'+kind+'.svg'
#         fn = re.sub('\^','', re.sub('\$','',fn))
#         fn = re.sub('\(','', re.sub('\)','',fn))
#         fn = re.sub(' ','_', re.sub('\/','',fn))
#         fn = re.sub('-','_', re.sub('\/','',fn)).lower()
#         #print(fn)
#         pl.savefig(fn,  bbox_inches='tight', dpi=1200)
                
#         pl.show()
#         #--            
# #%%
# #kind='count'
# #for (p,d,e,o), df in C.groupby(['Phase','Dataset','Estimator','Output']):  
# #  if p!= 'TRAIN':  
# #    if e!= ref_estimator:
# #      if '-FS' in e:
# #        print (p+'\t'+d+'\t\t'+e+'\t'+str(len(df)))
# #        kwargs={'edgecolor':"k", 'alpha':0.95, 'dodge':True, 'aspect':1.618, 'legend':None, } if kind=='count' else None
# #        n=[]
# #        for a in df['Active Variables']:
# #            for i in a.replace(' ','').split(','):
# #                #v=i.replace(' ','')
# #                #n.append(replace_names(v))
# #                n.append(i)
# #                
# #        #n = [replace_names(i) for i in n]
# #        P=pd.DataFrame(data=n, columns=['Variable'])
# #        P['count']=1
# #        g=sns.catplot(y='Variable', data=P, kind='count',
# #                      order = P['Variable'].value_counts().index,
# #                      **kwargs,
# #                      )
# #        #g.set_xticklabels(labels=g.ax.get_xticklabels(), rotation=90)
# #        g.ax.legend(title=e)#labels=[e])
# #        fmtx='%d'
# #        for ax in g.axes.ravel():
# #            #ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=90)
# #            ax.set_xlabel('Count')
# #            ax.set_ylabel('Feature')
# #            if kind=='count':
# #                #ax.set_xlim([0, 1.21*ax.get_xlim()[1]])
# #                #ax.set_xlabel('Day'); #ax.set_ylabel(m);
# #                _h=[patch.get_width() for patch in ax.patches]
# #                 
# #                _h=np.array(_h)
# #                _h=_h[~np.isnan(_h)]
# #                _h_max = np.max(_h)
# #                for patch in ax.patches:
# #                    __h= 0 if np.isnan(patch.get_width()) else patch.get_width()
# #                    patch.set_width(__h)                
# #                    ax.text(
# #                            y=patch.get_y() + 0.7*patch.get_height(), 
# #                            #y=1.04*patch.get_height(), 
# #                            x=_h_max*0.02+patch.get_width(), 
# #                            s=fmtx % patch.get_width() if patch.get_width()>0 else None, 
# #                            #fontsize=16, 
# #                            color='black', ha='left', 
# #                            va='bottom', rotation=0, weight='bold',
# #                            )
# #            
# #        fn = basename+'300dpi_active_features_distribution_'+e+'__'+kind+'_h.png'
# #        fn = re.sub('\^','', re.sub('\$','',fn))
# #        fn = re.sub('\(','', re.sub('\)','',fn))
# #        fn = re.sub(' ','_', re.sub('\/','',fn))
# #        fn = re.sub('-','_', re.sub('\/','',fn)).lower()
# #        #print(fn)
# #        pl.savefig(fn,  bbox_inches='tight', dpi=1200)
# #                
# #        pl.show()
#         #--            
# #%%
# kind='box'
# for (p,d,e,o), df in C.groupby(['Phase','Dataset','Estimator','Output']):  
#   if p!= 'TRAIN':  
#     #if e!= ref_estimator:
#       if '-FS' in e:
#         print (p+'\t'+d+'\t\t'+e+'\t'+str(len(df)))
#         kind='count'
#         kwargs={'edgecolor':"k", 'alpha':0.95, 'dodge':True, 'aspect':1.2, 'legend':None, } if kind=='count' else None
#         g=sns.catplot(x='Active Variables', col='Estimator', hue='Phase', data=df, 
#                                #order=ds, hue_order=hs, 
#                                kind=kind, sharey=False, 
#                                order = df['Active Variables'].value_counts().index,
#                                #col_wrap=2, palette=palette_color_1,
#                                **kwargs,
#                                )                    
#         #g.despine(left=True)
#         #g.ax.legend(title=e)#labels=[e])
#         fmtx='%d'
#         for ax in g.axes.ravel():
#             ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=90)
#             ax.set_xlabel('Active Features')
#             ax.set_ylabel('Count')
#             if kind=='count':
#                 #ax.set_ylim([0, 1.21*ax.get_ylim()[1]])
#                 #ax.set_ylim([0, 1.21*ax.get_ylim()[1]])
#                 #ax.set_xlabel('Day'); #ax.set_ylabel(m);
#                 _h=[]
#                 for patch in ax.patches:
#                     _h.append(patch.get_height())
                 
#                 _h=np.array(_h)
#                 _h=_h[~np.isnan(_h)]
#                 _h_max = np.max(_h)
#                 for patch in ax.patches:
#                     _h= 0 if np.isnan(patch.get_height()) else patch.get_height()
#                     patch.set_height(_h)                
#                     ax.text(
#                             x=patch.get_x() + patch.get_width()/2., 
#                             #y=1.04*patch.get_height(), 
#                             y=0.02*_h_max+patch.get_height(), 
#                             s=fmtx % patch.get_height() if patch.get_height()>0 else None, 
#                             #fontsize=16, 
#                             color='black', ha='center', 
#                             va='bottom', rotation=90, weight='bold',
#                             )
#             #pl.legend(bbox_to_anchor=(-0.00, 1.2), loc=10, borderaxespad=0., ncol=n_estimators, fontsize=16, ) 
#             #pl.legend(bbox_to_anchor=(0.80, 0.8), loc=10, ncol=n_estimators, borderaxespad=0.,)
            
#         fn = basename+'300dpi_active_features_sets'+'_'+e+'__'+kind+'.svg'
#         fn = re.sub('\^','', re.sub('\$','',fn))
#         fn = re.sub('\(','', re.sub('\)','',fn))
#         fn = re.sub(' ','_', re.sub('\/','',fn))
#         fn = re.sub('-','_', re.sub('\/','',fn)).lower()
#         #print(fn)
#         pl.savefig(fn,  bbox_inches='tight', dpi=1200)
                
#         pl.show()

# #%%
# kind='box'
# for (p,d,o), df in C.groupby(['Phase','Dataset','Output']):
#   #df = df[df['Estimator']!=ref_estimator]  
#   if p!= 'TRAIN':
#       if len(df)>0:
#         print (p+'\t'+d+'\t\t'+'\t'+str(len(df)))
#         kind='count'
#         kwargs={'edgecolor':"k", 'alpha':0.95, 'dodge':True, 'aspect':1.2, 'legend':None, } if kind=='count' else None
#         g=sns.catplot(x='Active Variables', col='Estimator', hue='Phase', data=df, 
#                                #order=ds, hue_order=hs, 
#                                kind=kind, sharey=False, 
#                                order = df['Active Variables'].value_counts().index,
#                                #col_wrap=2, palette=palette_color_1,
#                                **kwargs,
#                                )                    
#         #g.despine(left=True)
#         #g.ax.legend(title=e)#labels=[e])
#         fmtx='%d'        
#         for ax in g.axes.ravel():
#             xticklabels=[]
#             for xticklabel, patch in zip(ax.get_xticklabels(),ax.patches):
#                 print(patch.get_height(), xticklabel)
#                 if patch.get_height()>0:
#                     xticklabels.append(xticklabel)
            
#             ax.set_xticklabels(labels=xticklabels,rotation=90)
#             ax.set_xlabel('Active Features')
#             ax.set_ylabel('Count')
#             if kind=='count':
#                 #ax.set_ylim([0, 1.21*ax.get_ylim()[1]])
#                 #ax.set_ylim([0, 1.21*ax.get_ylim()[1]])
#                 #ax.set_xlabel('Day'); #ax.set_ylabel(m);
#                 _h=[]
#                 for patch in ax.patches:
#                     _h.append(patch.get_height())
                 
#                 _h=np.array(_h)
#                 _h=_h[~np.isnan(_h)]
#                 _h_max = np.max(_h)
#                 for patch in ax.patches:
#                     _h= 0 if np.isnan(patch.get_height()) else patch.get_height()
#                     patch.set_height(_h)                
#                     ax.text(
#                             x=patch.get_x() + patch.get_width()/2., 
#                             #y=1.04*patch.get_height(), 
#                             y=0.02*_h_max+patch.get_height(), 
#                             s=fmtx % patch.get_height() if patch.get_height()>0 else None, 
#                             #fontsize=16, 
#                             color='black', ha='center', 
#                             va='bottom', rotation=90, weight='bold',
#                             )
#             #pl.legend(bbox_to_anchor=(-0.00, 1.2), loc=10, borderaxespad=0., ncol=n_estimators, fontsize=16, ) 
#             #pl.legend(bbox_to_anchor=(0.80, 0.8), loc=10, ncol=n_estimators, borderaxespad=0.,)
#             #pl.legend()
            
#         fn = basename+'300dpi_active_features_sets'+'_'+e+'__'+kind+'.svg'
#         fn = re.sub('\^','', re.sub('\$','',fn))
#         fn = re.sub('\(','', re.sub('\)','',fn))
#         fn = re.sub(' ','_', re.sub('\/','',fn))
#         fn = re.sub('-','_', re.sub('\/','',fn)).lower()
#         #print(fn)
#         pl.savefig(fn,  bbox_inches='tight', dpi=1200)
                
#         pl.show()

# #%%
# for (p,d,e,o), df in C.groupby(['Phase','Dataset','Estimator','Output']):
#  if p!= 'TRAIN':  
#   #if e!= ref_estimator:
#       if '-FS' in e:
#         print (p+'\t'+d+'\t\t'+e+'\t'+str(len(df)))
#         #aux={}
#         #epar = pd.DataFrame(list(df['Parameters']))
#         for kind in ['bar',]:
#             for m in metrics:
#                 kwargs={'edgecolor':"k", 'capsize':0.05, 'alpha':0.95, 'ci':'sd', 'errwidth':1, 'dodge':True, 'aspect':3.0618, 'legend':None, } if kind=='bar' else {'notch':0, 'ci':'sd','aspect':1.0618,}
#         #        sns.catplot(x='Dataset', y='R$^2$', data=C, hue='Phase', kind='bar', col='Estimator')
#         #        g=sns.catplot(x='Dataset', y=m, col='Estimator', data=C, 
#         #                       kind=kind, sharey=False, hue='Phase', 
#         #                       **kwargs,);
#         #        g=sns.catplot(col='Dataset', y=m, hue='Estimator', data=C, 
#         #                       kind=kind, sharey=False, x='Phase', 
#         #                       **kwargs,);
#                 if kind=='bar':
#                     g=sns.catplot(x='Active Variables', y=m, hue='Estimator', row='Phase', data=df, 
#                                #order=ds, hue_order=hs, 
#                                kind=kind, sharey=False,  
#                                #col_wrap=2, palette=palette_color_1,
#                                **kwargs,)                    
#                 else:
#                     pass
                
#                 #g.despine(left=True)
#                 fmtx='%2.3f'
#                 for ax in g.axes.ravel():
#                     ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=90)
#                     if kind=='bar':
#                         ax.set_ylim([0, 1.21*ax.get_ylim()[1]])
#                         #ax.set_xlabel('Day'); #ax.set_ylabel(m);
#                         _h=[]
#                         for p in ax.patches:
#                             _h.append(p.get_height())
                         
#                         _h=np.array(_h)
#                         _h=_h[~np.isnan(_h)]  
#                         _h_max = np.max(_h)
#                         for p in ax.patches:
#                             _h= 0 if np.isnan(p.get_height()) else p.get_height()
#                             p.set_height(_h)                
#                             ax.text(
#                                     x=p.get_x() + p.get_width()/4., 
#                                     #y=1.04*p.get_height(), 
#                                     y=0.02*_h_max+p.get_height(), 
#                                     s=fmtx % p.get_height() if p.get_height()>0 else None, 
#                                     #fontsize=16, 
#                                     color='black', ha='center', 
#                                     va='bottom', rotation=90, weight='bold',
#                                     )
#                     #pl.legend(bbox_to_anchor=(-0.00, 1.2), loc=10, borderaxespad=0., ncol=n_estimators, fontsize=16, ) 
#                     pl.legend(bbox_to_anchor=(0.80, 1.0), loc=10, ncol=n_estimators, borderaxespad=0.,)
                    
#                 fn = basename+'300dpi_active_features'+'_metric_'+m.lower()+'_'+kind+'.svg'
#                 fn = re.sub('\^','', re.sub('\$','',fn))
#                 fn = re.sub('\(','', re.sub('\)','',fn))
#                 fn = re.sub(' ','_', re.sub('\/','',fn))
#                 fn = re.sub('-','_', re.sub('\/','',fn)).lower()
#                 #print(fn)            
#                 pl.savefig(fn,  bbox_inches='tight', dpi=1200)
                        
#                 pl.show()

#%%
# to = './'
# fr = '"/home/goliatt/Dropbox/Apps/Overleaf/[ICCSA 2021] Machine Learning Approaches for Estimating Total Organic Carbon/img/"'

# fg=[
#     'zhao_heatmap_correlation.png',
#     'eml__300dpi_radarchart_train_yudongnan_toc.png',
#     'eml___comparison_datasets_table_yudongnan_train.tex',
#     'eml__300dpi_comparison_uncertainty_rmse.png',
#     'eml__300dpi_taylor_diagram__yudongnan__train__toc.png',
#     ]
# f1=glob.glob('eml__300dpi_comparison_datasets_parameters___*')
# f2=[]
# for i in range(len(f1)):
#     x=f1[i]
#     if 'scaler' not in x:
#         f2.append(x)
        
# fg+= f2

# for f in fg:
#     cmd='cp -v '+f+' '+fr
#     os.system(cmd)
    
#%%    



import glob
import os

img_files = []
img_files += glob.glob('*.png')
img_files += glob.glob('*.jpeg')

s='\n\n'
for fn in img_files:
    #txt = 'convert '+fn+' -trim +repage '+fn
    #fnc =fn.replace('.png', '.jpeg').replace('.PNG', '.jpeg')
    fnc =fn.replace('.png', '.svg').replace('.PNG', '.svg').replace('.jpeg','.svg')
    txt = 'convert -units PixelsPerInch '+fnc+' -density 1200 '+fnc
    print(txt)
    s=s+'cp  '+fnc+' '+fnc.replace('_train_yudongnan_toc','')+' \n'
    s=s+'cp  '+fnc+' /tmp; \n'
    #os.system(txt)

print(s)


s=''
for fn in glob.glob('*cross_table*'):
   s='mv '+fn+' '+fn.replace('_train_yudongnan_toc','') 
   print(s)
   os.system(s)
   

