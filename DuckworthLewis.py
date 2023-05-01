import pandas
import numpy as np
import math
import matplotlib.pyplot as plot
from scipy.optimize import minimize

def func(Z,doc):
    L=Z[10]
    MSE=0
    for wickets in range(1,11):
        new = doc[doc['Wickets.in.Hand']==wickets]
        overs = 50-new['Over']
        rem_runs = new['Innings.Total.Runs']-new['Total.Runs']       # or rem_runs = new['Runs.Remaining']
        prod_func_runs = Z[wickets-1]*(1 - np.exp((-1*L/Z[wickets-1])*overs))
        MSE = MSE + (np.sum((rem_runs - prod_func_runs)**2))/len(doc)
    return MSE

# Data Preprocessing

doc = pandas.read_csv('./04_cricket_1999to2011.csv')
doc = doc[doc['Innings']==1 & (doc['Wickets.in.Hand'] != 0)]
req_data = doc[['Over','Innings.Total.Runs','Total.Runs','Wickets.in.Hand']]

# or  req_data = doc[['Over', 'Runs.Remaining', 'Wickets.in.Hand']]

'''The below Section is for cleaning Total.Runs column in case it consists of incorrect values
Also if Runs.Remaining consist of some error then we need to do Innings.Total.Runs - Total.Runs'''

for i in range(len(doc)):
    if doc.iloc[i]['Over']==1:
        doc.iloc[i]['Total.Runs'] = doc.iloc[i]['Runs']
    else:
        doc.iloc[i]['Total.Runs'] = doc.iloc[i]['Runs'] + doc.iloc[i-1]['Total.Runs']


def DuckworthLewis():
    x_init = np.array([280 for i in range(11)])
    x_init[10]=0.32
    MSE = np.zeros(10)
    opt_para = minimize(fun=func,x0=x_init,args=(req_data))
    Z = opt_para.x[:10]
    L = opt_para.x[10]
    #print(opt_para.fun)
    for wickets in range(1,11):
        new = req_data[req_data['Wickets.in.Hand']==wickets]
        overs = 50-new['Over']
        rem_runs = new['Innings.Total.Runs']-new['Total.Runs']       # or  new['Runs.Remaining']
        prod_func_runs = Z[wickets-1]*(1 - np.exp((-1*L/Z[wickets-1])*overs))
        MSE[wickets-1] = MSE[wickets-1] + (np.sum((rem_runs - prod_func_runs)**2))/len(req_data)
    return Z,L,MSE


Z0,L,MSE = DuckworthLewis()

for w in range(1,11):
    axis1 = range(0,51)
    axis2 = [Z0[w-1]*(1-math.exp(-1*L*o/Z0[w-1])) for o in range(0,51)]
    plot.plot(axis1,axis2)
    plot.text(axis1[37],axis2[37],w)
    plot.xlabel('Overs')
    plot.ylabel('Resources Remaining')
    plot.title('Run Production Function')
plot.grid()
plot.show()
print('The Optimal value of Parameter Z0 for 10 wickets are:')
for i in range(10):
    print(Z0[i])
print("The optimal value of L = ", L)
print('Mean Square Error(Normalised Error) = ', np.sum(MSE))
