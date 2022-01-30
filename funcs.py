from cmath import sqrt
import numpy as np

def mean_inter(data,t):
    n = len(data)
    stad = np.std(data)
    means = np.mean(data)
    if n >= 30:
        S = sqrt(n/(n-1)) * stad
        interL = means - (t*S/sqrt(n))
        interU = means + (t*S/sqrt(n))
        interv = np.array([interL,interU])
    elif n < 30:
        interL = means - (t*stad/sqrt(n-1))
        interU = means + (t*stad/sqrt(n-1))
        interv = np.array([interL,interU])
    return interv

def var_inter(data,t=None,Xl=None,Xu=None):
    n = len(data)
    stad = np.std(data)
    means = np.mean(data)
    S = (n/(n-1))*(stad**2)
    if n >= 30:
        interL = S - (t*S/sqrt(2/n-1))
        interU = S + (t*S/sqrt(2/n-1))
        interv = np.array([interL,interU])
    if n < 30:
        interL = S*(n*-1)/Xu
        interU = S*(n*-1)/Xl
        interv = np.array([interL,interU])
    return interv