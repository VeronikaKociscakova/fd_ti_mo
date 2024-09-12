import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize
from data import gibbs_b,gibbs_l,gibbs_a,d_gibbs_a,d_gibbs_b,d_gibbs_l
from tqdm.auto import tqdm
from points import top_miscibility,melting,trafo,triple
matplotlib.rcParams["figure.dpi"] = 300


def badness(X,G1,G2,dG1,dG2,T,punish_proximity=False):
    """
    false = allow one solution
    true = add error, to make two solutions

    """
    x1 = X[0]
    x2 = X[1]
    err1 = np.abs(dG1(x1,T) - dG2(x2,T))
    err2 = np.abs(G2(x2,T)-(G1(x1,T)+(x2-x1)*dG1(x1,T)))
    return err1 + err2 + punish_proximity*(1/np.abs(x1-x2))


def lines(list_of_T,G1,G2,dG1,dG2,x0=[0.01,0.9],punishment=[1,0]):
    x1 = np.zeros(len(list_of_T))
    x2 = np.zeros(len(list_of_T))
    
    for i,T in tqdm(enumerate(list_of_T)):
        approx_res = minimize(fun = badness,
                       x0 = x0,
                       args = (G1,G2,dG1,dG2,T,punishment[0]),
                       method = "Nelder-Mead",
                       tol = 1e-8)
        res = minimize(fun = badness,
                       x0 = approx_res.x,
                       args = (G1,G2,dG1,dG2,T,punishment[1]),
                       method = "Nelder-Mead",
                       tol = 1e-8)
        x1[i] = min(res.x)
        x2[i] = max(res.x)
    
    return x1,x2

#%%

list_of_T = np.arange(500,1004)
x1,x2 = lines(list_of_T,gibbs_a,gibbs_b,d_gibbs_a,d_gibbs_b,punishment=[1,0])
plt.plot(x1,list_of_T,"k")
plt.plot(x2,list_of_T,"k")

list_of_T = np.arange(1005,1154)
x1,x2 = lines(list_of_T,gibbs_a,gibbs_b,d_gibbs_a,d_gibbs_b,x0=[0.01,0.1],punishment=[1,0])
plt.plot(x1,list_of_T,"k")
plt.plot(x2,list_of_T,"k")

list_of_T = np.arange(1005,1063)
x1,x2 = lines(list_of_T,gibbs_b,gibbs_b,d_gibbs_b,d_gibbs_b,x0=[0.1,0.4],punishment=[20,0.1])
plt.plot(x1,list_of_T,"k")
plt.plot(x2,list_of_T,"k")
plt.plot([x1[-1],x2[-1]],[list_of_T[-1],list_of_T[-1]],"k")

list_of_T = np.arange(1941,2899)
x1,x2 = lines(list_of_T,gibbs_b,gibbs_l,d_gibbs_b,d_gibbs_l,x0=[0.1,0.9],punishment=[1,0])
plt.plot(x1,list_of_T,"k")
plt.plot(x2,list_of_T,"k")
plt.plot([0,x1[0]],[list_of_T[0],list_of_T[0]],"k")
plt.plot([1,x1[-1]],[list_of_T[-1],list_of_T[-1]],"k")

for f in [top_miscibility,melting,trafo,triple]:
    c,T = f()
    plt.scatter(c,T,c="r",zorder=10)
    
plt.xlabel('Concentration of Mo [at%]')
plt.ylabel('Temperature [K]')

plt.xlim(0,1)
plt.ylim(500,3000)
plt.savefig("pictures/"+"ekvilibria.png")






