import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from data import dd_gibbs_b,gibbs_b,gibbs_l,gibbs_a,d_gibbs_a,d_gibbs_b


#%% Top of miscibility gap

def top_miscibility():
    def err_T(T):
        res = minimize(fun = dd_gibbs_b,
                       x0 = 0.5,
                       args = (T),
                       method = "Nelder-Mead",
                       tol = 1e-6)
        min_x = res.x
        err = np.abs(dd_gibbs_b(min_x,T))    
        return err
    
    T_max = minimize(fun = err_T,
                   x0 = 1000,
                   method = "Nelder-Mead",
                   tol = 1e-6)
    
    c_T_max = minimize(fun = dd_gibbs_b,
                   x0 = 0.5,
                   args = (T_max.x[0]),
                   method = "Nelder-Mead",
                   tol = 1e-6)
    
    print(f"The top of the miscibility gap is {T_max.x[0]:.2f} K"
          +f" and {100*c_T_max.x[0]:.2f} at% of Mo.")
    
    return [c_T_max.x[0]],[T_max.x[0]]

#%% Melting points

def melting():
    def err_melting(T,c_mo):
        delta = np.abs(gibbs_b(c_mo, T) - gibbs_l(c_mo, T))
        return delta
        
    T_melting_Ti = minimize(fun = err_melting,
                            x0 = 1000,
                            args = (0),
                            method = "Nelder-Mead",
                            tol = 1e-6)
    
    T_melting_Mo = minimize(fun = err_melting,
                            x0 = 1000,
                            args = (1),
                            method = "Nelder-Mead",
                            tol = 1e-6)
    
    print(f"The melting point of Ti is {T_melting_Ti.x[0]:.2f} K"
          +f" and for Mo is {T_melting_Mo.x[0]:.2f} K.")

    return [0,1],[T_melting_Ti.x[0],T_melting_Mo.x[0]]

#%% Transformation Ti from alpha to beta

def trafo():
    def err_trafo(T,c_mo):
        delta_trafo = np.abs(gibbs_a(c_mo, T) - gibbs_b(c_mo, T))
        return delta_trafo
    
    T_trafo_Ti = minimize(fun = err_trafo,
                          x0 = 1000,
                          args = (0),
                          method = "Nelder-Mead",
                          tol = 1e-6)
    
    print(f"The temperature of the transformation of Ti is {T_trafo_Ti.x[0]:.2f} K.")

    return [0],[T_trafo_Ti.x[0]]                     


#%% The triple point

def triple():
    def err_triple(x):
        T,c1,c2,c3 = x[0],x[1],x[2],x[3]
        delta1 = np.abs(d_gibbs_a(c1, T) - d_gibbs_b(c2, T))
        delta2 = np.abs(d_gibbs_b(c2, T) - d_gibbs_b(c3, T))
        delta3 = np.abs(gibbs_a(c1, T) - gibbs_b(c2, T) - d_gibbs_b(c2, T)*(c1-c2))
        delta4 = np.abs(gibbs_b(c2, T) - gibbs_b(c3, T) - d_gibbs_b(c2, T)*(c2-c3))
        delta = delta1 + delta2 + delta3 + delta4
        return delta
        
    T_triple = minimize(fun = err_triple,
                        x0 = [1000,0.01,0.15,0.4],
                        method = "Nelder-Mead",
                        tol = 1e-8,
                        options = {"maxiter":1000})
        
    # print(T_triple) 
    # x = np.linspace(0,1,100)
    # plt.plot(x,gibbs_a(x, T_triple.x[0]))
    # plt.plot(x,gibbs_b(x, T_triple.x[0]))
    # plt.scatter(T_triple.x[1],gibbs_a(T_triple.x[1], T_triple.x[0]))
    # plt.scatter(T_triple.x[2],gibbs_b(T_triple.x[2], T_triple.x[0]))
    # plt.scatter(T_triple.x[3],gibbs_b(T_triple.x[3], T_triple.x[0]))
    # plt.xlim(0,0.5)
    # plt.ylim(-4000,0)
    # plt.show()
    
    print(f"The temperature of the triple point is {T_triple.x[0]:.2f} K.")
    
    return [T_triple.x[1],
            T_triple.x[2],
            T_triple.x[3]],[T_triple.x[0],
                            T_triple.x[0],
                            T_triple.x[0]]


#%% the triple point, the hard and robust way

def badness(X,f1,f2,df1,df2,punish_proximity=False):
    x1 = X[0]
    x2 = X[1]
    err1 = np.abs(df1(x1) - df2(x2))
    err2 = np.abs(f2(x2)-(f1(x1)+(x2-x1)*df1(x1)))
    return err1 + err2 + punish_proximity*(1/np.abs(x1-x2))


def tangent_beta(T):
    Ga = lambda c_mo: gibbs_a(c_mo,T)
    Gb = lambda c_mo: gibbs_b(c_mo,T)
    dGa = lambda c_mo: d_gibbs_a(c_mo,T)
    dGb = lambda c_mo: d_gibbs_b(c_mo,T)
    res = minimize(fun = badness,
                   x0 = [0.15,0.5],
                   args = (Gb,Gb,dGb,dGb,True),
                   method = "Nelder-Mead",
                   tol = 1e-8)
    res = minimize(fun = badness,
                   x0 = res.x,
                   args = (Gb,Gb,dGb,dGb,False),
                   method = "Nelder-Mead",
                   tol = 1e-8)
    c2, c3 = res.x
    return c2, c3


def fun_diff(x,f1,f2):
    return f2(x)-f1(x)


def line(x,x1,x2,y1,y2):
    y = y1 + ((y2-y1)/(x2-x1))*(x-x1)
    return y


def err_triple(T):
    Ga = lambda c_mo: gibbs_a(c_mo,T)
    Gb = lambda c_mo: gibbs_b(c_mo,T)
    dGa = lambda c_mo: d_gibbs_a(c_mo,T)
    dGb = lambda c_mo: d_gibbs_b(c_mo,T)

    c2, c3 = tangent_beta(T)
    tangent = lambda x: line(x,c2,c3,Gb(c2),Gb(c3))

    c1_res = minimize(fun = fun_diff,
                      x0 = 0.01,
                      args = (tangent,Ga),
                      method = "Nelder-Mead",
                      tol = 1e-8)

    delta = np.abs(fun_diff(c1_res.x,tangent,Ga))[0]
    return delta

#x = np.linspace(950,1050,100)
#plt.plot(x,[err_triple(T) for T in x])
#plt.show()

T_triple = minimize(fun = err_triple,
                    x0 = 950,
                    method = "Nelder-Mead",
                    tol = 1e-8)
    
#print(T_triple)

if __name__ == "__main__":
    x = np.linspace(0,1,2000)
    T = T_triple.x[0]
    Ga = lambda c_mo: gibbs_a(c_mo,T)
    Gb = lambda c_mo: gibbs_b(c_mo,T)
    c2, c3 = tangent_beta(T)
    tangent = lambda x: line(x,c2,c3,Gb(c2),Gb(c3))
    c1 = minimize(fun = fun_diff,
                      x0 = 0.01,
                      args = (tangent,Ga),
                      method = "Nelder-Mead",
                      tol = 1e-8).x[0]
    plt.scatter([c1,c2,c3],[0,0,0],c="r")
    plt.plot(x,Ga(x)-tangent(x),label="(G_alpha - tangent)")
    plt.plot(x,Gb(x)-tangent(x),label="(G_beta - tangent)")
    plt.text(0.25,50,f"T = {T:.2f} K")
    plt.legend()
    plt.xlim(0,0.5)
    plt.ylim(0,100)

    
    
    
    
    
    
    
    
    
    
    