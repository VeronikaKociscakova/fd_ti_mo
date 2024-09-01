# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 14:46:50 2024

@author: Verča

definition of three Gibbs energies for Ti-Mo
three Gibbs = three phases ->  a for alpha,
                               b for beta,
                               l for liquid
"""

import numpy as np

# gas constant (J mol^-1 K^-1)
R = 8.314463 


def gibbs_mix(c1,c2,T):
    """
    Mixture Gibbs energy for 2-elemenet mixture.
    
    Explanation:
        border case: c1 = 0 or c2 = 0, then using limits and lHospital rule is
            equation c1 * lnc1 = 0, so instead counting element after element, 
            using quicker method for array,
            so instead nan we defined as 0 by numpy - function nan to num
    
    Parameters
    ----------
    c1 : float or np.array of float
        at% of the first component.
    c2 : float or np.array of float
        at% of the second component.
    T : float
        The temperature (K) of the mixture.

    Returns
    -------
    mix_energy : float or np.array of float
        The mixture term of the Gibbs energy.

    """
    
    mix_energy =  R * T * (c1 * np.log(c1) + c2 * np.log(c2))
    mix_energy = np.nan_to_num(mix_energy)
    
    return mix_energy


def gibbs_l(c_mo,
            T):
    """
    The Gibbs potential for the liquid phase of Ti-Mo.

    Parameters
    ----------
    c_mo : float or np.array of float
        Molybdenum fraction (at%). 
    T : float
        Temperature (K).

    Returns
    -------
    energy : float or np.array of float
        The Gibbs energy for the combination of c_mo, T.

    """
    
    c_ti = 1 - c_mo
    
    f_term = (16234 - 8.368 * T) * c_ti + (24267 - 8.368 * T) * c_mo
    mix_term = gibbs_mix(c_ti,c_mo,T)
    b_term = 10136 * c_mo * c_ti
    c_term = 0
   
    energy = f_term + mix_term + b_term + c_term
    
    return energy


def gibbs_b(c_mo,
            T):
    """
    The Gibbs potential for the beta (bcc) phase of Ti-Mo.

    Explanation of the internals: 1 - 2 * c_mo = c_ti - c_mo /for c_term

    Parameters
    ----------
    c_mo : float or np.array of float
        Molybdenum fraction (at%).
    T : float
        Temperature (K).

    Returns
    -------
    energy : float or np.array of float
        The Gibbs energy for the combination of c_mo, T.

    """
    c_ti = 1 - c_mo
    
    f_term = 0
    mix_term = gibbs_mix(c_ti,c_mo,T)
    b_term = 10000 * c_mo * c_ti
    c_term = 9000 * c_mo * c_ti * (c_ti - c_mo)
   
    energy = f_term + mix_term + b_term + c_term
   
    return energy


def gibbs_a(c_mo,
            T):
    """
    The Gibbs potential for the alpha (hcp) phase of Ti-Mo.

    Parameters
    ----------
    c_mo : float or np.array of float
        Molybdenum fraction (at%).
    T : float
        Temperature (K).

    Returns
    -------
    energy : float or np.array of float
        The Gibbs energy for the combination of c_mo, T.

    """
    c_ti = 1 - c_mo
    
    f_term = (-4351 + 3.77 * T) * c_ti + 8368 * c_mo
    mix_term = gibbs_mix(c_ti,c_mo,T)
    b_term = 24386 * c_mo * c_ti
    c_term = 0
   
    energy = f_term + mix_term + b_term + c_term
    
    return energy


#%% This section is only executed as this script is the main process.
#   Otherwise (such as if imported by another script), this section is 
#   not executed.
#   It is used for visual check.

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for T in [300,800,1500,3000,5000]: #K
        x = np.linspace(0,100,1000) #at%
        plt.plot(x,gibbs_a(x/100, T),label="alpha")
        plt.plot(x,gibbs_b(x/100, T),label="beta")
        plt.plot(x,gibbs_l(x/100, T),label="liquid")
        plt.xlabel("Mo [at%]")
        plt.suptitle(f"{T} K")
        plt.legend()
        plt.show()
    