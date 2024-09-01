import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull as ch
from tqdm.auto import tqdm
from data import gibbs_a, gibbs_b, gibbs_l

matplotlib.rcParams["figure.dpi"] = 300


def pba(T,c,list_of_gibbs):
    """
    pba = phase binary alloys
    
    What does it do:
        It determines the phase (or the mixture) for the combination
        of temperature and concentration, by the given gibbs energies.
        Determination of phase:
            It takes the lowest curve in diagram (c, T).
            Mixture is the tangent between two curves.
        
    Parameters
    ----------
    T : float
        temperature [K]
    c : array of float
        concentration (0-1) [at%/100]
    list_of_gibbs : list of functions (c,T)
                    list of all gibbs energies

    Returns
    -------
    p : array of int
        array / list of phases at the temperature T
            len(c) = len(p)
            single digit numbers (1-9) correspond to the gibbs energies
                in the list_of_gibbs, starting with 1!
                that means: single digit = single phase
            double digit numbers (12-89) correspond to mixtures of two phases
                example: 12 means mixture of phases 1 and 2
                rule: 12 and 21 is the same mixture
                      the first digit is the smaller of the two digits, 
                          i.e. 21 is not used.
    """

    #1. matici (n-řádků x 4-sloupce) - řádky, koncentrace, gibbs, fáze    
    number_of_row = np.arange(len(list_of_gibbs)*len(c)+2)

    concentration = np.tile(c,len(list_of_gibbs))
    concentration = np.append(concentration,(0,1)) #add end points (for Hull)

    gibbs = np.zeros(0)
    for g in list_of_gibbs:
        gibbs = np.append(gibbs,g(c,T))
    gibbs = np.append(gibbs,(1e10,1e10)) #add end points (for Hull)

    phase = np.zeros(0)
    for i in range(len(list_of_gibbs)):
        phase = np.append(phase,np.tile(i+1,len(c)))
    phase = np.append(phase,(0,0)) #add end points (for Hull)
    
    mat = np.vstack((number_of_row,concentration,gibbs,phase))
    gibbs_matrix = mat.transpose()
        
    #2. pustím Hulla
    hull = ch(gibbs_matrix[:,1:3])
       
    #3. posloupnost vrcholů
    hull_vertices = hull.vertices[:]   
    # Note:   nutno ověřit, že jsou přítomny 2 krajní body pro Hulla,
    #   tedy s pořadovým číslem len(number_of_row)-1 a len(number_of_row)-2,
    #   neboli 2 nejvyšší čísla
    edge_points_Hull = number_of_row[-2:]
    for point in edge_points_Hull:
        if point in hull_vertices:
            pass
        else:
            raise Exception("Problem with edge_points_Hull, points are missing")
    
    for i in range(2): #delete end points (for Hull)
        hull_vertices = np.delete(hull_vertices,np.argmax(hull_vertices))
           
    #4. vrcholům přiřadím ostatní info ze sloupců z matice 1 - gibbs_matrix
    gibbs_matrix = gibbs_matrix[:,[0,1,3]]
    info = gibbs_matrix[hull_vertices,:]
        
    #5. fáze a směs fází
    info = info[:,[1,2]]                 
    null = np.zeros(len(c))
    phase_start = np.vstack((c,null))
    phase_start = phase_start.transpose()
    
    for i in range(len(c)): #zapisuje fáze
        for j in range(len(info[:,0])):
            if phase_start[i,0] == info[j,0]:
                phase_start[i,1] = info[j,1]
        #možnost zkrátit 2 for cykly, urychlilo by to program
    
    phase_progress = np.array(phase_start[:,1]).astype(int)
    for i in range(len(phase_start[:,0])-1):
        if phase_progress[i+1] == 0:
            phase_progress[i+1] = phase_progress[i]
        else:
            pass
        
    phase_progress_2 = np.array(phase_start[:,1]).astype(int)
    for i in np.arange(len(phase_start[:,0])-1,-1,-1):
        if phase_progress_2[i-1] == 0:
            phase_progress_2[i-1] = phase_progress_2[i]
        else:
            pass
        
    phase = phase_progress_2 - phase_progress    
    for i in range(len(phase_start[:,0])):
        if phase[i]!=0:
            if phase_progress[i] < phase_progress_2[i]:
                phase_start[i,1] = int(str(phase_progress[i])+str(phase_progress_2[i]))
            else:
                phase_start[i,1] = int(str(phase_progress_2[i])+str(phase_progress[i]))
        else:
            phase_start[i,1] = phase_progress[i]
           
    p = phase_start      
    return p


def popd(T,c,list_of_gibbs):
    """
    popd = plot of phase diagram
    
    It takes function pba (phase binary alloys) and uses it for whole vector of 
    temperature. This makes me matrix of temperature and their phase
    composition. that means each temperature has phase composition.   
    Using for cycle on pba function.    
    Show plot of phase diagram of binary alloys.

    Parameters
    ----------
    T : np.array of float
        temperature [K]
    c : np.array of float
        concentration (0-1) [at%/100]
    list_of_gibbs : list of functions (c,T)
                    list of all gibbs energies        

    Returns
    -------
    m: np.array of float
        Matrix of phase composition. One row means one temperature.   
    """
    
    m = np.zeros((len(T),len(c)),int)    
    for i, t in tqdm(enumerate(T)):
        n = pba(t,c,list_of_gibbs)
        m[i,:] = n[:,1]
        
    m = np.flip(m,0)
    m = np.flip(m,1)         
    return m


def plot(m,T,c,fill=False,name=None):
    """
    Show plot in 6 different colours.

    Parameters
    ----------
    m : np.array of float
        Matrix of phase composition. One row means one temperature.
    T : np.array of float
        temperature [K]
    c : np.array of float
        concentration (0-1) [at%/100]
    fill :  bool, optional
            selection of type of the plot, default is false, which means
                line plot
    name :  str or None, optional
            name of file, default is None, which means file is not saved

    Returns
    -------
    None.

    """   
    fig, ax = plt.subplots(1,1)
    
    if fill:        
            # define color map 
        color_map = {1: np.array([249, 220, 92]),  # yellow
                     2: np.array([1, 25, 54]),  # blue, dark
                     3: np.array([250, 216, 214]),  # rose
                     12: np.array([82, 153, 211]), # blue, light
                     23: np.array([255, 63, 0])} # red        
            # make a 3d numpy array that has a color channel dimension   
        m_3d = np.ndarray(shape=(m.shape[0], m.shape[1], 3), dtype=int)
        for i in range(0, m.shape[0]):
            for j in range(0, m.shape[1]):
                m_3d[i,j,:] = color_map[m[i,j]]        
            # display the plot 
        ax.imshow(np.flip(m_3d,1),
                  extent=[c.min(),
                          c.max()*100,
                          T.min(),
                          T.max()])
    else:
        ax.contour(np.flip(m),
                   levels=[1.5,2.5,20],
                  extent=[c.min(),
                          c.max()*100,
                          T.min(),
                          T.max()],
                  colors="black"
                  )
    ax.set_xlabel('Concentration of Mo [at%]')
    ax.set_ylabel('Temperature [K]')
    ax.set_aspect((c.max()*100-c.min())/(T.max()-T.min()))
    
    if name is not None:
        fig.savefig("pictures/"+name+".png")
    else:
        pass
    
    return




T = np.linspace(50,3500,2000)
c = np.linspace(0,1,2000)
gibbs = [gibbs_a,gibbs_b,gibbs_l]
vysledek = popd(T,c,gibbs)
plot(vysledek,T,c,fill=False,name="diagram")
plot(vysledek,T,c,fill=True,name="diagram_filled")

