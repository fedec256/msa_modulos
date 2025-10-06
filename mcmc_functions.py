import numpy as np
import numba
from numba import njit
import random
from dca_functions import E_tot
from joblib import Parallel, delayed
from typing import Optional

#Estas de acá abajo son funciones para generar secuencias optimizadas con el campo por pasos de montecarlo por metropoli hastings
@njit(inline="always")
def MCseq(nsteps:int, npos:int, Naa:int, temp:float,
          Hi: np.ndarray,
          Jij: np.ndarray,
          save_each:int ,transient:int, seq0:Optional[np.ndarray] = None    
          ):
    
    """
    Descripción

    Parameters
    -----------

    Returns
    -----------

    """
    if seq0 is not None:
        seq = seq0
    else:
        seq=np.random.randint(0, Naa, size=npos) # generate random sequence

    e0=E_tot(seq,Hi,Jij)
    
    n_saves = (nsteps - transient) // save_each
    energies = np.zeros(n_saves)
    seq_to_save=np.zeros((n_saves,npos),dtype=numba.int64)
    save_count = 0
    
    for i in range(nsteps):
        residues=list(range(0,Naa))
        x=np.random.randint(npos) # choice random position in sequence 
        old_res=seq[x]
        residues.remove(old_res)
        seq[x] = np.random.choice(np.array(residues)) # mutation
        ef=E_tot(seq,Hi,Jij) # energy after mutation
        de=ef-e0 # change in energy
        # metropolis criterium
        if de<=0: 
            e0=ef
        else:
            if np.random.rand()<np.exp(-de/(temp)):
                e0=ef
            else:
                seq[x] = old_res # don't accept    
        if i%save_each==0 and i>=transient:

            seq_to_save[int((i-transient)/save_each),:] = seq.copy()
            energies[save_count] = e0
            save_count += 1
    return energies, seq_to_save
    


def generate_seq_ensemble(path,name_energies, name_seqs, num_cores,Hi,Jij,NSeq,temp=1.0,transient=40000,save_each=5000):
    
    """
    Descripción

    Parámetros
    -----------

    Returns
    -----------

    """
    
    npos,Naa=Hi.shape
    
    nseq=int(NSeq/num_cores)
    nsteps=transient+save_each*nseq #ese save_each es la cantidad de pasos para que dejen de estar correlacionadas 2 secuencias (funcion q lo calcula existe) además cada esa cantidad de secuencias voy a guardar una secuencia de la simulación
    args=nsteps,npos,Naa,temp,Hi,Jij,save_each,transient
    r=Parallel(n_jobs=num_cores,verbose=10)(delayed(MCseq)(*j) for (i,j) in [(i_,args) for i_ in np.arange(num_cores)])
    energies_, seqs_= zip(*r)
    energies=np.concatenate(energies_)
    ali=np.concatenate(seqs_)
    np.save(path+name_energies,energies)
    np.save(path+name_seqs,ali)

#Estas dos de acá abajo sirven para ver el tiempo de montecarlo necesario para que dos secuencias dejen de estar autocorrelacionadas, será el tiempo que tendrás que 
#simular entre un guardado y otro de los algoritmos anteriores. En general con 10000 pasos ya se pierde la autocorrelación entre dos secuencias, pero no está de más probar esto.
#No está de más anotar que mierda es x y lags tampoco!!
#
#
def autocorr1(x,lags):
        
    """
    Descripción

    Parámetros
    -----------

    Returns
    -----------

    """
    
    corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
    return np.array(corr)


def autocorr_time(y,lags):
        
    """
    Descripción

    Parámetros
    -----------

    Returns
    -----------

    """
    
    corr=autocorr1(y,lags)
    aux=np.where(corr>np.exp(-1))[0]
    if len(aux)>0:
        out=aux[-1]+1
    else:
        out=0
    return out
