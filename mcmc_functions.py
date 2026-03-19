import numpy as np
import numba
from numba import njit, jit
from joblib import Parallel, delayed
from typing import Optional
import os
from multiprocessing import cpu_count
import datetime

@jit(nopython=True)
def E_tot(seq,h,J):
    
    """
    Descripción

    Parámetros
    -----------

    Returns
    -----------

    """
    

    E_seq = 0
    l = len(seq)
    for i in range (l):
        ai = int(seq[i]) #acá tengo cual es el aminoácido en la posicion i, ai
        jij = 0
        for j in range (i+1, l):
            bj = int(seq[j]) #acá tengo cual es el aminoacido en la posicion j, bj
            jij += J[i,j,ai,bj] #ahí tengo todos los acoplamientos del aminoácido ai con todos los bj
        E_seq += - (h[i,ai]) - jij
    return E_seq

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
    if seq0 is None:
        seq = np.random.randint(0, Naa, size=npos) #random sequence
    else:
        seq = seq0.copy()

    e0=E_tot(seq,Hi,Jij)
    
    n_saves = (nsteps - transient) // save_each
    energies = np.zeros(n_saves)
    seq_to_save=np.zeros((n_saves,npos),dtype=numba.int64)
    save_count = 0
    
    for i in range(nsteps):
#        residues=list(range(0,Naa))
        x=np.random.randint(npos) # choice random position in sequence 
        old_res=seq[x]
#        residues.remove(old_res)

        new_res = old_res
        while new_res == old_res:            #choice random aa until is different from original.
            new_res = np.random.randint(Naa) #should it also accept the same aa at each step???

        seq[x] = new_res # mutation
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
        if (i - transient) % save_each == 0 and i>=transient:

            seq_to_save[save_count,:] = seq.copy()
            energies[save_count] = e0
            save_count += 1
    return energies, seq_to_save
    
@njit(inline="always")
def MCseq_until_energy_treshold(npos:int, Naa:int, temp:float, 
                       Hi:np.ndarray, Jij:np.ndarray,
                       target_energy:float, temp_freeze = 1e-12, 
                       save_each = 1, 
                       seq0:Optional[np.ndarray] = None, 
                       max_steps = 1000000):
    if seq0 is None:
        seq = np.random.randint(0, Naa, size=npos)
    else:
        seq = seq0.copy()

    e0 = E_tot(seq, Hi, Jij)

    max_saves = (max_steps ) // save_each 
    energies = np.zeros(max_saves)
    seq_to_save=np.zeros((max_saves,npos),dtype=numba.int64)
    save_count = 0

    step = 0
    current_temp = temp

    while step < max_steps and e0 > target_energy:

        # if treshold is surpassed, temp of freezing
        if e0 <= target_energy:
            current_temp = temp_freeze

        # --- MCMC step ---
        x = np.random.randint(npos)
        old_res = seq[x]

        new_res = old_res
        while new_res == old_res: 
            new_res = np.random.randint(Naa) #choice new until different

        seq[x] = new_res
        ef = E_tot(seq, Hi, Jij)
        de = ef - e0

        if de <= 0: #metropolis criterium same as before
            e0 = ef
        else:
            if np.random.rand() < np.exp(-de / current_temp):
                e0 = ef
            else:
                seq[x] = old_res

        # save
        if step % save_each == 0:
            energies[save_count] = (e0)
            seq_to_save[save_count] = seq
            save_count += 1

        step += 1

    return energies[:save_count], seq_to_save[:save_count]


def generate_trajectory_to_freeze(
        path, 
        Hi,Jij,
        target_energy, temp=1.0, temp_freeze = 1e-12,
        seq0=None,
        max_steps=1000000, save_each=1, extra_steps = 10000):
    
    npos,Naa=Hi.shape
    energies_to_treshold, sequences_to_treshold = MCseq_until_energy_treshold(
        npos, Naa, temp,
        Hi, Jij,
        target_energy, temp_freeze, 
        save_each, 
        seq0, max_steps
        )
    
    seq_final = sequences_to_treshold[-1]
    energies_after_freezing, sequences_after_freezing = MCseq( #extra steps with Tfrozen
        extra_steps, npos, Naa, temp_freeze,
        Hi, Jij,
        save_each ,transient = 0, seq0=seq_final)
    
    energies_freezing = np.concatenate([energies_to_treshold, energies_after_freezing])
    sequences_freezing = np.concatenate([sequences_to_treshold, sequences_after_freezing])

    


    #Create a unique name por each simulation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sequences_file_name = f'sequences_to_freezing_{timestamp}.npy'
    energies_file_name = f'energies_to_freezing_{timestamp}.npy'
    params_file_name    = f"params_to_freezing_{timestamp}.npz"
    saving_dir = os.path.join(path, f"simulation_to_freezing_{timestamp}")
    os.makedirs(saving_dir, exist_ok=True)
    np.save(os.path.join(saving_dir,sequences_file_name), sequences_freezing)
    np.save(os.path.join(saving_dir,energies_file_name), energies_freezing)

    np.savez(
        os.path.join(saving_dir, params_file_name),
        target_energy=target_energy,
        temp=temp,
        temp_freeze=temp_freeze,
        max_steps=max_steps,
        save_each=save_each,
        npos=npos,
        Naa=Naa,
        seq0_provided=(seq0 is not None),
        timestamp=timestamp
    )




def generate_trajectory_from_random_to_folded(
        path, 
        Hi,Jij,
        NSeq, temp=1.0, transient=0, save_each=1):
    
    npos,Naa=Hi.shape
    nsteps=transient+save_each*NSeq
    energies_evolving, sequences_evolving = MCseq(
        nsteps, npos, Naa, temp,
        Hi, Jij,
        save_each ,transient)
    
    #Create a unique name por each simulation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sequences_file_name = f'sequences_from_random_{timestamp}.npy'
    energies_file_name = f'energies_from_random_{timestamp}.npy'
    params_file_name    = f"params_from_random_{timestamp}.npz"
    saving_dir = os.path.join(path, f"simulation_from_random_{timestamp}")
    os.makedirs(saving_dir, exist_ok=True)
    np.save(os.path.join(saving_dir,sequences_file_name), sequences_evolving)
    np.save(os.path.join(saving_dir,energies_file_name), energies_evolving)

    np.savez(
        os.path.join(saving_dir, params_file_name),
        temp=temp,
        nsteps=nsteps,
        save_each=save_each,
        transient=transient,
        npos=npos,
        Naa=Naa,
        timestamp=timestamp
    )

def run_single_sequence(i, MSA, nsteps, npos, Naa, temp, Hi, Jij, transient):
    
    seq_i = MSA[i, :].copy()

    energies, seqs = MCseq(
        nsteps,
        npos,
        Naa,
        temp,
        Hi,
        Jij,
        save_each = 1,
        transient = transient,
        seq0 = seq_i
    )

    final_energy = energies[-1]
    final_seq = seqs[-1]

    return i, final_energy, final_seq


def freezing_alignment(path, MSA, nsteps, Hi, Jij, temp = 1e-16, transient = 0, 
                       path_in_process:str = None):

    nseq, npos = MSA.shape
    Naa = Hi.shape[1]
    
    if path_in_process is None:
        #Create a unique name por each simulation
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        saving_dir = os.path.join(path, f"frozen_alignment_{timestamp}")
        os.makedirs(saving_dir, exist_ok=True)
    
        frozen_alignment = np.zeros((nseq, npos), dtype=np.int64)
        frozen_energies = np.zeros(nseq)
        frozen_energies[:] = 0.0

    else:
        saving_dir = os.path.normpath(path_in_process)

        timestamp = os.path.basename(saving_dir).replace("frozen_alignment_", "") #getting timestamp from folder

        frozen_alignment = np.load(os.path.join(saving_dir, "frozen_alignment.npy"))
        frozen_energies  = np.load(os.path.join(saving_dir, "frozen_energies.npy"))

    indices = [i for i in range(nseq) if frozen_energies[i] == 0]
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))

    print(f"Total secuencias: {nseq}")
    print(f"Pendientes: {len(indices)}")

    if len(indices) == 0:
        print("Nada para hacer, todo ya computado.")
        return

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_single_sequence)(
            i, MSA, nsteps, npos, Naa, temp, Hi, Jij, transient
        )
        for i in indices
    )

    for i, energy, seq in results:
        frozen_energies[i] = energy
        frozen_alignment[i, :] = seq


    np.save(os.path.join(saving_dir, "frozen_alignment.npy"), frozen_alignment)
    np.save(os.path.join(saving_dir, "frozen_energies.npy"), frozen_energies)

    np.savez(
        os.path.join(saving_dir, f"frozen_params_{timestamp}.npz"),
        temp=temp,
        nsteps=nsteps,
        transient=transient,
        npos=npos,
        Naa=Naa,
        timestamp=timestamp
    )








#this one i think is broken we should DESTROY!!! no, fix
def generate_seq_ensemble(path, num_cores,Hi,Jij,NSeq,temp=1.0,transient=40000,save_each=5000):
    
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
#    np.save(path+name_energies,energies)
#    np.save(path+name_seqs,ali)

    #Create a unique name por each simulation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sequences_file_name = f'ensemble_of_sequences_{timestamp}.npy'
    energies_file_name = f'ensemble_of_energies_{timestamp}.npy'
    params_file_name    = f"params_of_ensemble_{timestamp}.npz"
    saving_dir = os.path.join(path, f"simulation_of_ensemble_{timestamp}")
    os.makedirs(saving_dir, exist_ok=True)
    np.save(os.path.join(saving_dir,sequences_file_name), ali)
    np.save(os.path.join(saving_dir,energies_file_name), energies)

    np.savez(
        os.path.join(saving_dir, params_file_name),
        temp=temp,
        Nseq=NSeq,
        nsteps=nsteps,
        save_each=save_each,
        transient=transient,
        npos=npos,
        Naa=Naa,
        timestamp=timestamp
    )


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
