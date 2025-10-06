import numpy as np
from numba import jit
from typing import Union
from pathlib import Path
import prody
import frustratometer
import pickle

import cazar_mariposas as cazar

AA_dict_full_pydca_like = {"A": 0, "C": 1,"D": 2,"E": 3,"F": 4,"G": 5,"H": 6,"I": 7,"K": 8,"L": 9,"M": 10,"N": 11,
                                "P": 12,"Q": 13,"R": 14,"S": 15,"T": 16,"V": 17,"W": 18,"Y": 19,"-": 20,'X':20,'Y':20,'Z':20,'B':20}

def get_fields(fields_and_couplings_all, MSA, states): 
    
    """
    Genera un numpy array de dimensiones (MSA.shape[1]), states) que contiene información sobre los campos locales hi para cada posición del alineamiento
    a partir del array que devuelve la librería pydca luego de computar un plmDCA. La función está adaptada de una implementación de la librería pydca.

    Parámetros
    -----------
    fields_and_couplings_all: un array generado por la librería pydca donde guarda la información de campos locales y acoplamientos obtenida mediante un plmDCA
    MSA: un alineamiento de secuencias. Usa esta información para calcular las dimensiones que tendrá el campo h.
    states: la cantidad de estados posibles para cada posición del alineamiento. Proteínas = 21 (20 aa + gap). ARN = 5 (4 b + gap) 

    Returns
    -----------
    Un numpy array de dimensiones (MSA.shape[1]), states) que contiene información sobre los campos locales hi de cada estado posible para cada posición
    del alineamiento de secuencias. 

    """
    

    Hi = np.zeros((MSA.shape[1], states))
    for i in range(MSA.shape[1]):
        for a in range(states):
            k = a+i*21
            h = fields_and_couplings_all[k]
            Hi[i,a] = h
    qq = [20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    Hi = Hi[:,qq]
    return Hi 

def get_Jij(fields_and_couplings_all, MSA, inst): 
    
    """
    Genera un numpy array de dimensiones (MSA.shape[1]), states) que contiene información sobre los acoplamientos Jij para todos los pares posibles de
    posiciones del alineamiento, a partir del array que devuelve la librería pydca luego de computar un plmDCA. La función está adaptada de una 
    implementación de la librería pydca.
    
    Parámetros
    -----------
    fields_and_couplings_all: un array generado por la librería pydca donde guarda la información de campos locales y acoplamientos obtenida mediante un plmDCA
    MSA: un alineamiento de secuencias. Usa esta información para calcular las dimensiones que tendrá el campo Jij.
#####states: la cantidad de estados posibles para cada posición del alineamiento. Proteínas = 21 (20 aa + gap). ARN = 5 (4 b + gap) 
##### #################################### esta función está definida de 1 para 21 estados o sea ta pensada para proteínas. habría que elegir un camino
#y casarse con eso, no me gustó hacer una cosa distinta en cada función no? capaz lo mejor es clavarle stataes a todo quien te dice que no vas a usar arn (?
    inst: la instancia de la clase PlmDCA de la librería pydca que tenés que crear para generar el array fields_and_couplings_all. Básicamente lo necesitas
    acá porque para ir a buscar el acoplamiento de cada posición a la librería pydca necesitas una función particular que depende de esta instancia. 
    (imaginate lo recauchutado de librerías que tengo este archivo!!!)

    Returns
    -----------
    Un numpy array de dimensiones (MSA.shape[1]), MSA.shape[1]), states, states) que contiene información sobre los acoplamientos Jij de cada estado posible
    para todos los pares de posiciones posibles del alineamiento de secuencias. 
    
    """
    

    Jij = np.zeros((MSA.shape[1],MSA.shape[1],21,21))
    for i in range(MSA.shape[1]-1):
        for j in range(i+1, MSA.shape[1]):
            for a in range(21):
                for b in range(21):
                    indx =  inst.map_index_couplings(i, j , a, b) 
                    Jij[i,j,a,b] = (fields_and_couplings_all[indx])      
    qq = [20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] #Si después de documentar todo esta porongueta 
    J_ = Jij[:,:,qq,:]                                                              #sigue andando acá hay algo q mejorar
    J_final = J_[:,:,:,qq]
    return J_final

def get_h_j(fields_and_couplings_all, MSA, states, inst, simetric_J = True):
    
    """
    Genera dos numpy arrays, h y J, que contienen información sobre los campos locales hi para cada posición del alineamiento y sobre los acoplamientos
    Jij para todos los pares posibles de posiciones del alineamiento, a partir del array que devuelve la librería pydca luego de computar un plmDCA. La función
    está adaptada de una implementación de la librería pydca.

    Parámetros
    -----------
    fields_and_couplings_all: un array generado por la librería pydca donde guarda la información de campos locales y acoplamientos obtenida mediante un plmDCA
    MSA: un alineamiento de secuencias. Usa esta información para calcular las dimensiones que tendrán los campos hi y Jij.
#####states: la cantidad de estados posibles para cada posición del alineamiento. Proteínas = 21 (20 aa + gap). ARN = 5 (4 b + gap) 
##### #################################### esta función está definida de 1 para 21 estados en J o sea ta pensada para proteínas. habría que elegir un camino
#y casarse con eso, no me gustó hacer una cosa distinta en cada función no? capaz lo mejor es clavarle stataes a todo quien te dice que no vas a usar arn (?
    inst: la instancia de la clase PlmDCA de la librería pydca que tenés que crear para generar el array fields_and_couplings_all. Básicamente lo necesitas
    acá porque para ir a buscar el acoplamiento de cada posición a la librería pydca necesitas una función particular que depende de esta instancia. 
    (imaginate lo recauchutado de librerías que tengo este archivo!!!)

    Returns
    -----------
    Dos numpy arrays de dimensiones (MSA.shape[1]), MSA.shape[1]), states, states) y (MSA.shape[1]), states) que contienen información sobre los acoplamientos
    Jij de cada estado posible para todos los pares de posiciones posibles del alineamiento de secuencias y sobre los campos locales hi de cada estado posible 
    para cada posición del alineamiento de secuencias, respectivamente (!!!).
    
    """
    

    h = get_fields(fields_and_couplings_all, MSA, states)
    J = get_Jij(fields_and_couplings_all, MSA, inst)
    if simetric_J is True:
        for i in range(J.shape[0]):
            for j in range(J.shape[0]):
                for a in range(states):
                    for b in range(states):
                        J[j,i,b,a] = J[i,j,a,b]
        return h, J
    else:
        return h,J
    

def J_simetric(potts_model):
    J_og = potts_model["J"]
    for i in range(J_og.shape[0]):
        for j in range(J_og.shape[1]):
            for a in range(J_og.shape[2]):
                for b in range(J_og.shape[3]):
                    if J_og[i,j,a,b] != J_og[j,i,b,a]:
                        return False
    return True

def zero_sum_gauge(Hi,Jij,m=2.0):
    # m=2 (j!=i)
    # m=1 (j>i)
    npos,q=Hi.shape

    ha=Hi.sum(axis=1)
    Ja=Jij.sum(axis=2) 
    Jb=Jij.sum(axis=3)
    Jab=Jij.sum(axis=(2,3))

    Ja_=np.einsum('lijk->ijlk',np.tile(Ja,(q,1,1,1)))
    Jb_=np.einsum('lijk->ijkl',np.tile(Jb,(q,1,1,1)))
    Jab_=np.einsum('lkij->ijkl', np.tile(Jab,(q,q,1,1)))   

    Hi_=np.zeros(Hi.shape)
    for i in range(npos):
        for a in range(q):
            Hi_[i,a]=Hi[i,a]-ha[i]/q+(Jij[i,:,a,:].sum()/q-Jij[i,:,:,:].sum()/(q**2)+Jij[:,i,:,a].sum()/q-Jij[:,i,:,:].sum()/(q**2))/m
        
    Jij_=Jij-Ja_/q-Jb_/q+Jab_/(q**2)
    
    return Hi_,Jij_


def load_potts(simetric_potts_model_path, file_format, return_awsem_model = False, structure = None):
    '''    
    Carga el modelo de potts construido usando DCA y genera un modelo de potts basado en AWSEM a partir de una estructura de referencia

    Parameters
    ----------
    simetric_potts_model_path : bastante declarativo el nombre, sobre todo la parte de SIMETRIC!!! Podría agregar en esta misma función una linea de 
    codigo que se fije si el modelo es simetrico o no, si no lo es que lo haga simetrico y que a partir de ahora trabaje con eso. también supongo
    q podría agregar un cartel de error que diga algo así como "tu modelo de potts tiene que ser simetrico capo!!!" si agarra un modelo asimetrico
    structure : un objeto del tipo que genera la funcion cargar_est_ref
     
    Returns
    -------
    potts_model_dca : el modelo de potts creado usando DCA; es necesario que el archivo a cargar sea un diccionario (con dos claves, "h" y "J", donde el valor
    asociado a cada una sean las matrices (en formato de numpy array) de campos locales y de acoplamentios, respectivamente) guardado como un archivo de 
    extension .pkl --> medio tosco todo eso no? pero si no me caso con un formato hay un millon de formas de guardar esta info. que se encargue el usuario 
    supongo
    potts_model_awsem : el modelo de potts creado usando AWSEM, en formato diccionario de con dos claves, "h" y "J", donde el valor asociado a cada una son
    las matrices (en formato de numpy array) de campos locales y de acoplamientos, respectivamente.
   '''
    formatos_validos = ["npz", "pkl"] #Abierto a modificar la función con distintos formatos para guardar esto
    if file_format not in formatos_validos:
        raise ValueError(f"Formato de archivo no válido. Debe ser uno de los siguientes: {formatos_validos}")

    elif file_format == "pkl":
        with open(simetric_potts_model_path, 'rb') as file:
            potts_model_dca = pickle.load(file)
    
    else:
        potts_model_dca = np.load(simetric_potts_model_path)

    h = potts_model_dca["h"]
    J = potts_model_dca["J"]
    
    if not J_simetric(potts_model_dca):
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                for a in range(J.shape[2]):
                    for b in range(J.shape[3]):
                        J[j,i,b,a] = J[i,j,a,b]
    
    Hi, Jij = zero_sum_gauge(h, J, m=2.0)
    potts_model_dca = {"h" : Hi, "J" : Jij}
    if return_awsem_model == True:
        awsem_inst = frustratometer.AWSEM(structure,
                                          k_electrostatics=0.0,
                                          min_sequence_separation_rho=3,
                                          min_sequence_separation_contact=2
)
        potts_model_awsem = awsem_inst.potts_model
        return  potts_model_dca, potts_model_awsem
    else:
        return potts_model_dca


def trimm_potts_and_seq(potts, sequence, alphabet, return_positions = False):
    valid_aas = set(alphabet)
    positions_to_keep = [i for i, aa in enumerate(sequence) if aa in valid_aas]
#    positions_to_keep = [i for i, aa in enumerate(sequence) if aa != '-']
    h = potts["h"]
    J = potts["J"]
    h_trimmed = h[positions_to_keep]
    J_trimmed = J[np.ix_(positions_to_keep, positions_to_keep, range(21), range(21))]
    seq_trimmed = ''.join([sequence[i] for i in positions_to_keep])
    potts_trimmed = {"h":h_trimmed, "J":J_trimmed}
    
    if return_positions ==True:
        return potts_trimmed, seq_trimmed, positions_to_keep
    else:
        return potts_trimmed, seq_trimmed

#Esta de acá abajo es para calcular la norma de Frobenius para plotear un mapa de contactos
def Fnorm (J):
    
    """
    Descripción

    Parámetros
    -----------

    Returns
    -----------

    """
    

    F = np.zeros((J.shape[0],J.shape[0],(J.shape[0])**2))
    for i in range(J.shape[0]):
        for j in range(J.shape[0]):
            f = 0
            for a in range(21):
                for b in range(21):
                    f += (J[i,j,a,b])**2
            Fij = np.sqrt(f)
            F[i,j,i*j] = Fij
    return F

#Esta de acá abajo es para calcular la energía de una secuencia dado un modelo de potts. 
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


def campo_h(f,corr=1e-4): #fiajte que esto te devuelve cosas positivas!!!! y será de suma importancia esto para el futuro
    h=np.log(f+corr)
    return h-np.tile(np.nansum(h,axis=1)/21,(21,1)).T

def gen_ind_model(fi, fij, corr = 1e-4):
    hi_ = campo_h(fi, corr)
    Jij_ =  np.zeros(fij.shape)
    hi, Jij = cazar.zero_sum_gauge(hi_,Jij_)
    return {"h":hi, "J":Jij}


def mask_Fnorm_to_plot(data, min_distance = 4, number_of_indices = 100):
    data = np.triu(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j-i <= min_distance:
                data[i,j] = 0
    
    
    # Paso 1: Encontrar las 100 coordenadas más altas
    flattened_data = data.flatten()  # Aplanar la matriz a 1D
    top_indices = np.argpartition(flattened_data, -number_of_indices)[-number_of_indices:]  # Índices de los 100 valores más altos
    
    # Paso 2: Crear una máscara para ocultar el resto
    mask = np.zeros_like(flattened_data, dtype=bool)  # Máscara inicializada en False
    mask[top_indices] = True  # Marcar las 100 coordenadas más altas
    mask = mask.reshape(data.shape)  # Volver a la forma original de la matriz

    
    # Paso 3: Aplicar la máscara al heatmap (los valores no seleccionados se ponen como NaN)
    masked_data = np.where(mask, data, np.nan)  # Usar NaN para los valores no seleccionados
    return masked_data

def multiplicador_ank (n, reps_dict): #n será la cantidad de repeticiones totales, reps_dict un diccionario que contenga info del h y J de una sola repeat (y de la información de la interacción entre 2 repeats obvio) 

    J_result = np.zeros((n*33, n*33, 21, 21))
    h_result = np.zeros((n*33, 21))
    
    
    if n == 1:
        J_result[:33, :33, :, :] == reps_dict["J_rep"]
        h_result[:33, :] == reps_dict["h_rep"]
        
    else: 
        for i in range(n):
            J_result[i*33:(i+1)*33, i*33:(i+1)*33, :, :] = reps_dict["J_rep"]
            h_result[i*33:(i+1)*33, :] = reps_dict["h_rep"]
            
            if i != n-1:
                J_result[i*33:(i+1)*33, (i+1)*33:(i+2)*33, :, :] = reps_dict["J_int_up"]

    for i in range(J_result.shape[0]):
        for j in range(J_result.shape[1]):
            for a in range(J_result.shape[2]):
                for b in range(J_result.shape[3]):
                    J_result[j,i,b,a] = J_result[i,j,a,b]

    

    return {"h":h_result, "J":J_result}


def write_tcl_script_DCA(pdb_file: Union[Path,str], chain: str, mask: np.array, distance_matrix: np.array, distance_cutoff: float, single_frustration: np.array,
                    pair_frustration: np.array, tcl_script: Union[Path, str] ='frustration.tcl',max_connections: int =None, movie_name: Union[Path, str] =None, still_image_name: Union[Path, str] =None) -> Union[Path, str]:
    """
    Writes a tcl script that can be run with VMD to superimpose the frustration patterns onto the corresponding PDB structure. 

    Parameters
    ----------
    pdb_file :  Path or str
        pdb file name
    chain : str
        Select chain from pdb
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
    distance_matrix : np.array
        LxL array for sequence of length L, describing distances between contacts
    distance_cutoff : float
        Maximum distance at which a contact occurs
    single_frustration : np.array
        Array containing single residue frustration index values
    pair_frustration : np.array
        Array containing pair (ex. configurational, mutational, contact) frustration index values
    tcl_script : Path or str
        Output tcl script file with static structure
    max_connections : int
        Maximum number of pair frustration values visualized in tcl file
    movie_name : Path or str
        Output movie file with rotating structure
    still_image_name : Path or str
        Output image file with still image
    

    Returns
    -------
    tcl_script : Path or str
        tcl script file
    """
    fo = open(tcl_script, 'w+')
    single_frustration = np.nan_to_num(single_frustration,nan=0,posinf=0,neginf=0)
    pair_frustration = np.nan_to_num(pair_frustration,nan=0,posinf=0,neginf=0)
    
    
    structure = prody.parsePDB(str(pdb_file))
    selection = structure.select('protein', chain=chain)
    residues = np.unique(selection.getResnums())

    fo.write(f'[atomselect top all] set beta 0\n')
    # Single residue frustration
    # Asignar valores de beta usando umbrales
    for r, f in zip(residues, single_frustration):
        if f < -0.83:
            fo.write(f'[atomselect top "chain {chain} and residue {int(r-1)}"] set beta -1.0\n')  # Verde
        elif f > 0.83:
            fo.write(f'[atomselect top "chain {chain} and residue {int(r-1)}"] set beta 1.0\n')  # Rojo
        else:
            fo.write(f'[atomselect top "chain {chain} and residue {int(r-1)}"] set beta 0.0\n')  # Gris
    
    # Mutational frustration:
    r1, r2 = np.meshgrid(residues, residues, indexing='ij')
    sel_frustration = np.array([r1.ravel(), r2.ravel(), pair_frustration.ravel(),distance_matrix.ravel(), mask.ravel()]).T
    #Filter with mask and distance
    if distance_cutoff:
        mask_dist=(sel_frustration[:, -2] <= distance_cutoff)
    else:
        mask_dist=np.ones(len(sel_frustration),dtype=bool)
    sel_frustration = sel_frustration[mask_dist & (sel_frustration[:, -1] > 0)]
    
    minimally_frustrated = sel_frustration[sel_frustration[:, 2] < -1.18]
    #minimally_frustrated = sel_frustration[sel_frustration[:, 2] < -1.78]
    sort_index = np.argsort(minimally_frustrated[:, 2])
    minimally_frustrated = minimally_frustrated[sort_index]
    if max_connections:
        minimally_frustrated = minimally_frustrated[:max_connections]
    fo.write('draw color green\n')
    

    for (r1, r2, f, d ,m) in minimally_frustrated:
        r1=int(r1)
        r2=int(r2)
        if abs(r1-r2) == 1: # don't draw interactions between residues adjacent in sequence
            continue
        pos1 = selection.select(f'resid {r1} and chain {chain} and (name CB or (resname GLY and name CA))').getCoords()[0]
        pos2 = selection.select(f'resid {r2} and chain {chain} and (name CB or (resname GLY and name CA))').getCoords()[0]
        distance = np.linalg.norm(pos1 - pos2)
        if d > 9.5 or d < 3.5:
            continue
        fo.write(f'lassign [[atomselect top "resid {r1} and name CA and chain {chain}"] get {{x y z}}] pos1\n')
        fo.write(f'lassign [[atomselect top "resid {r2} and name CA and chain {chain}"] get {{x y z}}] pos2\n')
        if 3.5 <= distance <= 6.5:
            fo.write(f'draw line $pos1 $pos2 style solid width 2\n')
        else:
            fo.write(f'draw line $pos1 $pos2 style dashed width 2\n')

    frustrated = sel_frustration[sel_frustration[:, 2] > 1.18]
    #frustrated = sel_frustration[sel_frustration[:, 2] > 0]
    sort_index = np.argsort(frustrated[:, 2])[::-1]
    frustrated = frustrated[sort_index]
    if max_connections:
        frustrated = frustrated[:max_connections]
    fo.write('draw color red\n')
    for (r1, r2, f ,d, m) in frustrated:
        r1=int(r1)
        r2=int(r2)
        if d > 9.5 or d < 3.5:
            continue
        fo.write(f'lassign [[atomselect top "resid {r1} and name CA and chain {chain}"] get {{x y z}}] pos1\n')
        fo.write(f'lassign [[atomselect top "resid {r2} and name CA and chain {chain}"] get {{x y z}}] pos2\n')
        if 3.5 <= d <= 6.5:
            fo.write(f'draw line $pos1 $pos2 style solid width 2\n')
        else:
            fo.write(f'draw line $pos1 $pos2 style dashed width 2\n')
    
    fo.write('''mol delrep top 0
            mol color Beta
            mol representation NewCartoon 0.300000 10.000000 4.100000 0
            mol selection all
            mol material Opaque
            mol addrep top
            color scale method GWR
            mol scaleminmax top 0 -1.0 1.0
            axes location Off
            color Display Background white
            display resize 800 800
            display projection Orthographic

            
            ''')

    fo.close()
    return tcl_script
