import pandas as pd
import numpy as np

from Bio import SeqIO
import Bio.PDB
import frustratometer
import pydca

import prody
import pickle
import os
import time

from scipy.spatial.distance import pdist,squareform
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol
import matplotlib

def cargar_est_ref(structure_path, chain = None, maximum_contact_distance = 10.0, minimum_sequence_separation = 4): #???
    '''
    Carga una estructura PDB de referencia usando la librería frustratometer. Devuelve la estructura de referencia, la secuencia, matriz de distancias,
    una máscara para computar la frustración DCA y una máscara para computar la frustración AWSEM.

    Parameters
    ----------
    structure_path : ...
    maximum_contact_distance : la distancia máxima para considerar un contacto. residuos a distancias mayores a este cutoff en la matriz de distancia serán 
    considerados False para armar la mask de AWSEM. ########### ¿¿¿Típicamente 8??? ##############
    minimum_sequence_separation : la distancia mínima entre dos residuos para considerarlos en contacto. ##### 1 implica que residuos que estén al lado serán 
    considerados en contacto; me suenta que la distancia mínima entre contactos tiene que ser 4, pero es verdad que las cosas que están al lado en 
    #secuencia están en contacto también obviamente, entonces?? ###########
    
    Returns
    -------
    structure : una instancia de la librería frustratometer la cual permite obtener distintos parámetros de una estructura de PDB (estructura primaria, matriz
    de distancia, ...
    distance_matrix : la matriz de distancias de los distintos residuos en la estructura de referencia. ########### esto solo lo usa para calcular en esta 
    misma función la máscara de awsem y puedo usarlo en otra función para crear archivos tcl que coloreen un pdb, que ya vimos que los colores no tienen
    mucho significado para un DCA; cappaz no es necesario que esta funcion devuelva esta matriz 
    mask_full : una "máscara" que contiene todos True, es decir que para posteriores cálculos considerará todos los pares de "contactos"
    mask_awsem : una "máscara" usada para calcular la frustración usando AWSEM, donde solo serán considerados como contactos aquellos residuos que estén 
    separados en la secuencia por una minimum_sequence_separation y que estén como mucho a una distancia igual a maximum_contact_distance en la matriz de 
    distancias

    '''
    structure = frustratometer.classes.Structure(pdb_file=structure_path, chain=chain, repair_pdb=repair_pdb)
#    seq = structure.sequence
    distance_matrix=structure.distance_matrix
    mask_full = frustratometer.frustration.compute_mask(distance_matrix)
    mask_awsem = frustratometer.frustration.compute_mask(distance_matrix, maximum_contact_distance, minimum_sequence_separation)

    return structure, mask_full, mask_awsem

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

def cargar_modelo(simetric_potts_model_path, structure, file_format):
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
    
        awsem_inst = frustratometer.classes.AWSEM(structure)
        potts_model_awsem = awsem_inst.potts_model

    else:
        potts_model_dca = np.load(simetric_potts_model_path)

        awsem_inst = frustratometer.classes.AWSEM(structure)
        potts_model_awsem = awsem_inst.potts_model

    h = potts_model_dca["h"]
    J = potts_model_dca["J"]
    
    if not J_simetric(potts_model_dca):
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                for a in range(J.shape[2]):
                    for b in range(J.shape[3]):
                        J[j,i,b,a] = J[i,j,a,b]
    
    Hi, Jij = zero_sum_gauge(h, J, m=2.0)
    #potts_sum_zero = {"h":Hi, "J":Jij}
    potts_model_dca = {"h" : Hi, "J" : Jij}
    
    return  potts_model_dca, potts_model_awsem

def load_potts(simetric_potts_model_path, file_format):
    '''    
    Parameters
    ----------    
    Returns
    -------

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
    
    return  potts_model_dca

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


def msa_subset(msa_path, outfile_path, n, seqs_weights, file_format = 'fasta'):
    
    fasta_sequences = list(SeqIO.parse(msa_path, file_format))
    seqs=[]
    names=[]
    for j in range(len(fasta_sequences)):
        names.append(fasta_sequences[j].id)
        seqs.append(fasta_sequences[j].seq)

    w_norm = seqs_weights / seqs_weights.sum()
    
    selected_indexes = np.random.choice(np.arange(len(seqs)), n, p=w_norm, replace=False)
    
    selected_seqs = []
    selected_names = []
    selected_w = []
    
    for index in selected_indexes:
        selected_seqs.append(seqs[index])
        selected_names.append(names[index])
        selected_w.append(seqs_weights[index])
    
    ofile = open(outfile_path, "w")
    
    for j in range(len(selected_seqs)):
        seq_j = "".join(np.array(selected_seqs[j]).tolist())
        ofile.write(">" + selected_names[j] + " " + str(selected_w[j]) + "\n" + seq_j + "\n")
    
    ofile.close()    

def cargar_msa(msa_path, file_format = 'fasta'):
    '''
    Parameters
    ----------
    
    
    Returns
    -------
    '''
#    alignment1 = '/home/fede/LFP/ank_frustra/MSA_subset_4.fasta'
    fasta_sequences = list(SeqIO.parse(msa_path, file_format))
    seqs=[]
    names=[]
    for j in range(len(fasta_sequences)):
        names.append(fasta_sequences[j].id)
        seqs.append(fasta_sequences[j].seq)
    return seqs, names

def catalogar_secuencia(energia, media, desvio):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if (energia < media - desvio): #Las que son al menos dos sd mejores q la media
        return "buena"
    elif (energia < media + desvio and energia > media - desvio): #están a menos de 1 sd de la media
        return "media"
    else: #el resto
        return "mala"

######################### RECONTRA CHECKPOINT HERMANAOOOOOOOOOOOOO ###########################

def categorizar(energies, media, desvio):
    categorizadas = []
    for i in range(len(energies)):
        cat_i = catalogar_secuencia(energies[i], media, desvio)
        categorizadas.append(cat_i)
    return categorizadas

def frustrar(seq, dca_model, awsem_model, mask_full, mask_awsem):
    
    residues_freq = frustratometer.frustration.compute_aa_freq(seq)
    pairwise_freq = frustratometer.frustration.compute_contact_freq(seq)
    
    #First DCA 
    decoy_fluctuation_sr_dca = - frustratometer.frustration.compute_singleresidue_decoy_energy_fluctuation(seq, dca_model, mask_full)
    frustration_sr_dca = frustratometer.frustration.compute_single_frustration(decoy_fluctuation_sr_dca, residues_freq)
    
    decoy_fluctuation_c_dca = - frustratometer.frustration.compute_mutational_decoy_energy_fluctuation(seq, dca_model, mask_full)
    frustration_c_dca = frustratometer.frustration.compute_pair_frustration(decoy_fluctuation_c_dca, pairwise_freq)

    
    #Then awsem
    decoy_fluctuation_sr_awsem = - frustratometer.frustration.compute_singleresidue_decoy_energy_fluctuation(seq, awsem_model, mask_awsem)
    frustration_sr_awsem = frustratometer.frustration.compute_single_frustration(decoy_fluctuation_sr_awsem, residues_freq)
    
    decoy_fluctuation_c_awsem = - frustratometer.frustration.compute_mutational_decoy_energy_fluctuation(seq, awsem_model, mask_awsem)
    frustration_c_awsem = frustratometer.frustration.compute_pair_frustration(decoy_fluctuation_c_awsem, pairwise_freq)

    return frustration_sr_dca, frustration_c_dca, frustration_sr_awsem, frustration_c_awsem

def frustrar_dca(seq, dca_model, mask_full):
    
    residues_freq = frustratometer.frustration.compute_aa_freq(seq)
    pairwise_freq = frustratometer.frustration.compute_contact_freq(seq)
    
    #First DCA 
    decoy_fluctuation_sr_dca = - frustratometer.frustration.compute_singleresidue_decoy_energy_fluctuation(seq, dca_model, mask_full)
    frustration_sr_dca = frustratometer.frustration.compute_single_frustration(decoy_fluctuation_sr_dca, residues_freq)
    
    decoy_fluctuation_c_dca = - frustratometer.frustration.compute_mutational_decoy_energy_fluctuation(seq, dca_model, mask_full)
    frustration_c_dca = frustratometer.frustration.compute_pair_frustration(decoy_fluctuation_c_dca, pairwise_freq)

    return frustration_sr_dca, frustration_c_dca


def single_residue_frustration_media_y_desvio_por_posicion(single_residue_frustration_list):
    media_por_posicion = []
    desvio_por_posicion = []
    
    for i in range((single_residue_frustration_list[0]).shape[0]):
        media_por_posicion.append(np.mean(np.array(single_residue_frustration_list)[:,i]))
        desvio_por_posicion.append(np.std(np.array(single_residue_frustration_list)[:,i]))
    
    return media_por_posicion, desvio_por_posicion

def contact_frustration_media_y_desvio_por_contacto(contact_frustration_list):

    contact_frustration_media = np.zeros(contact_frustration_list[0].shape)
    contact_frustration_desvio = np.zeros(contact_frustration_list[0].shape)
    
    for j in range((contact_frustration_list[0]).shape[0]):
            for k in range((contact_frustration_list[0]).shape[1]):
                
                contact_frustration_media[j,k] = np.mean(np.array(contact_frustration_list)[:,j,k])            
                contact_frustration_desvio[j,k] = np.std(np.array(contact_frustration_list)[:,j,k])
    return contact_frustration_media, contact_frustration_desvio

def frustar_secuencias(seqs_list, dca_model, awsem_model, mask_full, mask_awsem):
    
    frustration_sr_dca_list = []
    frustration_c_dca_list = []
    
    frustration_sr_awsem_list = []
    frustration_c_awsem_list = []
    
    for i in range(len(seqs_list)):
        frustration_sr_dca, frustration_c_dca, frustration_sr_awsem, frustration_c_awsem = frustrar(seqs_list[i], dca_model, awsem_model, mask_full, mask_awsem)
        frustration_sr_dca_list.append(frustration_sr_dca)
        frustration_c_dca_list.append(frustration_c_dca)
        frustration_sr_awsem_list.append(frustration_sr_awsem)
        frustration_c_awsem_list.append(frustration_c_awsem)
    
    return frustration_sr_dca_list, frustration_c_dca_list, frustration_sr_awsem_list, frustration_c_awsem_list


def frustra_calc(structure_path, simetric_potts_model_path, msa_path,
                 maximum_contact_distance = 10.0, 
                 minimum_sequence_separation = 3,
                 msa_file_format = 'fasta',
                 potts_file_format = 'npz',
                 seqid = 0.8
                ):
    
    structure, mask_full, mask_awsem = cargar_est_ref(structure_path, maximum_contact_distance, minimum_sequence_separation)
    
    dca_model, awsem_model = cargar_modelos(simetric_potts_model_path, structure, potts_file_format)
    
    seqs, names = cargar_msa(msa_path, msa_file_format)
    
    energies = frustratometer.frustration.compute_sequences_energy(seqs, dca_model, mask_full)

    #plmdca_inst = pydca.plmdca.PlmDCA(msa_path, 'protein', seqid)
    #w = plmdca_inst.compute_seqs_weight() #me gustaria una que calcule pesos pero esta en particular es malisssima

    categorias = categorizar(energies, np.mean(energies), np.std(energies))

    frustration_sr_dca_list, frustration_c_dca_list, frustration_sr_awsem_list, frustration_c_awsem_list = frustar_secuencias(seqs, dca_model, awsem_model, mask_full, mask_awsem)

    df_seqs = pd.DataFrame({
        "energy" : energies,
        "id" : names, 
        "seq" : seqs,
        "category" : categorias,
        "DCA_frustration_single_residue" : frustration_sr_dca_list,
        "DCA_frustration_pairwise" : frustration_c_dca_list,
        "AWSEM_frustration_single_residue" : frustration_sr_awsem_list,
        "AWSEM_frustration_pairwise" : frustration_c_awsem_list,
        })

    frustration_sr_media_DCA, frustration_sr_desvio_DCA = single_residue_frustration_media_y_desvio_por_posicion(frustration_sr_dca_list)
    frustration_c_media_DCA, frustration_c_desvio_DCA = contact_frustration_media_y_desvio_por_contacto(frustration_c_dca_list)
    
    frustration_sr_media_AWSEM, frustration_sr_desvio_AWSEM = single_residue_frustration_media_y_desvio_por_posicion(frustration_sr_awsem_list)
    frustration_c_media_AWSEM, frustration_c_desvio_AWSEM = contact_frustration_media_y_desvio_por_contacto(frustration_c_awsem_list)

    df_single_residue_frustration = pd.DataFrame({
        "Frustracion_media_single_residue_DCA" : frustration_sr_media_DCA,
        "Frustracion_desvio_single_residue_DCA" : frustration_sr_desvio_DCA,

        "Frustracion_media_single_residue_AWSEM" : frustration_sr_media_AWSEM,
        "Frustracion_desvio_single_residue_AWSEM" : frustration_sr_desvio_AWSEM})

    df_pairwise_frustration = pd.DataFrame({
        "Frustracion_contacto_media_DCA" : [frustration_c_media_DCA],
        "Frustracion_contacto_desvio_DCA" : [frustration_c_desvio_DCA],
        
        "Frustracion_contacto_media_AWSEM" : [frustration_c_media_AWSEM],
        "Frustracion_contacto_desvio_AWSEM": [frustration_c_desvio_AWSEM]})

    #df_pairwise_frustration["Frustracion_contacto_media_DCA"][0] así para acceder a cada valor de este data frame medio horrible

    return df_seqs, df_single_residue_frustration, df_pairwise_frustration

def run_frustration_analisis(path, pdb_name, potts_model_name, msa_name, 
                            n = None, 
                            maximum_contact_distance = 10.0, 
                            minimum_sequence_separation = 3,
                            msa_file_format = 'fasta',
                            potts_file_format = 'npz',
                            seqid = 0.8,
                            seqs_weights = None):
    
    pdb_path = path + pdb_name
    potts_model_path = path + potts_model_name
    msa_path = path + msa_name
    
    if n is not None:
        saving_path = f"{path}results/random_test_de_{n}_seqs/"
        os.makedirs(saving_path, exist_ok=True)
#        outfile_path = saving_path + "msa.fasta" #the one below ensures that it never overwrites
        outfile_path = os.path.join(saving_path, f"msa_subset_{n}_{int(time.time())}.fasta")

        msa_subset(msa_path, outfile_path, n, seqs_weights, msa_file_format)
        
        df_seqs, df_single_residue_frustration, df_pairwise_frustration = frustra_calc(pdb_path, potts_model_path, outfile_path, maximum_contact_distance, minimum_sequence_separation, msa_file_format, potts_file_format, seqid)
        
        with open(f"{saving_path}df_seqs.pkl", 'wb') as f1:
            pickle.dump(df_seqs, f1)
        
        with open(f"{saving_path}df_single_residue_frustration.pkl", 'wb') as f2:
            pickle.dump(df_single_residue_frustration, f2)
        
        with open(f"{saving_path}df_pairwise_frustration.pkl", 'wb') as f3:
            pickle.dump(df_pairwise_frustration, f3)

    else:
        saving_path = f"{path}results/full_msa/"
        os.makedirs(saving_path, exist_ok=True)

        df_seqs, df_single_residue_frustration, df_pairwise_frustration = frustra_calc(pdb_path, potts_model_path, msa_path, maximum_contact_distance, minimum_sequence_separation, msa_file_format, potts_file_format, seqid)
        
        with open(f"{saving_path}df_seqs.pkl", 'wb') as f1:
            pickle.dump(df_seqs, f1)
        
        with open(f"{saving_path}df_single_residue_frustration.pkl", 'wb') as f2:
            pickle.dump(df_single_residue_frustration, f2)
        
        with open(f"{saving_path}df_pairwise_frustration.pkl", 'wb') as f3:
            pickle.dump(df_pairwise_frustration, f3)
