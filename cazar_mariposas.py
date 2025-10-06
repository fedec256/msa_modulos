import pandas as pd
import numpy as np
import frustratometer
import pickle
import os
import time
from alingments_functions import load_msa, msa_subset
from dca_functions import load_potts

def cargar_est_ref(structure_path, chain = None, maximum_contact_distance = 10.0, minimum_sequence_separation = 4):
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
    structure = frustratometer.Structure(pdb_file=structure_path, chain=chain, repair_pdb=False)
    distance_matrix=structure.distance_matrix
    mask_full = frustratometer.frustration.compute_mask(distance_matrix)
    mask_awsem = frustratometer.frustration.compute_mask(distance_matrix, maximum_contact_distance, minimum_sequence_separation)

    return structure, mask_full, mask_awsem

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
    
    dca_model, awsem_model = load_potts(simetric_potts_model_path, potts_file_format, return_awsem_model = True, structure = structure)
    
    seqs, names = load_msa(msa_path, msa_file_format)
    
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
