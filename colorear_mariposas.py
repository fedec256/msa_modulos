import pandas as pd
import numpy as np
from Bio import SeqIO
import seaborn as sns
import matplotlib.pyplot as plt
import random
import Bio.PDB
import numba
from numba import jit, njit
from joblib import Parallel, delayed
import pydca
import prody
import os
import pickle
import frustratometer
import prody
from matplotlib.colors import TwoSlopeNorm, Normalize
import matplotlib.colors as mcolors

from scipy.spatial.distance import pdist,squareform
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol
from matplotlib import cm, colors


import cazar_mariposas as cazar


def colores_normalizados(color_names, frustra_values, vmin, vmax, method = None):

    norm = plt.Normalize(vmin, vmax)
    if method == "log":
        norm = colors.LogNorm(vmin, vmax)

    if method == "TwoSlop":
        #vmin = np.min(frustration_sr_dca)
        #vmax = np.max(frustration_sr_dca)
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    if type(color_names) == str:
        cmap = plt.get_cmap(color_names)
#        norm = plt.Normalize(vmin, vmax)
        colores_rgba = cmap(norm(frustra_values))
    
    elif type(color_names) == list:
        cmap = colors.LinearSegmentedColormap.from_list('my_cmap', color_names)
#        norm = plt.Normalize(vmin, vmax)
        colores_rgba = cmap(norm(frustra_values))

    else:
        print("Chequeate el formato en que pasas los colores querés")
        
    return colores_rgba

def rgba_a_hex(colores_rgba):
    colores_hex = [colors.to_hex(color) for color in colores_rgba]
    return np.array(colores_hex)

def get_CA_coord(view, resid):
    atoms = view.getModel().getAtomList(byresi=resid)
    for atom in atoms:
        if atom.name == 'CA':
            return [atom.coord.x, atom.coord.y, atom.coord.z]
    return None

def pintarPDB(pdb_path, residue_frustration, pairwise_frustration, mask_awsem, color_scale, 
              vmin_sr = None, vmax_sr = None, vmin_pairs = None, vmax_pairs = None,
              title = None, method_sr = None, method_pairs = None, all_atom_residues = None):

    
    if (vmin_sr, vmax_sr, vmin_pairs, vmax_pairs) == (None, None, None, None):
        vmin_sr = np.min(residue_frustration)
        vmax_sr = np.max(residue_frustration)
        pair_values_list = pairwise_frustration[mask_awsem]
        vmin_pairs = np.min(pair_values_list)
        vmax_pairs = np.max(pair_values_list)
    
    with open(pdb_path, 'r') as f: 
        pdb_string = f.read()

    ali_seq = np.arange(1,len(residue_frustration)+1)
    
    view = py3Dmol.view(width=1600, height=1600)
    view.addModel(pdb_string, 'pdb')
    view.setBackgroundColor('white')
    view.setStyle({'cartoon': {'color': 'white'}})

#    colores_single_residue = rgba_a_hex(colores_normalizados(color_scale, residue_frustration, np.min(residue_frustration), np.max(residue_frustration)))
    colores_single_residue = rgba_a_hex(colores_normalizados(color_scale, residue_frustration, vmin_sr, vmax_sr, method_sr))

    
    for i, res in enumerate(ali_seq):
        if res > 0:  # Skip gaps (res=-1)
            view.addStyle({'chain': 'A', 'resi': [str(res)]}, {'cartoon': {'color': colores_single_residue[i]}})

    pair_values_list = pairwise_frustration[mask_awsem]
    
    
    if method_pairs == None:
        norm = colors.Normalize(vmin_pairs, vmax_pairs)
    elif method_pairs == "TwoSlope":
        norm = colors.TwoSlopeNorm(vmin=vmin_pairs, vcenter=0, vmax=vmax_pairs) #Normalize(vmin_pairs, vmax_pairs)

    
    if type(color_scale) == str:
        cmap = plt.get_cmap(color_scale)
    
    elif type(color_scale) == list:
        cmap = colors.LinearSegmentedColormap.from_list('my_cmap', color_scale, 256)
    
#    if type
#    cmap = colors.LinearSegmentedColormap.from_list('my_cmap', color_scale, 256)

    for i in range(len(residue_frustration)):
        for j in range(len(residue_frustration)):
            if mask_awsem[i,j]:
                value = pairwise_frustration[i, j]
                # Normalizar el valor
                normalized_value = norm(value)
                # Obtener el color a partir de la paleta
                color = cmap(normalized_value)
                # Convertir el color a un formato hexadecimal
                hex_color = colors.rgb2hex(color)
                
                view.addLine({'start': {'chain': 'A', 'resi': [str(i + 1)]},
                              'end': {'chain': 'A', 'resi': [str(j + 1)]},
                              'color': hex_color, 'dashed': False, 'linewidth': 400000})

    
    # Mostrar múltiples residuos en formato all-atom (stick)
    if all_atom_residues is not None:
        if type(all_atom_residues) == int: #Teniendo en cuenta que pueda pasarle un único residuo
            view.addStyle({'chain': 'A', 'resi': [str(all_atom_residues)]}, {'stick': {}})
        else: #Y esto por si el usuario se le ocurre pintar varios residuos
            for residue in all_atom_residues:  # Iterar sobre la lista de residuos
                view.addStyle({'chain': 'A', 'resi': [str(residue)]}, {'stick': {}})

    

    
    if title:
        view.addLabel(title, backgroundColor = "white", fontColor = "black", fontsize = 18, alignment = "topLeft")
    view.zoomTo(viewer=(0, 0))
    view.show()

#    return view

def comparar_single_residue_experimentos_dif_n(path, replace_fig = False, fold_name = None, labels = None, dca_awsem = "dca"):
    dca_sr_mean = []
    dca_sr_se = []
    awsem_sr_mean = []
    awsem_sr_se = []
    muestras = []

    for experimento in os.listdir(path):
        path_exp = os.path.join(path, experimento)
        if os.path.isdir(path_exp):
            with open(f'{path_exp}/df_seqs.pkl', 'rb') as Efile:
                df_seqs = pickle.load(Efile)
        
            n = len(df_seqs)    
        
            with open(f'{path_exp}/df_single_residue_frustration.pkl', 'rb') as Efile:
                df_single_residue = pickle.load(Efile)
        
            dca_sr_mean.append(df_single_residue["Frustracion_media_single_residue_DCA"])
            dca_sr_se.append((df_single_residue["Frustracion_desvio_single_residue_DCA"])/np.sqrt(n))
            awsem_sr_mean.append(df_single_residue["Frustracion_media_single_residue_AWSEM"])
            awsem_sr_se.append((df_single_residue["Frustracion_desvio_single_residue_AWSEM"])/np.sqrt(n))
            muestras.append(n)


    if dca_awsem == "dca":
        plt.figure(figsize=(35, 12))
        
        if labels is not None:
            if len(labels) == len(dca_sr_mean):
                a = plt.errorbar(np.arange(len(dca_sr_mean[0])), dca_sr_mean[0], yerr = dca_sr_se[0], label = f"{labels[0]}", linewidth = 8)
                for i in range(1, len(dca_sr_mean)):
                    a = a + plt.errorbar(np.arange(len(dca_sr_mean[i])), dca_sr_mean[i], yerr = dca_sr_se[i], label = f"{labels[i]}", linewidth = 8)
        else:
            a = plt.errorbar(np.arange(len(dca_sr_mean[0])), dca_sr_mean[0], yerr = dca_sr_se[0], label = f"{muestras[0]} obs", linewidth = 8)
            for i in range(1, len(dca_sr_mean)):
                a = a + plt.errorbar(np.arange(len(dca_sr_mean[i])), dca_sr_mean[i], yerr = dca_sr_se[i], label = f"{muestras[i]} obs", linewidth = 8)
        
    #    a
        plt.legend()
        plt.xlabel('Position', fontsize=40)
        plt.ylabel('Mean Single Residue Frustration Index', fontsize=35)
        plt.title(f"Mean DCA single residue frustration index por Posición", fontsize=18)
        x=np.arange(5, len(dca_sr_mean[0])-5, 5)
        xticks=np.arange(5, len(dca_sr_mean[0])+1-5, 5)
        plt.xticks(x, xticks, fontsize = 26)
        plt.yticks(fontsize=26)
    
        if fold_name is not None:
            plt.title(f"Mean DCA single residue frustration index por Posición - {fold_name}", fontsize=18)
            
        
        if not os.path.exists(os.path.join(path, "f_sr_distinto_largo.pdf")) or replace_fig == True:
            plt.savefig(f"{path}/f_sr_distinto_largo_dca.pdf", format="pdf", dpi=300, bbox_inches="tight")
            plt.savefig(f"{path}/f_sr_distinto_largo_dca.png", format="png", dpi=300, bbox_inches="tight")

    
    elif dca_awsem == "awsem":
        plt.figure(figsize=(35, 12))
        
        if labels is not None:
            if len(labels) == len(dca_sr_mean):
                a = plt.errorbar(np.arange(len(awsem_sr_mean[0])), awsem_sr_mean[0], yerr = awsem_sr_se[0], label = f"{labels[0]}", linewidth = 8)
                for i in range(1, len(dca_sr_mean)):
                    a = a + plt.errorbar(np.arange(len(awsem_sr_mean[i])), awsem_sr_mean[i], yerr = awsem_sr_se[i], label = f"{labels[i]}", linewidth = 8)
        else:
            a = plt.errorbar(np.arange(len(dca_sr_mean[0])), awsem_sr_mean[0], yerr = awsem_sr_se[0], label = f"{muestras[0]} obs", linewidth = 8)
            for i in range(1, len(dca_sr_mean)):
                a = a + plt.errorbar(np.arange(len(awsem_sr_mean[i])), awsem_sr_mean[i], yerr = awsem_sr_se[i], label = f"{muestras[i]} obs", linewidth = 8)
        
    #    a
        plt.legend()
        plt.xlabel('Position', fontsize=40)
        plt.ylabel('Mean Single Residue Frustration Index', fontsize=35)
        plt.title(f"Mean AWSEM single residue frustration index por Posición", fontsize=18)
        x=np.arange(5, len(dca_sr_mean[0])-5, 5)
        xticks=np.arange(5, len(dca_sr_mean[0])+1-5, 5)
        plt.xticks(x, xticks, fontsize = 26)
        plt.yticks(fontsize=26)
    
        if fold_name is not None:
            plt.title(f"Mean AWSEM single residue frustration index por Posición - {fold_name}", fontsize=18)
            
        
        if not os.path.exists(os.path.join(path, "f_sr_distinto_largo.pdf")) or replace_fig == True:
            plt.savefig(f"{path}/f_sr_distinto_largo_awsem.pdf", format="pdf", dpi=300, bbox_inches="tight")
            plt.savefig(f"{path}/f_sr_distinto_largo_awsem.png", format="png", dpi=300, bbox_inches="tight")




def mean_single_residue_plot(awsem_sr_mean, dca_sr_mean, dca_std_error, awsem_std_error, fold_name = None, path = None, replace_fig = False):
    plt.figure(figsize=(8, 8))
        
    plt.errorbar(x = awsem_sr_mean, y = dca_sr_mean, yerr = dca_std_error, xerr = awsem_std_error, fmt = 'o', ms = "4")
    
    plt.xlabel('AWSEM mean Single Residue Frustration ', fontsize=16)
    plt.ylabel('DCA mean Single Residue Frustration', fontsize=16)
    plt.xlim(-2, 2)
    plt.ylim(-4, 0)
    
    plt.title(f"Mean single residue frustration AWSEM vs DCA", fontsize=18)
    if fold_name is not None:
        plt.title(f"Mean single residue frustration AWSEM vs DCA - {fold_name}", fontsize=18)

    
    if path is not None and (not os.path.exists(os.path.join(path, "mean_single_residue_awsem_vs_dca.pdf")) or replace_fig == True):
        plt.savefig(f"{path}/mean_single_residue_awsem_vs_dca.pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"{path}/mean_single_residue_awsem_vs_dca.png", format="png", dpi=300, bbox_inches="tight")


def mean_single_residue_per_position(dca_sr_mean, dca_sr_se, fold_name = None, path = None, replace_fig = False, dca_awsem = "dca"):

    plt.figure(figsize=(35, 12))
    
    plt.errorbar(np.arange(len(dca_sr_mean)), dca_sr_mean, yerr = dca_sr_se, linewidth = 8)
    
    plt.xlabel('Position', fontsize=40)
    plt.ylabel('Mean Single Residue Frustration Index', fontsize=35)
    x=np.arange(5, len(dca_sr_mean)-5, 5)
    xticks=np.arange(5, len(dca_sr_mean)+1-5, 5)
    plt.xticks(x, xticks, fontsize = 26)
    plt.yticks(fontsize=26)

    if dca_awsem == "dca":
        plt.title(f"Mean DCA single residue frustration index por Posición", fontsize=18)
    
        
        if fold_name is not None:
            plt.title(f"Mean DCA single residue frustration index por Posición - {fold_name}", fontsize=18)
            
        if path is not None and (not os.path.exists(os.path.join(path, "mean_single_residue_dca_position.pdf")) or replace_fig == True):
            plt.savefig(f"{path}/mean_single_residue_dca_position.pdf", format="pdf", dpi=300, bbox_inches="tight")
            plt.savefig(f"{path}/mean_single_residue_dca_position.png", format="png", dpi=300, bbox_inches="tight")

    elif dca_awsem == "awsem":
        plt.title(f"Mean AWSEM single residue frustration index por Posición", fontsize=18)
    
        
        if fold_name is not None:
            plt.title(f"Mean AWSEM single residue frustration index por Posición - {fold_name}", fontsize=18)
            
        if path is not None and (not os.path.exists(os.path.join(path, "mean_single_residue_awsem_position.pdf")) or replace_fig == True):
            plt.savefig(f"{path}/mean_single_residue_awsem_position.pdf", format="pdf", dpi=300, bbox_inches="tight")
            plt.savefig(f"{path}/mean_single_residue_awsem_position.png", format="png", dpi=300, bbox_inches="tight")



def box_plot_single_residue(df_seqs, dca_sr_mean, std_error, vmin_sr = None, vmax_sr = None, fold_name = None, path = None, replace_fig = False):

#    fig, ax = plt.subplots()

    df_sr_seqs_dca = pd.DataFrame(np.array(df_seqs["DCA_frustration_single_residue"].tolist()))
    
    color_scale = ['green', 'white', 'red']
    
    colores_single_residue = rgba_a_hex(colores_normalizados(color_scale, dca_sr_mean, np.min(dca_sr_mean), np.max(dca_sr_mean)))
    
    if (vmin_sr is not None) and (vmax_sr is not None):
        colores_single_residue = rgba_a_hex(colores_normalizados(color_scale, dca_sr_mean, vmin_sr, vmax_sr))
    
    plt.figure(figsize=(35, 12))
    
    plt.errorbar(np.arange(len(dca_sr_mean)), dca_sr_mean, yerr = std_error, linewidth = 6, c = "gray", alpha = 0.5)
    
    sns.boxplot(data=df_sr_seqs_dca, orient='v', palette=colores_single_residue)

    cmap = colors.LinearSegmentedColormap.from_list('my_cmap', color_scale)

    norm = colors.TwoSlopeNorm(vmin=vmin_sr, vcenter=0, vmax=vmax_sr)
#    norm = plt.Normalize(vmin=vmin_sr, vmax=vmax_sr)
    colorbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    colorbar.ax.tick_params(labelsize=25)
    
    plt.title(f"Boxplots del Single residue frustration index de todas las secuencias por Posición", fontsize=18)
    if fold_name is not None:
        plt.title(f"Boxplots del Single residue frustration index de todas las secuencias por Posición - {fold_name}", fontsize=18)
    
    plt.xlabel('Position', fontsize=40)
    plt.ylabel('Single Residue Frustration Index', fontsize=35)
    x=np.arange(5, len(dca_sr_mean)-5, 5)
    xticks=np.arange(5, len(dca_sr_mean)+1-5, 5)
    plt.xticks(x, xticks, fontsize = 26)

#    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    
    if path is not None and (not os.path.exists(os.path.join(path, "mean_single_residue_dca_position.pdf")) or replace_fig == True): 
        plt.savefig(f"{path}/boxplot_y_mean_single_residue_frustration_position.pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"{path}/boxplot_y_mean_single_residue_frustration_position.png", format="png", dpi=300, bbox_inches="tight")



def get_cmap_from_arg(color_scale):
    """
    Devuelve un objeto matplotlib Colormap a partir de:
      - nombre de colormap (str),
      - lista de colores (list),
      - o un objeto Colormap ya instanciado.
    """
    if isinstance(color_scale, str):
        cmap = cm.get_cmap(color_scale)
    elif isinstance(color_scale, list):
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', color_scale, N=256)
    elif isinstance(color_scale, mcolors.Colormap):
        cmap = color_scale
    else:
        raise ValueError(f"color_scale must be a str, list, or Colormap instance, not {type(color_scale)}")
    return cmap

def color_pdb_by_vector(
    pdb_path,
    vector_to_project: np.ndarray,
    vmax=None,
    vmin=None,
    highlighted_sites=None,
    color_scale='viridis',
    chain='A',
    index_base=0,
    highlight_style='stick',
    sphere_color='#FFD700',
    sphere_radius=1.2,
    cartoon_radius=0.8,
    opacity = 1.0
):
    """
    Pinta un PDB con py3Dmol usando los valores en vector_to_project para colorear
    residuo por residuo. Devuelve el objeto py3Dmol.view.
    
    Args:
        pdb_path (str): ruta al archivo .pdb
        vector_to_project (np.ndarray): array 1D con un valor por residuo (floats)
        vmax, vmin (float|None): límites. Si None se usan max/min del vector.
        highlighted_sites (list|np.ndarray|None): índices de residuos a resaltar
            (ver index_base).
        colormap (str): nombre de colormap de matplotlib (ej 'viridis').
        chain (str): cadena por defecto para seleccionar en el pdb (ej 'A').
        index_base (int): 0 si highlighted_sites usa indexing 0-based (default),
            1 si usa 1-based (PDB style).
        sphere_color (str): color hex de las esferas de resaltado.
        sphere_radius (float): radio de las esferas de resaltado (Å).
        cartoon_radius (float): grosor/escala para estilo cartoon (opcional).
    Returns:
        py3Dmol.view
    """

    # --- parse pdb, extrayendo CA por residuo (orden natural del archivo) ---
    residues = []  # lista de dicts: {'chain','resSeq','resName','ca':(x,y,z)}
    seen = set()
    with open(pdb_path, 'r') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain_id = line[21].strip() or ' '
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain_id, resseq, icode)
            if atom_name == 'CA' and key not in seen:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                residues.append({
                    'chain': chain_id,
                    'resSeq': int(resseq),
                    'resName': resname,
                    'ca': (x, y, z)
                })
                seen.add(key)

    n_res = len(residues)
    vector = np.asarray(vector_to_project, dtype=float).ravel()
    if vector.size != n_res:
        raise ValueError(f"Mismatch: vector length = {vector.size} but pdb has {n_res} residues (CA count). "
                         "Si tu vector no comienza en la primera posición del PDB tendrás que alinear "
                         "las posiciones (p.ej. con un argumento pdb_seq_beg en una versión futura).")

    # --- determine vmin/vmax and normalization ---
    if vmax is None:
        vmax = float(np.nanmax(vector))
    if vmin is None:
        vmin = float(np.nanmin(vector))

    # avoid degenerate case vmax == vmin
    if np.isclose(vmax, vmin):
        # expand a tiny bit so Normalize won't blow up
        eps = 1e-6 if vmax == 0 else abs(vmax) * 1e-6
        vmax = vmax + eps
        vmin = vmin - eps

    # choose norm: TwoSlopeNorm centered at 0 if 0 lies between vmin and vmax
    if (vmin < 0) and (vmax > 0):
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        # all positive or all negative -> linear normalization
        norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = get_cmap_from_arg(color_scale)

    # map values -> hex colors
    rgba_colors = cmap(norm(vector))  # Nx4
    hex_colors = [mcolors.to_hex(rgba) for rgba in rgba_colors]

    # --- build py3Dmol view and add PDB ---
    view = py3Dmol.view(width=900, height=600)
    with open(pdb_path, 'r') as f:
        pdb_text = f.read()
    view.addModel(pdb_text, 'pdb')
    view.setStyle({'cartoon': {'radius': cartoon_radius,
                              'opacity': opacity}})  # base style for everything
    view.zoomTo()

    # Apply per-residue colors by adding a style per-residue
    # (py3Dmol acepta selecciones por chain+resi)
    for i, res in enumerate(residues):
        sel = {'chain': res['chain'], 'resi': str(res['resSeq'])}
        # apply a cartoon color per residue
        view.addStyle(sel, {'cartoon': {'color': hex_colors[i]}})

    # Highlighted sites: draw EITHER spheres or sidechains
    if highlighted_sites is not None:
        hlist = np.asarray(highlighted_sites).astype(int).ravel().tolist()
        for pos in hlist:
            idx = int(pos) - index_base
            if idx < 0 or idx >= n_res:
                print(f"warning: highlighted site {pos} (index_base={index_base}) out of range -> skipped")
                continue
    
            res = residues[idx]
            x, y, z = res['ca']
            sel = {'chain': res['chain'], 'resi': str(res['resSeq'])}
    
            if highlight_style == 'sphere':
                view.addSphere({
                    'center': {'x': x, 'y': y, 'z': z},
                    'radius': sphere_radius,
                    'color': sphere_color,
                    'opacity': 1.0
                })
            elif highlight_style == 'stick':
                view.addStyle(sel, {'stick': {'radius': 0.3}, 'sphere': {'scale': 0.3}})
            else:
                raise ValueError(f"Unknown highlight_style '{highlight_style}'. Use 'sphere' or 'stick'.")
    # final touches
    view.setBackgroundColor('0xFFFFFF')  # white background
    return view


def mean_contact_frustration_heatmap(pdb_path, df_contact, fold_name = None, path = None, replace_fig = False, vmin_c = None, vmax_c = None, 
                                     dca_awsem = "dca", mask_value = 'awsem'):
    fig, ax = plt.subplots(figsize=(8, 8))
    structure, mask_full, mask_awsem = cazar.cargar_est_ref(pdb_path)
    if dca_awsem == "dca":
        df = pd.DataFrame(df_contact["Frustracion_contacto_media_DCA"][0])
        pair_values_list = df_contact["Frustracion_contacto_media_DCA"][0][mask_awsem]
        if mask_value == 'dca':
            pair_values_list = df_contact["Frustracion_contacto_media_DCA"][0][mask_full] #Incluso podría hacerme una opción para que pasarle manualmente una 
        N = len(pair_values_list)                                                         #mascara
        color_scale = ['green', 'white', 'red']
#        cmap = colors.LinearSegmentedColormap.from_list('my_cmap', color_scale, N = N)

    elif dca_awsem == "awsem":
        df = pd.DataFrame(df_contact["Frustracion_contacto_media_AWSEM"][0])
        pair_values_list = df_contact["Frustracion_contacto_media_AWSEM"][0][mask_awsem]
        if mask_value == 'dca':
            pair_values_list = df_contact["Frustracion_contacto_media_AWSEM"][0][mask_full]
        N = len(pair_values_list)
        color_scale = ['green', 'white', 'red']
#        cmap = colors.LinearSegmentedColormap.from_list('my_cmap', color_scale, N = N)

    if (vmin_c, vmax_c) == (None, None):
        if dca_awsem == "dca":
            (vmin_c, vmax_c) = (np.min(pair_values_list), np.max(pair_values_list))
        elif dca_awsem == "awsem":
            (vmin_c, vmax_c) = (-0.78, 1)


    norm = colors.TwoSlopeNorm(vmin=vmin_c, vcenter=0, vmax=vmax_c)
    cmap = colors.LinearSegmentedColormap.from_list('my_cmap', color_scale, N = N)
    sns.heatmap(df.mask(~mask_awsem, np.nan), cmap = cmap, norm=norm)

    plt.title(f"Heatmap DCA contact frustration media masked", fontsize=18)
    if dca_awsem == "awsem":
        plt.title(f"Heatmap AWSEM contact frustration media masked", fontsize=18)
    
    if fold_name is not None:
        if dca_awsem == "dca":
            plt.title(f"Heatmap DCA contact frustration media masked - {fold_name}", fontsize=18)
        elif dca_awsem == "awsem":
            plt.title(f"Heatmap AWSEM contact frustration media masked - {fold_name}", fontsize=18)
    
    plt.xlabel('Position', fontsize=25)
    plt.ylabel('Position', fontsize=25)
    x=np.arange(5, len(df)-5, 5)
    xticks=np.arange(5, len(df)+1-5, 5)
    plt.xticks(x, xticks, fontsize = 16)
    plt.yticks(x, xticks, fontsize = 16)
    fig.set_facecolor('white')
#ax.set_facecolor('white')
    
    if path is not None and (not os.path.exists(os.path.join(path, "heatmap_fc_mean_masked.pdf")) or replace_fig == True):
        if dca_awsem == "dca":
            plt.savefig(f"{path}/heatmap_fc_dca_mean_masked.pdf", format="pdf", dpi=300, bbox_inches="tight")
            plt.savefig(f"{path}/heatmap_fc_dca_mean_masked.png", format="png", dpi=300, bbox_inches="tight")
        elif dca_awsem == "awsem":
            plt.savefig(f"{path}/heatmap_fc_awsem_mean_masked.pdf", format="pdf", dpi=300, bbox_inches="tight")
            plt.savefig(f"{path}/heatmap_fc_awsem_mean_masked.png", format="png", dpi=300, bbox_inches="tight")


def desvio_contact_frustration_heatmap(pdb_path, df_contact, 
                                       fold_name = None, path = None, replace_fig = False, vmin_c = 0.0, vmax_c = 2.0, center_c = 0.771,
                                      dca_awsem = "dca"):
                                        #ese center es la mediana de la distribución de desvios por posición para varias familias 
                                        #y vmin y vmax son el primer y el tercer cuartil de esa distribución respectivamente
    plt.figure(figsize=(8, 8))
    structure, mask_full, mask_awsem = cazar.cargar_est_ref(pdb_path)
    if dca_awsem == "dca":
        
        dfsd = pd.DataFrame(df_contact["Frustracion_contacto_desvio_DCA"][0])
        N = len((df_contact["Frustracion_contacto_desvio_DCA"][0])[mask_awsem])

    elif dca_awsem == "awsem":
        dfsd = pd.DataFrame(df_contact["Frustracion_contacto_desvio_AWSEM"][0])
        N = len((df_contact["Frustracion_contacto_desvio_AWSEM"][0])[mask_awsem])
        

    sns.heatmap(dfsd.mask(~mask_awsem, np.nan), cmap = 'viridis', vmin = vmin_c, vmax = vmax_c, center = center_c)

    if dca_awsem == "dca":
        plt.title(f"Heatmap DCA contact frustration desvio masked", fontsize=18)
        
        if fold_name is not None:
            plt.title(f"Heatmap DCA contact frustration desvio masked - {fold_name}", fontsize=18)

    elif dca_awsem == "awsem":
        plt.title(f"Heatmap AWSEM contact frustration desvio masked", fontsize=18)
        
        if fold_name is not None:
            plt.title(f"Heatmap AWSEM contact frustration desvio masked - {fold_name}", fontsize=18)

    
    plt.xlabel('Position', fontsize=25)
    plt.ylabel('Position', fontsize=25)
    x=np.arange(5, len(dfsd)-5, 5)
    xticks=np.arange(5, len(dfsd)+1-5, 5)
    plt.xticks(x, xticks, fontsize = 16)
    plt.yticks(x, xticks, fontsize = 16)

    if dca_awsem == "dca":
        if path is not None and (not os.path.exists(os.path.join(path, "heatmap_fc_dca_desvio_masked.pdf")) or replace_fig == True):
            plt.savefig(f"{path}/heatmap_fc_dca_desvio_masked.pdf", format="pdf", dpi=300, bbox_inches="tight")
            plt.savefig(f"{path}/heatmap_fc_dca_desvio_masked.png", format="png", dpi=300, bbox_inches="tight")
    elif dca_awsem == "awsem":
        if path is not None and (not os.path.exists(os.path.join(path, "heatmap_fc_awsem_desvio_masked.pdf")) or replace_fig == True):
            plt.savefig(f"{path}/heatmap_fc_awsem_desvio_masked.pdf", format="pdf", dpi=300, bbox_inches="tight")
            plt.savefig(f"{path}/heatmap_fc_awsem_desvio_masked.png", format="png", dpi=300, bbox_inches="tight")
