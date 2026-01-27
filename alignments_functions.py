import numpy as np
from scipy.spatial.distance import pdist,squareform
from Bio import SeqIO
from Bio.Align import substitution_matrices
import dca_functions
from numba import jit
from numba import prange as parallel_range


'''
Define some dictionaries to work with protein sequences. 
AA_dict_full translates from one letter code to number code to work in python. Every character that is not a residue (X, Z, B) translate into
the number 0 corresponding to the gap "-" character. 

letters_dict is the inverse dictionary, translates from number code to one letter code, with 0 becoming the gap "-" character. 
'''
AA_dict_full={'X':0,'Z':0,'B':0, '-':0,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,
                  'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}

letters_dict = {value: key for key, value in AA_dict_full.items()}

def load_msa(msa_path, file_format = 'fasta'):
    '''
    Parameters
    ----------
    msa_path : path of the file wich contains a multiple sequences alignment (or a multifasta file)

    file_format : extension of the MSA. fasta by default 
    
    Returns
    seqs : list of the sequence in the MSA, stored as biopython seq objects
    names: 
    -------
    '''
    fasta_sequences = list(SeqIO.parse(msa_path, file_format))
    seqs=[]
    names=[]
    for j in range(len(fasta_sequences)):
        names.append(fasta_sequences[j].id)
        seqs.append(fasta_sequences[j].seq)
    return seqs, names

def seq_to_numpy(seq):
    seq_np = np.ones(len(seq))
    for i in range(len(seq)):
        a = int(AA_dict_full[seq[i]])
        seq_np[i] = a
    return seq_np

def MSA_to_numpy(MSA):
    #un MSA guardado en un array de numpy con los aa en formato unicode, te lo transforma en un array de numpy de iguales dimensiones pero con numeritos
    MSA_np = np.zeros([MSA.shape[0],MSA.shape[1]], dtype=np.int64)
    MSA_np.shape
    for seq in range(MSA.shape[0]):
        for i in range (MSA.shape[1]):
            a = int(AA_dict_full[MSA[seq,i]])
            MSA_np[seq,i] = a
    return MSA_np

def numpy_to_MSA(MSA_numeric):
    # hace lo opuesto a MSA_to_numpy: convierte índices numéricos en letras
    MSA_letters = np.empty(MSA_numeric.shape, dtype='<U1')  # array de unicode 1 caracter
    for seq in range(MSA_numeric.shape[0]):
        for i in range(MSA_numeric.shape[1]):
            MSA_letters[seq, i] = letters_dict[int(MSA_numeric[seq, i])]
    return MSA_letters

def write_fasta(seqs, names, outfile_path):
    ofile = open(outfile_path, "w")
    
    for j in range(len(seqs)):
        seq_j = "".join(np.array(seqs[j]).tolist())
        ofile.write(">" + names[j] + "\n" + seq_j + "\n")
    
    ofile.close()  

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


def get_eff(msa,eff_cutoff=0.9):
  '''compute effective weight for each sequence'''
  ncol = msa.shape[1]

  # pairwise identity
  msa_sm = 1.0 - squareform(pdist(msa,"hamming"))

  # weight for each sequence
  msa_w = (msa_sm >= eff_cutoff).astype(float)
  msa_w = 1/np.sum(msa_w,-1)
  return msa_w

@jit(nopython=True, parallel=True)
def compute_sequences_weight_for_many_sequences(alignment_data=None, seqid=0.9):
    """Computes weight of sequences. The weights are calculated by lumping
    together sequences whose identity is greater that a particular threshold.
    For example, if there are m similar sequences, each of them will be assigned
    a weight of 1/m. Note that the effective number of sequences is the sum of
    these weights.

    Parameters
    ----------
        alignmnet_data : np.array()
            Numpy 2d array of the alignment data, after the alignment is put in
            integer representation
        seqid : float
            Value at which beyond this sequences are considered similar. Typical
            values could be 0.7, 0.8, 0.9 and so on

    Returns
    -------
        seqs_weight : np.array()
            A 1d numpy array containing computed weights. This array has a size
            of the number of sequences in the alignment data.
    """
    alignment_shape = alignment_data.shape
    num_seqs = alignment_shape[0]
    seqs_len = alignment_shape[1]
    seqs_weight = np.zeros((num_seqs,), dtype=np.float64)
    #count similar sequences
    for i in parallel_range(num_seqs):
        seq_i = alignment_data[i]
        for j in range(num_seqs):
            seq_j = alignment_data[j]
            iid = np.sum(seq_i==seq_j)
            if np.float64(iid)/np.float64(seqs_len) > seqid:
                seqs_weight[i] += 1
    #compute the weight of each sfreq(ali,npos,Naa,w)equence in the alignment
    for i in range(num_seqs): seqs_weight[i] = 1.0/float(seqs_weight[i])
    return seqs_weight


def freq(MSA,states,w): #npos es el largo de cada secuencia del alineamiento
    
    """
    Con esta de acá abajo a calcular frecuencias de uno y dos cuerpos para un alineamiento en principio de proteínas, 
    es perfectamente usable para alineamientos de ARN (de ADN no existen alineamientos?).

    Parámetros
    -----------
    MSA: un alineamiento de secuencias, en formato Numpy array, en algún código numérico como el definido por el diccionario de este archivo, pdict. 
####npos: el largo de cada secuencia del alineamiento, es decir la dimensión ali.shape[1]. --> ¿Podríamos de hecho hacer eso y que no necesite este parámetro?
    states: la cantidad de estados posibles para cada posición del alineamiento. Proteínas = 21 (20 aa + gap). ARN = 5 (4 b + gap)
    w: el peso ¿estadístico? de cada secuencia en el alineamiento. Diferentes formas de definirlo en ppo, la librería pydca obtiene uno viendo similaritud de
    seqs.

    Returns
    -----------
    (fij, fi): las frecuencias de dos y de un cuerpo respectivamente de los estados (aminoácidos x ej) en cada posición del alineamiento.
    
    """

    npos = MSA.shape[1]
    
    fij=np.zeros((npos,npos,states,states)) 
    fi=np.zeros((npos,states)) 
    for i in range(npos):
        fi[i,:]=np.histogram(MSA[:,i],bins=np.arange(-0.5,states+0.5),weights=w)[0]
        for j in range(npos):
            #if j!=i:
             fij[i,j,:,:]=np.histogram2d(MSA[:,i],MSA[:,j],bins=np.arange(-0.5,states+0.5),weights=w)[0]
    fi=fi/sum(w)
    fij=fij/sum(w)

    return fij,fi

#Estas de blosum están super mal


"""def blosum_distance_norm(seq1, seq2, substitution_matrix): #tendría que especificar que matrices acepto? ni loco no? 
    assert len(seq1) == len(seq2)
    
    mat = substitution_matrices.load(substitution_matrix)

    return sum(mat[a,a] - blosum_score(a, b, mat) for a, b in zip(seq1, seq2))"""


def blosum_score(a, b, blosum_matrix):
    key = (a, b)
    if key in blosum_matrix:
        return blosum_matrix[key]
    key = (b, a)
    if key in blosum_matrix:
        return blosum_matrix[key]
    return 0

def blosum_distance_norm(seq1, seq2, blosum_matrix): #tendría que especificar que matrices acepto? ni loco no? 
    assert len(seq1) == len(seq2)
    
#    mat = substitution_matrices.load(substitution_matrix)

    return sum(blosum_matrix[a,a] - blosum_score(a, b, blosum_matrix) for a, b in zip(seq1, seq2))

def blosum_distance_matrix(sequences, substitution_matrix = "BLOSUM62"):

    blosum_matrix = substitution_matrices.load(substitution_matrix)

    distance_blosum = np.zeros((len(sequences), len(sequences))) 
    for i in range(len(sequences) - 1):
        for j in range(i+1, len(sequences)):
            distance_blosum[i,j] = blosum_distance_norm(sequences[i], sequences[j], blosum_matrix)
    distance_blosum = distance_blosum + distance_blosum.T
    return distance_blosum

def distance_in_hi (sequences, potts_model_path, potts_file_format = "npz"):
    hi_rbm = dca_functions.load_potts(potts_model_path, potts_file_format)["h"]

    hi_RBM_by_seq_matrix = np.zeros(sequences.shape)
    for i in range(sequences.shape[0]): #cada seq
        for j in range(sequences.shape[1]): #cada pos del ali
            hi_RBM_by_seq_matrix[i,j] = hi_rbm[j, int(sequences[i,j])] #la posicion j de la secuencia, el aminoacido que esta en la seq i y la pos j

    distance_in_hi = np.zeros((len(sequences), len(sequences)))
    for i in range(len(sequences) - 1):
        for j in range(i+1, len(sequences)):
            distance_in_hi[i,j] = np.sqrt(np.mean((hi_RBM_by_seq_matrix[i] - hi_RBM_by_seq_matrix[j])**2))

    distance_in_hi = distance_in_hi + distance_in_hi.T

    return distance_in_hi