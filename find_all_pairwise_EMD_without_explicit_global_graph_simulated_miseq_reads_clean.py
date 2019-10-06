from Bio import SeqIO
from de_bruijn_graph import DeBruijnGraph
import networkx as nx
from networkx.relabel import convert_node_labels_to_integers
import pyemd
import numpy as np
import sys,os
import ntpath
import concurrent.futures
import time
import itertools
import pickle
from multiprocessing import Pool
from contextlib import closing
import itertools  
import networkx.algorithms.shortest_paths.unweighted
#just need this for fast testing of small instances
from networkx.algorithms.shortest_paths.dense import floyd_warshall_numpy
import multiprocessing.pool
import time
from multiprocessing import Process, Manager
from random import randint
from time import sleep

def get_numbers(f):
    with open(f) as input_file:
        for line in input_file:
            line = line.strip()
            for number in line.split():
                yield float(number)

current_milli_time = lambda: int(round(time.time()))

class ExceptionWrapper(object):

    def __init__(self, ee):
        self.ee = ee
        __,  __, self.tb = sys.exc_info()

    def re_raise(self):
        raise (self.ee, None, self.tb)

class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NoDaemonPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))

def count_kmers(read, k):
    counts = {}
    num_kmers = len(read) - k + 1
    for i in range(num_kmers):
        kmer = read[i:i+k]
        if kmer not in counts:
            counts[kmer] = 0
        counts[kmer] += 1
    return counts

def build_graph_for_all_hosts(in_path_to_outbreaks_root, in_output_path, k, all_files, all_hosts):
    all_kmers_str = []
    preprocessing_dir = in_output_path + str(k) + '/preprocessing/'
    if os.path.isfile(in_output_path + str(k) + '/preprocessing/all_kmers_k' + str(k) + '.npy'):
        print('K-mers alredy available at ' + in_output_path + str(k) + '/preprocessing/all_kmers_k' + str(k) + '.npy')
        all_kmers_np = np.load(in_output_path + str(k) + '/preprocessing/all_kmers_k' + str(k) + '.npy')
        all_kmers_str = all_kmers_np.tolist()
    else:
        if not os.path.exists(preprocessing_dir):
            os.makedirs(preprocessing_dir)

        print("Getting kmers for all files...")
        kmers_start_time = current_milli_time()
        kmers_files = {}

        if os.path.isfile(in_output_path + str(k) + '/all_files_kmers_dictionary.pkl'):
            print("Kmers for all files are already available. Reading file...")
            kmers_files = load_obj(in_output_path + str(k) + '/all_files_kmers_dictionary')
            print("Done.")
        else:
            __kmers = get_kmers_for_all_files(all_files, all_hosts, k)
            kmers_files = __kmers[0]
            print("Done.")
            print("Saving kmers for all files to disk...")
            save_obj(kmers_files, in_output_path + str(k) + '/all_files_kmers_dictionary')
            print("Done.")

        all_kmers = __kmers[1]

        print("Getting kmers for all files... Done. Took %d seconds." % (current_milli_time() - kmers_start_time))
        print("Total number of kmers for k = %d is %d." %(k, len(all_kmers)))
        all_kmers_str = [str(kmer) for kmer in all_kmers]

    print("Done getting kmers for all files...")
    np.save(preprocessing_dir + '/all_kmers_k' + str(k) + '.npy', np.array(all_kmers_str))

    distances = []
    dbg_node_labels = []

    if os.path.isfile(preprocessing_dir + '/all_distances_k' + str(k) + '.npy'):
        print(preprocessing_dir + '/all_distances_k' + str(k) + '.npy')
        distances = np.load(preprocessing_dir + '/all_distances_k' + str(k) + '.npy')
        print('Nodes alredy available at ' + preprocessing_dir + '/all_distances_k' + str(k) + '.npy')
        dbg_node_labels_np = np.load(preprocessing_dir + '/all_distances_k' + str(k) + '.npy')
        dbg_node_labels = dbg_node_labels_np.tolist()
    else:
        dbg = lambda: None
        if os.path.isfile(in_output_path + str(k) + '/graph_' + str(k) + '.adjlist'):
            print('De Bruijn graph already available. Reading from disk...')
            G = nx.read_adjlist(in_output_path + str(k) + '/graph_' + str(k) + '.adjlist')
            dbg.G = G
            print('Done.')
        else:
            print("De Bruijn graph not available. Building...")
            graph_start_time = current_milli_time()
            dbg = DeBruijnGraph(all_kmers_str, k + 1)
            nx.write_adjlist(dbg.G,in_output_path + str(k) + '/graph_' + str(k) + '.adjlist')
            print("Building De Bruijn graph... Done. Took %d seconds." % (current_milli_time() - graph_start_time))

    return(dbg.G)


def process_global_distance_matrix_entry(in_data):
    i = in_data[0]
    j = in_data[1]
    dbg_node_labels = in_data[2]
    kmers_in_current_origin = in_data[3]
    G = in_data[4]
    return nx.shortest_path_length(G, dbg_node_labels[i], kmers_in_current_origin[j])

# Returns list of tuples (pair, kmers_1, kmers_2, frequency_distribution_1, frequency_distribution_2)
def get_emd_inputs_for_current_pairs(current_origin_host, current_pairs, kmers_files, k, G, in_path_to_preprocessed_dir, in_path_to_results_output, build_partial_distance_matrix=False):
    distance_matrix_for_current_file_and_the_rest = []
    if(build_partial_distance_matrix):
        if (os.path.isfile(in_path_to_preprocessed_dir + '/partial_distance_matrix_for_origin_' + os.path.basename(current_origin_host) + '.npy')):
            distance_matrix_for_current_file_and_the_rest = np.load(in_path_to_preprocessed_dir + '/partial_distance_matrix_for_origin_' + os.path.basename(current_origin_host) + '.npy')
        else:
            kmers_in_current_origin = [str(i) for i in kmers_files[current_origin_host].keys()] 
            dbg_nodes = G.nodes(data='label')
            dbg_node_labels = []
            for node in dbg_nodes:
                dbg_node_labels.append(node[0])

            partial_distance_matrix = [[0 for i in range(len(dbg_node_labels))] for j in range(len(kmers_in_current_origin))]
    
            print("New distance matrix height is %d" % (len(partial_distance_matrix)))
            print("New distance matrix width is %d" % (len(partial_distance_matrix[0])))

            inputs_list = []

            for i in range(len(partial_distance_matrix[0])):
                for j in range(len(partial_distance_matrix)):
                    inputs_list.append((i, j, dbg_node_labels, kmers_in_current_origin, G))

            matrix_init_start_time = current_milli_time()
            print("Filling new distance matrix...")
            pool = multiprocessing.Pool(100)
            partial_distance_matrix_1d = pool.map(process_global_distance_matrix_entry, inputs_list)
            pool.close()
            pool.join()
            print("Filling new distance matrix... Done. Took %d seconds." % (current_milli_time() - matrix_init_start_time))
            print(len(partial_distance_matrix_1d))

            partial_distance_matrix_np = np.reshape(partial_distance_matrix_1d, (len(partial_distance_matrix), len(partial_distance_matrix[0])))
            np.save(in_path_to_preprocessed_dir + '/partial_distance_matrix_for_origin_' + os.path.basename(current_origin_host) + '.npy', np.array(partial_distance_matrix_np))
            distance_matrix_for_current_file_and_the_rest = partial_distance_matrix_np

    out = []
    current_pair_idx = 0

    clean_kmers = []
    for pair in current_pairs:
        current_pair_idx = current_pair_idx + 1
        out.append((pair, kmers_files, k, G, current_origin_host, distance_matrix_for_current_file_and_the_rest, build_partial_distance_matrix, clean_kmers, in_path_to_results_output))
    return out

# Returns dict {filename : kmer_counts}
def get_kmers_for_all_files(all_files, all_hosts, k):
    kmers_for_files_dict = {}
    kmers_sequences_only_for_de_bruijn = []
    read_files_for_hosts = dict()

    for host in all_hosts:
        read_files_for_hosts[host] = []
        for file in all_files:
            if (host.replace('__', '_') in file):
                 read_files_for_hosts[host].append(file)

    ## Don't need this for 1 pair test     
    for host, read_files_for_host in read_files_for_hosts.items():
        if (len(read_files_for_host) == 0):
            print("Error: no read files for host %s" % (host))
            sys.exit()

    get_kmers_for_all_files_parallel_inputs = []
    for host in all_hosts:
        get_kmers_for_all_files_parallel_inputs.append((host, read_files_for_hosts[host]))
    
    with closing(NoDaemonPool(processes=100)) as pool:
        kmers_with_counts = pool.map(get_file_kmers_with_counts, get_kmers_for_all_files_parallel_inputs)
        pool.terminate()
        for host, kmers, current_kmers_sequences_only_for_de_bruijn in kmers_with_counts: 
            kmers_for_files_dict.setdefault(host, []).append(kmers)
            kmers_sequences_only_for_de_bruijn.extend(current_kmers_sequences_only_for_de_bruijn) 
    
    return (kmers_for_files_dict, kmers_sequences_only_for_de_bruijn)

def get_file_kmers_with_counts(inputs):
    host = inputs[0]
    read_files_for_host = inputs[1]
    kmers_for_host = {}
    read_file_idx = 0
    _kmers_sequences_only_for_de_bruijn = []
    for host_reads_file in read_files_for_host:
            read_file_idx = read_file_idx + 1
            print('Getting kmers for host %s (read file %d out of %d).' % (host, read_file_idx, len(read_files_for_host)))
            sequences = SeqIO.parse(open(host_reads_file), 'fasta')
            sequence_idx = 0
            for sequence in sequences:
                sequence_idx = sequence_idx + 1
                print('Getting kmers for host %s (read file %d out of %d (sequence %d out of %d))' % (host, read_file_idx, len(read_files_for_host), sequence_idx, len(list(sequences))))
                current_seq_counts = count_kmers(sequence.seq, k)
                for kmer, count in current_seq_counts.items():
                    if kmer in kmers_for_host.keys():
                        kmers_for_host[kmer] = kmers_for_host[kmer] + count
                    else:
                        kmers_for_host[kmer] = count
                current_seq_counts_for_de_bruijn = count_kmers(sequence.seq, k + 1)
                for kmer, count in current_seq_counts_for_de_bruijn.items():
                    if kmer not in _kmers_sequences_only_for_de_bruijn:
                        _kmers_sequences_only_for_de_bruijn.append(str(kmer))
            read_file_idx = read_file_idx + 1
                    
    return (host, kmers_for_host, _kmers_sequences_only_for_de_bruijn)



def process_local_distance_matrix_entry(in_data):
    i = in_data[0]
    j = in_data[1]
    if (i == -1 and j == -1):
        return 0
    else:
        kmers_union = in_data[2]
        G = in_data[3]
        return nx.shortest_path_length(G, kmers_union[i], kmers_union[j])

def find_emd_between_2_hosts(in_data):
    path_to_first_fasta = in_data[0][0]
    path_to_second_fasta = in_data[0][1]
    kmers_files = in_data[1]
    k = in_data[2]
    G = in_data[3]
    current_origin_host = in_data[4]
    distance_matrix_for_current_file_and_the_rest = in_data[5]
    use_partial_distance_matrix = in_data[6]
    clean_kmers = in_data[7]
    out_results_root = in_data[8]

    dbg_nodes = G.nodes(data='label')

    dbg_node_labels = []
    for node in dbg_nodes:
       dbg_node_labels.append(node[0])

    print("Getting kmers for file (first) %s"  % (path_to_first_fasta))

    first_file_kmers = kmers_files[path_to_first_fasta]
    second_file_kmers = kmers_files[path_to_second_fasta]

    kmers_in_first = [str(i) for i in first_file_kmers[0].keys()]
    kmers_in_second = [str(j) for j in second_file_kmers[0].keys()]

    frequencies_first = first_file_kmers[0].values()
    frequencies_second = second_file_kmers[0].values()

    print("Calculating EMD for pair %s and %s..." % (path_to_first_fasta, path_to_second_fasta))
    id_a = os.path.basename(path_to_first_fasta)
    id_b = os.path.basename(path_to_second_fasta)
    outfile = out_results_root + id_a + '_' + id_b + 'emd.txt'
    if (os.path.isfile(outfile)):
        print("Results for current pair are already available. Skipping...")
        return
    else:
        kmers_union = union(kmers_in_first, kmers_in_second)
        frequencies_first_for_both = []
        frequencies_second_for_both = []

        for kmer in kmers_union:
            if kmer in kmers_in_first:
                frequencies_first_for_both.append(frequencies_first[kmers_in_first.index(kmer)])
            else:
                frequencies_first_for_both.append(0)

        for kmer in kmers_union:
            if kmer in kmers_in_second:
                frequencies_second_for_both.append(frequencies_second[kmers_in_second.index(kmer)])
            else:
                frequencies_second_for_both.append(0)

        pair_distance_matrix = [[0 for i in range(len(kmers_union))] for j in range(len(kmers_union))]

        print("Filling new distance matrix (non-parallel) ...")
        matrix_init_start_time = current_milli_time()
        # initialize distance matrix
        for i in range(len(kmers_union)):
            for j in range(len(kmers_union)):
                if i == j:
                    pair_distance_matrix[i][j] = 0
                elif (pair_distance_matrix[i][j] == 0 and pair_distance_matrix[j][i] == 0):
                    current_dist = nx.shortest_path_length(G, kmers_union[i], kmers_union[j])
                    pair_distance_matrix[i][j] = current_dist
                    pair_distance_matrix[j][i] = current_dist

        
        print("Filling new distance matrix... Done. Took %d seconds." % (current_milli_time() - matrix_init_start_time))

        pair_distance_matrix_np = np.asarray(pair_distance_matrix)

        print("Starting EMD computation...")

        frequencies_first_norm = [float(i)/sum(frequencies_first_for_both) for i in frequencies_first_for_both]
        frequencies_second_norm = [float(i)/sum(frequencies_second_for_both) for i in frequencies_second_for_both]

        print(len(frequencies_first_norm))
        print(len(frequencies_second_norm))
        print(len(pair_distance_matrix))

        result_2 = pyemd.emd(np.asarray(frequencies_first_norm), np.asarray(frequencies_second_norm), pair_distance_matrix_np.astype(float), extra_mass_penalty=0.0)
       
        print("################################################## Calculated emd for pair %s and %s: (pyemd lib) ##################################################" % (id_a, id_b))
        print(result_2)

        if result_2 == 0.0:
            print(frequencies_first)
            print(frequencies_second)
            print(kmers_in_first)
            print(kmers_in_second)


        outf= open(outfile,"w+")
        outf.write(str(result_2))
        outf.close()

def process_chunk_concurrently(in_chunk):
    with closing(NoDaemonPool(processes=1)) as pool:
        print(pool.map(find_emd_between_2_hosts, in_chunk))
        pool.terminate()


if __name__ == '__main__':
    k = 5

    all_files = []
    all_hosts = []

    path_to_reads = sys.argv[1]
    path_to_output = sys.argv[2]

    preprocessing_dir = path_to_output + str(k) + '/preprocessing/'

    if not os.path.exists(preprocessing_dir):
            os.makedirs(preprocessing_dir)

    print("Getting all files...")
    for path, subdirs, files in os.walk(path_to_reads):
        for name in files:
            all_files.append(os.path.join(path, name))
    
    print("Getting all hosts...")
    for path in all_files:
        all_hosts.append(path.split('host')[0])

    print("Total number of read files is %d" % (len(all_files)))
    print("Total number of read hosts is %d" % (len(all_hosts)))
    
    G = build_graph_for_all_hosts(path_to_reads, path_to_output, k, all_files, all_hosts)

    print("Getting all pairs...")
    all_pairs = list(itertools.combinations(all_hosts, 2))
    print("Total number of pairs is %d" % (len(all_pairs)))

    print("Removing already computed pairs from all pairs list...")

    current_out_dir = path_to_output + str(k) + '/results/'
    if not os.path.exists(current_out_dir):
        os.makedirs(current_out_dir)

    all_pairs_updated = []
    for p in all_pairs:
        f1 = current_out_dir + p[0] + '_' + p[1] + '_emd.txt'
        f2 = current_out_dir + p[1] + '_' + p[0] + '_emd.txt'
        if os.path.isfile(f1):
            if (list(get_numbers(f1))[0] == 0.0):
                all_pairs_updated.append(p)
        elif os.path.isfile(f2):
            if (list(get_numbers(f2))[0] == 0.0):
                all_pairs_updated.append(p)
        else:
            all_pairs_updated.append(p)

    print("Removing already computed pairs from all pairs list... Done.")
    print("Total number of pairs is %d" % (len(all_pairs_updated)))

    print("Getting pre-computed kmers with counts...")

    kmers_files = {}

    if os.path.isfile(path_to_output + str(k) + '/all_files_kmers_dictionary.pkl'):
        print("Kmers for all files are already available. Reading file...")
        kmers_files = load_obj(path_to_output + str(k) + '/all_files_kmers_dictionary')
        print("Done.")
    else:
        kmers_files = get_kmers_for_all_files(all_files, all_hosts, k)
        print("Done.")
        print("Saving kmers for all files to disk...")
        save_obj(kmers_files, path_to_output + str(k) + '/all_files_kmers_dictionary')
        print("Done.")

    path_to_preprocessed_dir = path_to_output + str(k) + '/preprocessing/'
    path_to_results_output = path_to_output + str(k) + '/results/'
    emd_inputs = get_emd_inputs_for_current_pairs(all_pairs_updated, all_pairs_updated, kmers_files, k, G, path_to_preprocessed_dir, path_to_results_output)
    print("Calculating EMD for all pairs...")
    emd_start_time = current_milli_time()

    bComputeEMDInParrallel = True
    print(len(emd_inputs))

    if (bComputeEMDInParrallel):
        with closing(NoDaemonPool(processes=100)) as pool:
            results = pool.map(find_emd_between_2_hosts, emd_inputs)
            pool.terminate()
            for result in results:
                if isinstance(result, ExceptionWrapper):
                    result.re_raise()
    else:
        for emd_input in emd_inputs:
            find_emd_between_2_hosts(emd_inputs)

    print("Calculating EMD for all pairs. Done.")