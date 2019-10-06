# kemd
##Installation
 Sript requires the following dependencies:
 
 * python 2.7.12
 * networkx 2.1
 * pyemd 0.5.1
 * numpy 1.15.1
 * biopython 1.72
 * DeNovoAssembly

All python modules, except DeNovoAssembly can be installed through pip. DeNovoAssembly can be downloaded from https://github.com/guojingyu/DeNovoAssembly
For convenience, de_bruijn_graph.py, eulerian.py and utils.py scripts of DeNovoAssembly library can be copied into kemd directory.

## Running the script
Once installed, algorithm can be executed using the following command:

~~~
python find_all_pairwise_EMD_without_explicit_global_graph_simulated_miseq_reads_clean.py ./test_data ./test_output
~~~

Example of test input can be found in test_data directory.
