Here are files accompanying the paper:

Building an ACT-R reader for eye-tracking corpus data

The paper has been/is to be published in Topics in Cognitive Science

Three csv files:

1. The file materials_final.csv consists of cleaned materials from the GECO corpus (just materials, no eye-tracking measures appear there)

2. The file summed_eyetracking_parameter_search.csv consists of  data used for paramater searching. It includes relevant eye-tracking measures.

3. The file summed_eyetracking_not_search.csv consists of data used for checking predictions of the Basic ACT-R Reader model. It also includes relevant eye-tracking measures.

The most important files are two Python scripts. They only work in Python3 (3.3 or higher). For both files, you need to install pyactr. This can be done using pip (see also https://github.com/jakdot/pyactr). pyactr has to be at least of version 0.2.1.

Python files:

parametersearch_MPI.py: This searches for parameter values in the data set summed_eyetracking_parameter_search.csv using the Metropolis algorithm. You will need to use Message Passing Interface, MPI. This is needed because the search is done in parallel computation and different processes have to pass information (vectors, texts). You have to start the file using MPI:

mpirun -n 2 python3 parametersearch_MPI.py

(where -n signals how many cores are used; at least 2 cores have to be used -- 1 core for Bayesian model, one core for ACT-R model; in practice, it does not make sense to run simulation with less than 100 cores; even with 100 cores 1000 draws will take around 100 hours)

After the sampling is done, the file outputs a csv file with samples and a traceplot and it prints an abbreviated traceframe in the console.

checkfound_MPI.py: This simulates reading on the data set summed_eyetracking_not_search.csv. It uses parameter values found independently using parametersearch_MPI.py. The values that are currently manually inserted in this file come from the simulation described in the paper ``Building an ACT-R reader for eye-tracking corpus data''.

This file also requires MPI. You can run it as:

mpirun -n 2 python3 checkfound_MPI.py

Unlike the previous Python file, this one is less computation-intensive. It might be possible to run it with only a few cores, maybe even just 2. It will output a csv file with simulated eye-fixation times per word.
