# Docker Cluster Results
For our container-based clusters that use one container per node. The image for our container can be found on Docker hub: sadeghil/cryptmpich:onbuild-coll-v15 
https://hub.docker.com/r/sadeghil/cryptmpich/tags (Its tag is onbuild-coll-v15)


To manage Docker Swarm cluster, we have used following git repo: 
https://github.com/NLKNguyen/alpine-mpich (In master branch and without any tag)

To reproduce the Docker Cluster results, one needs to log into one of the containers, and then set "LD_LIBRARY_PATH" with the following command:

```export LD_LIBRARY_PATH=/home/mpi/boringssl/build/crypto:/project/cryptMPI-mpich-3.3/Install_MPICH/lib/```

Then, one run the full benchmarks with the provided script under the following directory:

```cluster/project/benchmarks```

We have also provided the batch scripts containing the instructions to run OSU benchmarks in MPICH_Bench_Script.tar.gz.

# Noleland results

We have provided all of the scripts for running the OSU benchmarks on the Noleland cluster at MVAPICH-Scripts.tar. 

# N-Body-Using-MPI

In physics an N-body is a simulation of a dynamical system of particles, usually under the influence of physical forces, such as gravity.

For more details about n-body simulations on Wikipedia: https://en.wikipedia.org/wiki/N-body_simulation

This Benchmark originally developed by this git repo: https://github.com/prashantmishra/n_body_mpi/blob/master/simulation.c
We have provided the modified code that measures the communication time too.

## Installation
To install modify MPI_INSTALLITION_DIR in Makefile and make it.

```bash
MPICC=/MPI_INSTALLITION_DIR/bin/mpicc
CCFLAGS= -O2 -lm

nbody: nbody.c
	$(MPICC) $(CCFLAGS) nbody.c -o nbody

clean:
	rm -f *.o nbody *~
```

For running this application, we added 1000 bodies with their initial x and y position and their mass given in initial_state.txt

## Running

A sample input file (initial_state.txt) is provided. One needs to fix the size of the input and use the provided script (batch.sh) inside the nbody.tar archive to reproduce the results of NBody in our paper.





# NAS Parallel Benchmarks 3.3.1

The NAS Parallel Benchmarks (NPB) are a small set of programs designed to help evaluate the performance of parallel supercomputers.

In this project we have used FT benchmark to evalute MPI_Alltoall performance.

FT: discrete 3D fast Fourier Transform, all-to-all communication


## Installation
Before installation, one needs to check the configuration file 'make.def' in the config directory and modify the file if necessary. 
If it does not (yet) exist, copy 'make.def.template' to 'make.def' and edit the content for machine-specific data.  


Then
```bash
       make <benchmark-name> NPROCS=<number> CLASS=<class> [SUBTYPE=<type>] [VERSION=VEC]
```
where 

```bash
<benchmark-name>  is "bt", "cg", "dt", "ep", "ft", "is", 
                              "lu", "mg", or "sp"
<number>          is the number of processes
<class>           is "S", "W", "A", "B", "C", "D", or "E"
```

For example for compiling FT with 16 processe in class C, you can make it in this way:

```bash
cd /NPB3.3.1_DIR/NPB3.3-MPI/
make ft NPROCS=16 CLASS=C
```

An output titled ft.c.16 will be created at bin directory.
 
## Running

This is a templete for runing FT benchmark. You can change host file based on your need.

```bash
/MPI_INSTALL_DIR/bin/mpiexec -n 16 -f host ./bin/ft.c.16
```
