[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3758775.svg)](https://doi.org/10.5281/zenodo.3758775)
# CryptMPI: A Fast Encrypted MPI Library
CryptMPI provides secure inter-node communication in the HPC cluster and cloud environment.
We implemented two prototypes in MPICH-3.3 (for Ethernet) and MVAPICH2-2.3.3(for Infiniband), both using AES-
GCM from the [BoringSSL library](https://boringssl.googlesource.com/boringssl/).

Up to now, we implemented secure approach for following routines: 

| Pont-to-point routines|  Collective routines   |
|:---------------------:|:----------------------:|
| MPI_Send          	| MPI_Allgather		     |
| MPI_Recv    	 		| MPI_Allreduce     	 |
| MPI_Isend     	    | MPI_Bcast		      	 |
| MPI_Irecv         	| MPI_Gather          	 |
| MPI_Wait          	| MPI_Scatter            |
| MPI_Waitall       	| MPI_Alltoall            |

## Installation
To install cryptMPI for the Infiniband and Ethernet network please follow following steps:
#### Package requirement
 autoconf version... >= 2.67
 automake version... >= 1.15
 libtool version... >= 2.4.4

To install the above package you could use get-lib.sh

After installing, set the path for the above packages.

```bash
export PATH=/HOME_DIR/automake/bin:$PATH
export LD_LIBRARY_PATH=/HOME_DIR/automake/lib:$LD_LIBRARY_PATH
```


#### Installation for BoringSSL

Download and unzip it:
```bash
wget https://github.com/google/boringssl/archive/master.zip
unzip
```

BoringSSL needs GO package. So, install GO in this way:

```bash
cd BORINGSSL-DIR/
wget https://golang.org/dl/go1.16.2.linux-amd64.tar.gz 
tar xzf go1.16.2.linux-amd64.tar.gz
export GOROOT=/BORINGSSL-DIR/go
export GOPATH=/BORINGSSL-DIR/go
export PATH=$GOPATH/bin:$GOROOT/bin:$PATH
```

After installing GO, countine BoringSSL installagtion:

```bash
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=1 ..
make
```

#### Installation for CryptMPI-MPICH (Ethernet) and  CryptMPI-MVAPICH (Infiniband)
Steps:
```bash
./autogen.sh
./configure --prefix=/INSTALLITION_DIR/install
```
*Note: In MVAPICH installation (Infiniband), for Intel Omni Path interconnect configure with --with-device=ch3:psm*

In the *Makefile*: 

1- add -L/YOUR_PATH_TO_BORINGSSL/build/crypto -lcrypto in *LIBS*

(e.g. LIBS =-L/YOUR_PATH_TO_BORINGSSL/build/crypto -lcrypto -libmad -lrdmacm -libumad -libverbs -ldl -lrt -lm -lpthread)

2- add -I/YOUR_PATH_TO_BORINGSSL/include -fopenmp in *CFLAGS*

3- add -fopenmp in *LDFLAGS*


```bash
export LD_LIBRARY_PATH=/YOUR_PATH_TO_BORINGSSL/build/crypto
make clean
make -j
make install
```

## Usage
To run MPI applications using CryptMPI please follow following steps:
#### CryptMPI-MVAPICH (Infiniband)
```bash
export LD_LIBRARY_PATH=/MVAPICH_INSTALL_DIR/install/lib:/YOUR_PATH_TO_MVAPICH/mvapich2-2.3.2/boringssl-master/build/crypto
export MV2_ENABLE_AFFINITY=1
export MV2_CPU_BINDING_POLICY=hybrid
export MV2_HYBRID_BINDING_POLICY=spread 
```

#### CryptMPI-MPICH (Ethernet)
```bash
export LD_LIBRARY_PATH=/MPICH_INSTALL_DIR/install/lib:/YOUR_PATH_TO_MPICH/mpich-3.4.2/boringssl-master/build/crypto
```


## Performance measurement
The performance was measured on 100Gb/s Infiniband and 10Gb/s Ethernet network. Benchmark program used:
- OSU micro-benchmark 5.8
- NAS parallel benchmarks 3.3.1 
- N-Body

## Flags List

The flags are discussed in this section, work in both MIPICH and MVAPICH.


#### Gather


```bash
export MV2_SECURITY_APPROACH=301
unset MV2_INTER_GATHER_TUNING
unset MV2_CONCURRENT_COMM
echo "Naive  [MPIR_Naive_Sec_Gather]" 


export MV2_SECURITY_APPROACH=302
export MV2_INTER_GATHER_TUNING=3 
echo "Opportunistic Binomial Gather (Direct - No Shared-Mem) [Gather_intra]" 


export MV2_SECURITY_APPROACH=302
export MV2_INTER_GATHER_TUNING=4
export MV2_CONCURRENT_COMM=1
echo "CHS [Gather_MV2_Direct_CHS]" 
```


#### Scatter


```bash
export MV2_SECURITY_APPROACH=200
export MV2_INTER_SCATTER_TUNING=6
export MV2_CONCURRENT_COMM=1
echo "b-s-c: concurrent with shared memory [Scatter_MV2_Direct_CHS]"


export MV2_SECURITY_APPROACH=200
export MV2_INTER_SCATTER_TUNING=7
echo "rr: round robin [Scatter_MV2_Direct_no_shmem_intra_RR]"


export MV2_SECURITY_APPROACH=200
export MV2_INTER_SCATTER_TUNING=9
echo "n-bcast: hierarchical broadcast [Scatter_MV2_Direct_HBcast]"


export MV2_SECURITY_APPROACH=200
export MV2_INTER_SCATTER_TUNING=10
echo "MPIR_Scatter_MV2_Direct_no_shmem [Scatter_MV2_Direct_no_shmem]"


export MV2_SECURITY_APPROACH=200
export MV2_INTER_SCATTER_TUNING=11
echo "MPIR_Scatter_MV2_two_level_Direct [Scatter_MV2_two_level_Direct]"


export MV2_SECURITY_APPROACH=200
export MV2_INTER_SCATTER_TUNING=12
echo "MPIR_Scatter_MV2_Direct [Scatter_MV2_Direct]"
```

#### AlltoAll

- MVAPICH

```bash

unset MV2_ALLTOALL_TUNING
unset MV2_SECURITY_APPROACH
unset MV2_INTER_ALLGATHER_TUNING
unset MV2_CONCURRENT_COMM
echo "Default"


export MV2_SECURITY_APPROACH=1002
echo "Naive"


export MV2_ALLTOALL_TUNING=0
export MV2_SECURITY_APPROACH=2001
echo "OBruck"


export MV2_ALLTOALL_TUNING=2
export MV2_SECURITY_APPROACH=2001
echo "OSD"


export MV2_ALLTOALL_TUNING=5
export CONCURRENT_COMM=1
unset MV2_SECURITY_APPROACH
echo "CHS"


export MV2_ALLTOALL_TUNING=5
export MV2_SECURITY_APPROACH=2001
echo "O-CHS"

export MV2_ALLTOALL_TUNING=5
export MV2_SECURITY_APPROACH=2002
echo "Naive-CHS"

```



- MPICH

```bash

unset MV2_ALLTOALL_TUNING
unset MV2_SECURITY_APPROACH
unset MV2_INTER_ALLGATHER_TUNING
unset MV2_CONCURRENT_COMM
echo "Default"


export MV2_SECURITY_APPROACH=1002
echo "Naive"


export MV2_ALLTOALL_TUNING=0
export MV2_SECURITY_APPROACH=2001
echo "OBruck"


export MV2_ALLTOALL_TUNING=2
export MV2_SECURITY_APPROACH=2001
echo "OSD"


export MV2_ALLTOALL_TUNING=5
unset MV2_SECURITY_APPROACH
echo "CHS"


export MV2_ALLTOALL_TUNING=5
export MV2_SECURITY_APPROACH=2002
echo "Naive-CHS"


export MV2_ALLTOALL_TUNING=5
export MV2_SECURITY_APPROACH=2001
echo "O-CHS"
```


#### Allgather

- MVAPICH

```bash
export MV2_SECURITY_APPROACH=1001
echo "Testing Naive Default"  


export MV2_SECURITY_APPROACH=2005
echo "Testing Opportunistic Default"  


export MV2_INTER_ALLGATHER_TUNING=12
unset MV2_SECURITY_APPROACH
echo "Testing C-Ring"  


export MV2_INTER_ALLGATHER_TUNING=12
export MV2_SECURITY_APPROACH=2005
echo "Testing Encrypted C-Ring"  


export MV2_INTER_ALLGATHER_TUNING=13
unset MV2_SECURITY_APPROACH
echo "Testing C-RD"  


export MV2_INTER_ALLGATHER_TUNING=13
export MV2_SECURITY_APPROACH=2005
echo "Testing Encrypted C-RD"  


export MV2_INTER_ALLGATHER_TUNING=18
export MV2_SECURITY_APPROACH=2006
echo "Testing HS2"  


export MV2_INTER_ALLGATHER_TUNING=17
export MV2_SECURITY_APPROACH=2005
echo "Testing O-RD2"  


export MV2_INTER_ALLGATHER_TUNING=14
unset MV2_SECURITY_APPROACH
echo "Testing Shared-Mem"  


export MV2_INTER_ALLGATHER_TUNING=14
export MV2_SECURITY_APPROACH=2006
export MV2_SHMEM_LEADERS=1
echo "Testing HS1"  


export MV2_INTER_ALLGATHER_TUNING=20
export MV2_SHMEM_LEADERS=1
export MV2_CONCURRENT_COMM=1
unset MV2_SECURITY_APPROACH
echo "Testing CHS"  


export MV2_INTER_ALLGATHER_TUNING=20
export MV2_SHMEM_LEADERS=1
export MV2_CONCURRENT_COMM=1
export MV2_SECURITY_APPROACH=2006
echo "Testing Encrypted CHS"  
```



- MPICH

```bash
export MV2_INTER_ALLGATHER_TUNING=1
export MV2_SECURITY_APPROACH=1001
echo "MPIR_Naive_Sec_Allgather"


export MV2_INTER_ALLGATHER_TUNING=8 
export MV2_SECURITY_APPROACH=2005
echo "MPIR_Allgather_Bruck_SEC"


export MV2_INTER_ALLGATHER_TUNING=9 
export MV2_SECURITY_APPROACH=2005
echo "MPIR_Allgather_Ring_SEC"


export MV2_INTER_ALLGATHER_TUNING=10 
export MV2_SECURITY_APPROACH=2005
echo "MPIR_Allgather_RD_MV2"


export MV2_INTER_ALLGATHER_TUNING=14
export MV2_SECURITY_APPROACH=2006
export MV2_SHMEM_LEADERS=1
echo "ALLGATHER_2LVL_SHMEM"


export MV2_INTER_ALLGATHER_TUNING=16 
export MV2_SECURITY_APPROACH=2005
echo "MPIR_Allgather_NaivePlus_RDB_MV2"


export MV2_INTER_ALLGATHER_TUNING=18
export MV2_SECURITY_APPROACH=2006
echo "MPIR_2lvl_SharedMem_Concurrent_Encryption_Allgather(Single-leader)"


export MV2_INTER_ALLGATHER_TUNING=20
export MV2_SHMEM_LEADERS=1
export MV2_CONCURRENT_COMM=1
export MV2_SECURITY_APPROACH=2006
echo "MPIR_Allgather_2lvl_Concurrent_Multileader_SharedMem(Multi-leaders)"


export MV2_INTER_ALLGATHER_TUNING=21
export MV2_SECURITY_APPROACH=2007
echo "MPIR_2lvl_Allgather_MV2(SH1 Not-uniform)"


export MV2_INTER_ALLGATHER_TUNING=22
export MV2_SECURITY_APPROACH=2007
echo "MPIR_2lvl_Allgather_nonblocked_MV2"
```

 
#### Allreduce

- MVAPICH

```bash
export MV2_Allgather_Reduce=1
export MV2_SECURITY_APPROACH=2005 
export MV2_OVERLAP_DECRYPTION=2
echo "Allreduce + Allgather"


export SUPER_NODE=1 
export MV2_SECURITY_APPROACH=2005
echo "Supernode"


export MV2_CONCUR_RS_METHOD=2
export MV2_CONCUR_INTER_METHOD=1 
export MV2_CONCUR_AllGTHER_METHOD=2
export MV2_SHMEM_BCAST=1
export MV2_SECURITY_APPROACH=2005
echo "Concurrent via Recursive Doubling (RD)"


export MV2_CONCUR_RS_METHOD=2
export MV2_CONCUR_INTER_METHOD=2
export MV2_CONCUR_AllGTHER_METHOD=2
export MV2_SHMEM_BCAST=1
export MV2_SECURITY_APPROACH=2005
echo "Concurrent via Reduce-scatter-Allgather (RS)"


export MV2_CONCUR_RS_METHOD=2
export MV2_CONCUR_INTER_METHOD=3 
export MV2_CONCUR_AllGTHER_METHOD=2
export MV2_SHMEM_BCAST=1
export MV2_SECURITY_APPROACH=2005
echo "Concurrent via Ring"
```

- MPICH

```bash
export MV2_SECURITY_APPROACH=2005 
export MV2_INTER_ALLREDUCE_TUNING=1
echo "Recursive Doubling (RD) Secure"


export MV2_SECURITY_APPROACH=2005 
export MV2_INTER_ALLREDUCE_TUNING=2
echo "Reduce-scatter-Allgather (RS) Secure"


export MV2_SECURITY_APPROACH=2005 
export MV2_INTER_ALLREDUCE_TUNING=3
echo "SMP (Single-leader + Shared Memory) Secure"


export MV2_SECURITY_APPROACH=2005 
export MV2_INTER_ALLREDUCE_TUNING=6
echo "Concurrent (Multileader + Shared Memory) via Recursive Doubling (RD)"


export MV2_SECURITY_APPROACH=2005 
export MV2_INTER_ALLREDUCE_TUNING=7
echo "Concurrent (Multileader + Shared Memory) via Reduce-scatter-Allgather (RS)"


export MV2_SECURITY_APPROACH=2005 
export MV2_INTER_ALLREDUCE_TUNING=8
echo "Concurrent (Multileader + Shared Memory) via Ring"
```

#### Bcast

Set these parameters to enable MPIR_Bcast_ML_Shmem_MV2() which is responsible for encrypted multi-leader Bcast:

- MVAPICH

```bash
unset MV2_CONCURRENT_COMM
unset MV2_CONCURRENT_BCAST
unset MV2_INTER_BCAST_TUNING
export MV2_SECURITY_APPROACH=1
echo "Naive"


export MV2_CONCURRENT_COMM=1
export MV2_CONCURRENT_BCAST=2
export MV2_INTER_BCAST_TUNING=13
export MV2_SECURITY_APPROACH=0
echo "Unencrypted CHS (Multileader + Shared Memory)"


export MV2_CONCURRENT_COMM=1
export MV2_CONCURRENT_BCAST=2
export MV2_INTER_BCAST_TUNING=13
export MV2_SECURITY_APPROACH=3
echo "Encrypted CHS (Multileader + Shared Memory)"
```

- MPICH

```bash
export MV2_SECURITY_APPROACH=1
echo "Naive"


export MV2_CONCURRENT_BCAST=1
export MV2_SECURITY_APPROACH=0
echo "Unencrypted CHS (Multileader + Shared Memory)"


export MV2_CONCURRENT_BCAST=1
export MV2_SECURITY_APPROACH=333
echo "Encrypted CHS (Multileader + Shared Memory)"
```


#### Hints

List all exported environment variables command:

```bash
printenv | grep MV2 | perl -ne 'print "export $_"'
```


Display Functions name and debuging points:

```bash
export MV2_PRINT_FUN_NAME=1
export MV2_DEBUG_INIT_FILE=1
```
