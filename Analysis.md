# CryptMPI: A Fast Encrypted MPI Library

##  Allreduce

- To enable NodeAware mode (only inter-node communications are encrypted):
```bash
export MV2_SECURITY_APPROACH=2005
```
- To enable Naive mode (both intra and inter-node communications are encrypted):
```bash
export MV2_SECURITY_APPROACH=2001 
```

##### 1-  Allreduce + Allgather:

In this mode which is suitable for small message sizes, only leader processes contribute in inter-node communication, and they perform Allgather to collect all nodes results

```bash
export MV2_Allgather_Reduce=1
export MV2_SECURITY_APPROACH=2005 
export MV2_OVERLAP_DECRYPTION=2  
```

##### 2-  Concurrent:

This mode involves three steps: Reduce-scatter (1st step) Allreduce (2nd step) and Allgather (3rd step)

<p align="center">
<img src="https://github.com/FSU-CS-EXPLORER-LAB/Encrypted_MPI/blob/main/figs/concurrent.JPG" width="50%" height="55%">
</p>
<p align="center">
<em>Fig 1: Illustration of the multi-leader algorithm on 3 nodes and 12 processes. LR and FR represent local reduction and final reduction, respectively.</em>
</p>

In the Concurrent mode there are two options for Reduce-scatter (1st step) and Allgather (3rd step):
- If MV2_CONCUR_RS_METHOD or MV2_CONCUR_AllGTHER_METHOD set to 0, they perform pt2pt process communication.
- If MV2_CONCUR_RS_METHOD or MV2_CONCUR_AllGTHER_METHOD set to 2, they perform Shared memory process communication.

Also, in Inter nodes communication (2nd step) there are 3 options:
- If MV2_CONCUR_INTER_METHOD set to 1, Recursive Doubling (RD) is applied (Default option).
- If MV2_CONCUR_INTER_METHOD set to 2, Reduce-scatter-Allgather (RS) is applied.
- If MV2_CONCUR_INTER_METHOD set to 3, Ring is applied.


```bash
export MV2_CONCUR_RS_METHOD=2
export MV2_CONCUR_INTER_METHOD=1 
export MV2_CONCUR_AllGTHER_METHOD=2
export MV2_SHMEM_BCAST=1
export MV2_SECURITY_APPROACH=2005
```

###### 3- Unencrypted Multi-leaders

In this mode, you can use concurrent methods in unencrypted approach:

```bash
export MV2_UNSEC_ALLREDUCE_MULTI_LEADER=1
unset MV2_SECURITY_APPROACH
```

- In current code these variable are set as following in 

/home/gavahi/explorer/cryptMPI-mvapich2-2.3.3/src/mpi/coll/ch3_shmem_coll.c

```bash
int mv2_allreduce_red_scat_allgather_algo_threshold=8*1024*1024; 
int mv2_allreduce_ring_algo_threshold=512;
int mv2_allreduce_ring_algo_ppn_threshold=64;
int mv2_red_scat_ring_algo_threshold=2048;
```
To change them export your values for these parameters:

```bash
MV2_ALLREDUCE_RED_SCAT_ALLGATHER_ALGO_THRESHOLD
MV2_ALLGATHER_RING_ALGO_THRESHOLD
MV2_ALLREDUCE_RING_ALGO_PPN_THRESHOLD
MV2_RED_SCAT_RING_ALGO_THRESHOLD
```

To disable any ring algorithm: 

```bash
export MV2_ALLRED_USE_RING=0
```

###### 4 -Supernode:

In this mode, first leader processes perform a reduce-scatter to perform a reduction on the local data. then leader processes send their data to super-node's leader in a hierarchy approach.

```bash
export SUPER_NODE=1 
export MV2_SECURITY_APPROACH=2005
```

<p align="center">
<img src="https://github.com/FSU-CS-EXPLORER-LAB/Encrypted_MPI/blob/main/figs/super_leader.JPG" width="50%" height="40%">
</p>
<p align="center">
<em>Fig 2: An example of the Super-leader Algorithm for 9 nodes and 2 process per node. The green arrows denote the communication pattern in the first round and the red arrows indicate the second round</em>
</p>

###### Debug Allreduce 

To check the correctness of final result, set following flag to print the calculated result for a specific message size. 
You can search in the code for `ALLREDUCE_PRINT_FUN` to change the message size.

```bash
export MV2_PRINT_FUN_NAME=1
export MV2_DEBUG_INIT_FILE=1
```

