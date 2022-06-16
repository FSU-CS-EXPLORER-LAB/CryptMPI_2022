/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/* Algorithm: Binomial bcast
 *
 * For short messages, we use a binomial tree algorithm.
 * Cost = lgp.alpha + n.lgp.beta
 */
#undef FUNCNAME
#define FUNCNAME MPIR_Bcast_intra_binomial
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Bcast_intra_binomial(void *buffer,
                              int count,
                              MPI_Datatype datatype,
                              int root, MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if BCAST_PRINT_FUN
   if (PRINT_FUN_NAME ){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		// printf("[Bcast [%d] [%d]  host = %s Sec=%d count = %d] Func: MPIR_Bcast_intra_binomial\n", comm_ptr->rank,init_rank,hostname,security_approach,count);fflush(stdout);
        printf("[Bcast rank = %d host = %s count = %d] Func: MPIR_Bcast_intra_binomial\n", comm_ptr->rank,hostname,count);fflush(stdout);
	}
#endif 	
    int rank, comm_size, src, dst;
    int relative_rank, mask;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint nbytes = 0;
    MPI_Aint recvd_size;
    MPI_Status status;
    int is_contig;
    MPI_Aint type_size;
    MPI_Aint position;
    void *tmp_buf = NULL;
    MPIR_CHKLMEM_DECL(1);

    // printf("*** Bcast_binomial 01 c = %d r = %d \n",count,init_rank);

    comm_size = comm_ptr->local_size;
    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 00 c=%d\n",init_rank,rank,count);
    rank = comm_ptr->rank;

    // if (1) fprintf(stderr,"[%d] [%d] BCast-Bio 000 c=%d\n",init_rank,rank,count);

    /* If there is only one process, return */
    if (comm_size == 1)
        goto fn_exit;

    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 01 c=%d\n",init_rank,rank,count);
	if (HANDLE_GET_KIND(datatype) == HANDLE_KIND_BUILTIN)
        is_contig = 1;
    else {
        MPIR_Datatype_is_contig(datatype, &is_contig);
    }

    MPIR_Datatype_get_size_macro(datatype, type_size);

    nbytes = type_size * count;
    if (nbytes == 0)
        goto fn_exit;   /* nothing to do */
// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 02 c=%d\n",init_rank,rank,count);
    if (!is_contig) {
		// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 03 c=%d\n",init_rank,rank,count);
        MPIR_CHKLMEM_MALLOC(tmp_buf, void *, nbytes, mpi_errno, "tmp_buf", MPL_MEM_BUFFER);

        /* TODO: Pipeline the packing and communication */
        position = 0;
        if (rank == root) {
			// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 03-1 c=%d\n",init_rank,rank,count);
            mpi_errno = MPIR_Pack_impl(buffer, count, datatype, tmp_buf, nbytes, &position);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
    }

    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;
// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 04  c=%d  root=%d  relative_rank=%d\n",init_rank,rank,count,root,relative_rank);
    /* Use short message algorithm, namely, binomial tree */

    /* Algorithm:
     * This uses a fairly basic recursive subdivision algorithm.
     * The root sends to the process comm_size/2 away; the receiver becomes
     * a root for a subtree and applies the same process.
     *
     * So that the new root can easily identify the size of its
     * subtree, the (subtree) roots are all powers of two (relative
     * to the root) If m = the first power of 2 such that 2^m >= the
     * size of the communicator, then the subtree at root at 2^(m-k)
     * has size 2^k (with special handling for subtrees that aren't
     * a power of two in size).
     *
     * Do subdivision.  There are two phases:
     * 1. Wait for arrival of data.  Because of the power of two nature
     * of the subtree roots, the source of this message is alwyas the
     * process whose relative rank has the least significant 1 bit CLEARED.
     * That is, process 4 (100) receives from process 0, process 7 (111)
     * from process 6 (110), etc.
     * 2. Forward to my subtree
     *
     * Note that the process that is the tree root is handled automatically
     * by this code, since it has no bits set.  */

    mask = 0x1;
    while (mask < comm_size) {
		// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 05 root=%d  relative_rank=%d  c=%d\n",init_rank,rank,root,relative_rank,count);
        if (relative_rank & mask) {
			//if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 06 c=%d\n",init_rank,rank,count);
            src = rank - mask;
            if (src < 0)
                src += comm_size;
            if (!is_contig){
				// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 07 c=%d src=%d [%d]\n",init_rank,rank,count,src,mask);
                mpi_errno = MPIC_Recv(tmp_buf, nbytes, MPI_BYTE, src,
                                      MPIR_BCAST_TAG, comm_ptr, &status, errflag);
									  
			}
            else{
				// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 08 c=%d src=%d [%d]\n",init_rank,rank,count,src,mask);
                mpi_errno = MPIC_Recv(buffer, count, datatype, src,
                                      MPIR_BCAST_TAG, comm_ptr, &status, errflag);
            
				
			}
			if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
			// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 09 c=%d\n",init_rank,rank,count);

            /* check that we received as much as we expected */
            MPIR_Get_count_impl(&status, MPI_BYTE, &recvd_size);
            if (recvd_size != nbytes) {
				// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 10 c=%d\n",init_rank,rank,count);
                if (*errflag == MPIR_ERR_NONE)
                    *errflag = MPIR_ERR_OTHER;
                MPIR_ERR_SET2(mpi_errno, MPI_ERR_OTHER,
                              "**collective_size_mismatch",
                              "**collective_size_mismatch %d %d", recvd_size, nbytes);
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
			// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 11 c=%d\n",init_rank,rank,count);
            break;
        }
        mask <<= 1;
    }

    /* This process is responsible for all processes that have bits
     * set from the LSB upto (but not including) mask.  Because of
     * the "not including", we start by shifting mask back down one.
     *
     * We can easily change to a different algorithm at any power of two
     * by changing the test (mask > 1) to (mask > block_size)
     *
     * One such version would use non-blocking operations for the last 2-4
     * steps (this also bounds the number of MPI_Requests that would
     * be needed).  */

    mask >>= 1;
	// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 12 c=%d\n",init_rank,rank,count);
    while (mask > 0) {
		// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 13 mask=%d  relative_rank=%d  c=%d\n",init_rank,rank,mask,relative_rank,count);
        if (relative_rank + mask < comm_size) {
			// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 14 c=%d\n",init_rank,rank,count);
            dst = rank + mask;
            if (dst >= comm_size)
                dst -= comm_size;
            if (!is_contig){
				// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 15 dst=%d c=%d [%d]\n",init_rank,rank,dst,count,mask);
                mpi_errno = MPIC_Send(tmp_buf, nbytes, MPI_BYTE, dst,
                                      MPIR_BCAST_TAG, comm_ptr, errflag);
			
			}
            else{
					// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 16 dst=%d [%d]\n",init_rank,rank,dst,count,mask);
                mpi_errno = MPIC_Send(buffer, count, datatype, dst,
                                      MPIR_BCAST_TAG, comm_ptr, errflag);
		
			}
			// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 17 c=%d\n",init_rank,rank,count);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
		// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 18 c=%d\n",init_rank,rank,count);
        mask >>= 1;
    }
// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 19 c=%d\n",init_rank,rank,count);
    if (!is_contig) {
		// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 20 c=%d\n",init_rank,rank,count);
        if (rank != root) {
            position = 0;
            mpi_errno = MPIR_Unpack_impl(tmp_buf, nbytes, &position, buffer, count, datatype);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

        }
    }
	// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] BCast-Bio 21 c=%d\n",init_rank,rank,count);

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    /* --END ERROR HANDLING-- */
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
