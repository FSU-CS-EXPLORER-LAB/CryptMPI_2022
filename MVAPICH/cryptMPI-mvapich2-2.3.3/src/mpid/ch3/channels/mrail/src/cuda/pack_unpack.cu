/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 *
 */

#define MPI_ORDER_C              56
#define MPI_ORDER_FORTRAN        57
extern int rdma_cuda_vec_thread_blksz;
extern int rdma_cuda_vec_thread_ysz;
extern int rdma_cuda_subarr_thread_blksz;
extern int rdma_cuda_subarr_thread_xdim;
extern int rdma_cuda_subarr_thread_ydim;
extern int rdma_cuda_subarr_thread_zdim;

struct iovec
{
	void *iov_base; /* Pointer to data.  */
	size_t iov_len; /* Length of data.  */
};

__global__ void pack_subarray_c_double( double *dst, double *src, int nx, int ny, int nz, int sub_nx, int sub_ny, int sub_nz, int h_x, int h_y, int h_z)
{
//==============================================================================
// 2 Registers | 3 arguments
//==============================================================================
    int   i, j, k;
//==============================================================================

    // Identify current thread
    // Notice the +1 shift that is used in order to avoid ghost nodes
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
            dst[ i * sub_ny * sub_nz + j * sub_nz + k ] = src[ (i + h_x) * ny * nz + (j + h_y) * nz + (k + h_z) ];
    }
}

__global__ void unpack_subarray_c_double( double *dst, double *src, int nx, int ny, int nz, int sub_nx, int sub_ny, int sub_nz, int h_x, int h_y, int h_z)
{
//==============================================================================
// 2 Registers | 3 arguments
//==============================================================================
    int   i, j, k;
//==============================================================================

    // Identify current thread
    // Notice the +1 shift that is used in order to avoid ghost nodes
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
            dst[ (i + h_x) * ny * nz + (j + h_y) * nz + (k + h_z) ] = src[ i * sub_ny * sub_nz + j * sub_nz + k ];
    }
}

__global__ void pack_subarray_f_double( double *dst, double *src, int nx, int ny, int nz, int sub_nx, int sub_ny, int sub_nz, int h_x, int h_y, int h_z)
{
//==============================================================================
// 2 Registers | 3 arguments
//==============================================================================
    int   i, j, k;
//==============================================================================

    // Identify current thread
    // Notice the +1 shift that is used in order to avoid ghost nodes
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
      dst[ i + sub_nx * j + sub_nx * sub_ny * k ] = src[ (i + h_x) + nx * (j + h_y)  + nx * ny * (k + h_z) ];
    }
}

__global__ void unpack_subarray_f_double( double *dst, double *src, int nx, int ny, int nz, int sub_nx, int sub_ny, int sub_nz, int h_x, int h_y, int h_z)
{
//==============================================================================
// 2 Registers | 3 arguments
//==============================================================================
    int   i, j, k;
//==============================================================================

    // Identify current thread
    // Notice the +1 shift that is used in order to avoid ghost nodes
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
      dst[ (i + h_x) + nx * (j + h_y)  + nx * ny * (k + h_z) ] = src[ i + sub_nx * j + sub_nx * sub_ny * k ];
    }
}

__global__ void pack_subarray_c_float( float *dst, float *src, int nx, int ny, int nz, int sub_nx, int sub_ny, int sub_nz, int h_x, int h_y, int h_z)
{
//==============================================================================
// 2 Registers | 3 arguments
//==============================================================================
    int   i, j, k;
//==============================================================================

    // Identify current thread
    // Notice the +1 shift that is used in order to avoid ghost nodes
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
      dst[ i * sub_ny * sub_nz + j * sub_nz + k ] = src[ (i + h_x) * ny * nz + (j + h_y) * nz  + (k + h_z) ];
    }
}

__global__ void unpack_subarray_c_float( float *dst, float *src, int nx, int ny, int nz, int sub_nx, int sub_ny, int sub_nz, int h_x, int h_y, int h_z)
{
//==============================================================================
// 2 Registers | 3 arguments
//==============================================================================
    int   i, j, k;
//==============================================================================

    // Identify current thread
    // Notice the +1 shift that is used in order to avoid ghost nodes
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
      dst[ (i + h_x) * ny * nz + (j + h_y) * nz  + (k + h_z) ] = src[ i * sub_ny * sub_nz + j * sub_nz + k ];
    }
}

__global__ void pack_subarray_f_float( float *dst, float *src, int nx, int ny, int nz, int sub_nx, int sub_ny, int sub_nz, int h_x, int h_y, int h_z)
{
//==============================================================================
// 2 Registers | 3 arguments
//==============================================================================
    int   i, j, k;
//==============================================================================

    // Identify current thread
    // Notice the +1 shift that is used in order to avoid ghost nodes
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
      dst[ i + sub_nx * j + sub_nx * sub_ny * k ] = src[ (i + h_x) + nx * (j + h_y)  + nx * ny * (k + h_z) ];
    }
}

__global__ void unpack_subarray_f_float( float *dst, float *src, int nx, int ny, int nz, int sub_nx, int sub_ny, int sub_nz, int h_x, int h_y, int h_z)
{
//==============================================================================
// 2 Registers | 3 arguments
//==============================================================================
    int   i, j, k;
//==============================================================================

    // Identify current thread
    // Notice the +1 shift that is used in order to avoid ghost nodes
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
      dst[ (i + h_x) + nx * (j + h_y)  + nx * ny * (k + h_z) ] = src[ i + sub_nx * j + sub_nx * sub_ny * k ];
    }
}

__global__ void pack_subarray_c_char( char *dst, char *src, int nx, int ny, int nz, int sub_nx, int sub_ny, int sub_nz, int h_x, int h_y, int h_z)
{
//==============================================================================
// 2 Registers | 3 arguments
//==============================================================================
    int   i, j, k;
//==============================================================================

    // Identify current thread
    // Notice the +1 shift that is used in order to avoid ghost nodes
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
      dst[ i * sub_ny * sub_nz + j * sub_nz + k ] = src[ (i + h_x) * ny * nz + (j + h_y) * nz  + (k + h_z) ];
    }
}

__global__ void unpack_subarray_c_char( char *dst, char *src, int nx, int ny, int nz, int sub_nx, int sub_ny, int sub_nz, int h_x, int h_y, int h_z)
{
//==============================================================================
// 2 Registers | 3 arguments
//==============================================================================
    int   i, j, k;
//==============================================================================

    // Identify current thread
    // Notice the +1 shift that is used in order to avoid ghost nodes
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
        dst[ (i + h_x) * ny * nz + (j + h_y) * nz + (k + h_z) ] = src[ i * sub_ny * sub_nz + j * sub_nz + k ];
    }
}

__global__ void pack_subarray_f_char( char *dst, char *src, int nx, int ny, int nz, int sub_nx, int sub_ny, int sub_nz, int h_x, int h_y, int h_z)
{
//==============================================================================
// 2 Registers | 3 arguments
//==============================================================================
    int   i, j, k;
//==============================================================================

    // Identify current thread
    // Notice the +1 shift that is used in order to avoid ghost nodes
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
      dst[ i + sub_nx * j + sub_nx * sub_ny * k ] = src[ (i + h_x) + nx * (j + h_y)  + nx * ny * (k + h_z) ];
    }
}

__global__ void unpack_subarray_f_char( char *dst, char *src, int nx, int ny, int nz, int sub_nx, int sub_ny, int sub_nz, int h_x, int h_y, int h_z)
{
//==============================================================================
// 2 Registers | 3 arguments
//==============================================================================
    int   i, j, k;
//==============================================================================

    // Identify current thread
    // Notice the +1 shift that is used in order to avoid ghost nodes
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
      dst[ (i + h_x) + nx * (j + h_y)  + nx * ny * (k + h_z) ] = src[ i + sub_nx * j + sub_nx * sub_ny * k ];
    }
}

extern "C" void pack_subarray( void *dst, void *src, int dim, int nx, int ny, int nz, int sub_nx, int sub_ny, int sub_nz, int h_x, int h_y, int h_z, int sub_order, int el_size, cudaStream_t stream) 
{
    int BLOCK_SIZE_X = 8;
    int BLOCK_SIZE_Y = 8;
    int BLOCK_SIZE_Z = 16;
    int BLOCK_THREAD_SIZE = 1024;

    if ( rdma_cuda_subarr_thread_xdim != 0 && !(rdma_cuda_subarr_thread_xdim & (rdma_cuda_subarr_thread_xdim-1)) ){
        BLOCK_SIZE_X = rdma_cuda_subarr_thread_xdim;
    }
    if ( dim > 1 && rdma_cuda_subarr_thread_ydim != 0 && !(rdma_cuda_subarr_thread_ydim & (rdma_cuda_subarr_thread_ydim-1)) ){
        BLOCK_SIZE_Y = rdma_cuda_subarr_thread_ydim;
    }
    if ( dim > 2 && rdma_cuda_subarr_thread_zdim != 0 && !(rdma_cuda_subarr_thread_zdim & (rdma_cuda_subarr_thread_zdim-1)) ){
        BLOCK_SIZE_Z = rdma_cuda_subarr_thread_zdim;
    }
    if ( rdma_cuda_subarr_thread_blksz != 0 && !(rdma_cuda_subarr_thread_blksz & (rdma_cuda_subarr_thread_blksz-1)) ){
		BLOCK_THREAD_SIZE = ( (rdma_cuda_subarr_thread_blksz < 1024) ? rdma_cuda_subarr_thread_blksz : 1024);
	}

    if ( 3 == dim && BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z > BLOCK_THREAD_SIZE ){
        BLOCK_SIZE_X = 8;
        BLOCK_SIZE_Y = 8;
        BLOCK_SIZE_Z = 16;
    } else if ( 2 == dim && BLOCK_SIZE_X * BLOCK_SIZE_Y > BLOCK_THREAD_SIZE ){
        BLOCK_SIZE_X = 16;
        BLOCK_SIZE_Y = 32;
        BLOCK_SIZE_Z = 1;
    } else if ( 1 == dim ) {
        BLOCK_SIZE_X = 256; 
        BLOCK_SIZE_Y = 4;
        BLOCK_SIZE_Z = 1;
    }

    int GRID_SIZE_X = (sub_nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    int GRID_SIZE_Y = (sub_ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    int GRID_SIZE_Z = (sub_nz + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
    dim3 dimblock( BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z );
    dim3 dimgrid( GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z );

    if ( MPI_ORDER_C == sub_order ){
        if (el_size == 4) {
            pack_subarray_c_float<<<dimgrid, dimblock, 0, stream>>>((float *) dst, (float *) src, nx, ny, nz, sub_nx, sub_ny, sub_nz, h_x, h_y, h_z );
        } else if (el_size == 1) {
            pack_subarray_c_char<<<dimgrid, dimblock, 0, stream>>>((char *) dst, (char *) src, nx, ny, nz, sub_nx, sub_ny, sub_nz, h_x, h_y, h_z );
        } else if (el_size == 8) {
            pack_subarray_c_double<<<dimgrid, dimblock, 0, stream>>>((double *) dst, (double *) src, nx, ny, nz, sub_nx, sub_ny, sub_nz, h_x, h_y, h_z );
        }
    } else if ( MPI_ORDER_FORTRAN == sub_order ){
        if (el_size == 4) {
            pack_subarray_f_float<<<dimgrid, dimblock, 0, stream>>>((float *) dst, (float *) src, nx, ny, nz, sub_nx, sub_ny, sub_nz, h_x, h_y, h_z );
        } else if (el_size == 1) {
            pack_subarray_f_char<<<dimgrid, dimblock, 0, stream>>>((char *) dst, (char *) src, nx, ny, nz, sub_nx, sub_ny, sub_nz, h_x, h_y, h_z );
        } else if (el_size == 8) {
            pack_subarray_f_double<<<dimgrid, dimblock, 0, stream>>>((double *) dst, (double *) src, nx, ny, nz, sub_nx, sub_ny, sub_nz, h_x, h_y, h_z );
        }
    }
}

extern "C" void unpack_subarray( void *dst, void *src, int dim, int nx, int ny, int nz, int sub_nx, int sub_ny, int sub_nz, int h_x, int h_y, int h_z, int sub_order, int el_size, cudaStream_t stream)
{
    int BLOCK_SIZE_X = 8;
    int BLOCK_SIZE_Y = 8;
    int BLOCK_SIZE_Z = 16;
    int BLOCK_THREAD_SIZE = 1024;

    if ( rdma_cuda_subarr_thread_xdim != 0 && !(rdma_cuda_subarr_thread_xdim & (rdma_cuda_subarr_thread_xdim-1)) ){
        BLOCK_SIZE_X = rdma_cuda_subarr_thread_xdim;
    }
    if ( dim > 1 && rdma_cuda_subarr_thread_ydim != 0 && !(rdma_cuda_subarr_thread_ydim & (rdma_cuda_subarr_thread_ydim-1)) ){
        BLOCK_SIZE_Y = rdma_cuda_subarr_thread_ydim;
    }
    if ( dim > 2 && rdma_cuda_subarr_thread_zdim != 0 && !(rdma_cuda_subarr_thread_zdim & (rdma_cuda_subarr_thread_zdim-1)) ){
        BLOCK_SIZE_Z = rdma_cuda_subarr_thread_zdim;
    }
    if ( rdma_cuda_subarr_thread_blksz != 0 && !(rdma_cuda_subarr_thread_blksz & (rdma_cuda_subarr_thread_blksz-1)) ){
		BLOCK_THREAD_SIZE = ( (rdma_cuda_subarr_thread_blksz < 1024) ? rdma_cuda_subarr_thread_blksz : 1024);
	}

    if ( 3 == dim && BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z > BLOCK_THREAD_SIZE ){
        BLOCK_SIZE_X = 8;
        BLOCK_SIZE_Y = 8;
        BLOCK_SIZE_Z = 16;
    } else if ( 2 == dim && BLOCK_SIZE_X * BLOCK_SIZE_Y > BLOCK_THREAD_SIZE ){
        BLOCK_SIZE_X = 16;
        BLOCK_SIZE_Y = 32;
        BLOCK_SIZE_Z = 1;
    } else if ( 1 == dim ) {
        BLOCK_SIZE_X = 256;
        BLOCK_SIZE_Y = 4;
        BLOCK_SIZE_Z = 1;
    }

    int GRID_SIZE_X = (sub_nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    int GRID_SIZE_Y = (sub_ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    int GRID_SIZE_Z = (sub_nz + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
    dim3 dimblock( BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z );
    dim3 dimgrid( GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z );

    if ( MPI_ORDER_C == sub_order ){
        if (el_size == 4) {
            unpack_subarray_c_float<<<dimgrid, dimblock, 0, stream>>>((float *) dst, (float *) src, nx, ny, nz, sub_nx, sub_ny, sub_nz, h_x, h_y, h_z );
        } else if (el_size == 1) {
            unpack_subarray_c_char<<<dimgrid, dimblock, 0, stream>>>((char *) dst, (char *) src, nx, ny, nz, sub_nx, sub_ny, sub_nz, h_x, h_y, h_z );
        } else if (el_size == 8) {
            unpack_subarray_c_double<<<dimgrid, dimblock, 0, stream>>>((double *) dst, (double *) src, nx, ny, nz, sub_nx, sub_ny, sub_nz, h_x, h_y, h_z );
        }
    } else if ( MPI_ORDER_FORTRAN == sub_order ){
        if (el_size == 4) {
            unpack_subarray_f_float<<<dimgrid, dimblock, 0, stream>>>((float *) dst, (float *) src, nx, ny, nz, sub_nx, sub_ny, sub_nz, h_x, h_y, h_z );
        } else if (el_size == 1) {
            unpack_subarray_f_char<<<dimgrid, dimblock, 0, stream>>>((char *) dst, (char *) src, nx, ny, nz, sub_nx, sub_ny, sub_nz, h_x, h_y, h_z );
        } else if (el_size == 8) {
            unpack_subarray_f_double<<<dimgrid, dimblock, 0, stream>>>((double *) dst, (double *) src, nx, ny, nz, sub_nx, sub_ny, sub_nz, h_x, h_y, h_z );
        }
    }
}

__global__ void pack_unpack_vector_double( double *dst, int dpitch, double *src, int spitch, int width, int height)
{
//==============================================================================
// 2 Registers | 2 arguments
//==============================================================================
    int   i, j;
//==============================================================================
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    if ( i < height && j < width ) {
            dst[i * dpitch + j] = src[i * spitch + j];
    }
}

__global__ void pack_unpack_vector_float( float *dst, int dpitch, float *src, int spitch, int width, int height)
{
//==============================================================================
// 2 Registers | 2 arguments
//==============================================================================
    int   i, j;
//==============================================================================
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    if ( i < height && j < width ) {
            dst[i * dpitch + j] = src[i * spitch + j];
    }
}

__global__ void pack_unpack_vector_char(  char *dst, int dpitch, char *src, int spitch, int width, int height)
{
//==============================================================================
// 2 Registers | 2 arguments
//==============================================================================
    int   i, j;
//==============================================================================
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < height && j < width ) {
            dst[i * dpitch + j] = src[i * spitch + j];
    }
}

extern "C" void pack_unpack_vector_kernel( void *dst, int dpitch, void *src, int spitch, int width, int height, cudaStream_t stream)
{
	int BLOCK_SIZE_Y, BLOCK_THREAD_SIZE;
    int dtsize;
	int elems;
    if ((0 == (width % sizeof(double))) && (0 == (dpitch % sizeof(double))) && (0 == (spitch % sizeof(double)))) {
		dtsize = sizeof(double);
    } else if ((0 == (width % sizeof(float))) && (0 == (dpitch % sizeof(float))) && (0 == (spitch % sizeof(float)))) {
		dtsize = sizeof(float);
    } else {
		dtsize = sizeof(char);
		}
    elems = width / dtsize;

    if ( rdma_cuda_vec_thread_ysz != 0 && !(rdma_cuda_vec_thread_ysz & (rdma_cuda_vec_thread_ysz-1)) ){
	    BLOCK_SIZE_Y = rdma_cuda_vec_thread_ysz;
	}else{
	    switch ( elems ){
        case 1:
	        BLOCK_SIZE_Y = 1;
		    break;
		case 2:
			BLOCK_SIZE_Y = 2;
			break;
		case 3: case 4:
			BLOCK_SIZE_Y = 4;
			break;
		case 5: case 6: case 7: case 8:
			BLOCK_SIZE_Y = 8;
			break;
		case 9: case 10: case 11: case 12: case 13: case 14: case 15: case 16:
			BLOCK_SIZE_Y = 16;
			break;
		default:
			BLOCK_SIZE_Y = 32;
			break;
		}
    }

    if ( rdma_cuda_vec_thread_blksz != 0 && !(rdma_cuda_vec_thread_blksz & (rdma_cuda_vec_thread_blksz-1)) ){
		BLOCK_THREAD_SIZE = ( (rdma_cuda_vec_thread_blksz < 1024) ? rdma_cuda_vec_thread_blksz : 1024);
	}
	else{
		BLOCK_THREAD_SIZE = 1024;
	}
    int BLOCK_SIZE_X = BLOCK_THREAD_SIZE / BLOCK_SIZE_Y;
    int GRID_SIZE_X = (height + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    int GRID_SIZE_Y = (elems + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    dim3 dimblock( BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 dimgrid( GRID_SIZE_X, GRID_SIZE_Y, 1);
		
    if ((0 == (width % sizeof(double))) && (0 == (dpitch % sizeof(double))) && (0 == (spitch % sizeof(double)))) {
        pack_unpack_vector_double<<<dimgrid, dimblock, 0, stream>>>((double *) dst, dpitch / sizeof(double), 
					(double *) src, spitch / sizeof(double), width / sizeof(double), height);
    } else if ((0 == (width % sizeof(float))) && (0 == (dpitch % sizeof(float))) && (0 == (spitch % sizeof(float)))) {
        pack_unpack_vector_float<<<dimgrid, dimblock, 0, stream>>>((float *) dst, dpitch / sizeof(float), 
					(float *) src, spitch / sizeof(float), width / sizeof(float), height);
    } else if ((0 == (width % sizeof(char))) && (0 == (dpitch % sizeof(char))) && (0 == (spitch % sizeof(char)))) {
        pack_unpack_vector_char<<<dimgrid, dimblock, 0, stream>>>((char *) dst, dpitch / sizeof(char), 
					(char *) src, spitch / sizeof(char), width / sizeof(char), height);
    }
}

