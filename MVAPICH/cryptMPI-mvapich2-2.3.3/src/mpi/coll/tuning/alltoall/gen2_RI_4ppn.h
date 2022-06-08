#define GEN2__RI__4PPN {		\
	{		\
	4,		\
	{0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},		\
	16,		\
	{		\
	{1, &MPIR_Alltoall_RD_MV2},		\
	{2, &MPIR_Alltoall_RD_MV2},		\
	{4, &MPIR_Alltoall_RD_MV2},		\
	{8, &MPIR_Alltoall_Scatter_dest_MV2},		\
	{16, &MPIR_Alltoall_Scatter_dest_MV2},		\
	{32, &MPIR_Alltoall_Scatter_dest_MV2},		\
	{64, &MPIR_Alltoall_Scatter_dest_MV2},		\
	{128, &MPIR_Alltoall_Scatter_dest_MV2},		\
	{256, &MPIR_Alltoall_Scatter_dest_MV2},		\
	{512, &MPIR_Alltoall_Scatter_dest_MV2},		\
	{1024, &MPIR_Alltoall_Scatter_dest_MV2},		\
	{2048, &MPIR_Alltoall_Scatter_dest_MV2},		\
	{4096, &MPIR_Alltoall_Scatter_dest_MV2},		\
	{8192, &MPIR_Alltoall_Scatter_dest_MV2},		\
	{16384, &MPIR_Alltoall_Scatter_dest_MV2},		\
	{32768, &MPIR_Alltoall_Scatter_dest_MV2}		\
	}		\
	}		 \
}
