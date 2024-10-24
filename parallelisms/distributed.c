#include <mpi.h>


#define rank0_printf(rank, ...) if (rank == 0) { printf(__VA_ARGS__); }


void allreduce_mean(float* input, int size, int world_size) {
    MPI_Allreduce(MPI_IN_PLACE, input, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    for (int i = 0; i < size; i++) {
        input[i] = input[i] / world_size;
    }
}


void allgather(float* shard, int shard_size, float* full) {
    MPI_Allgather(
        shard, shard_size, MPI_FLOAT, full, shard_size, MPI_FLOAT, MPI_COMM_WORLD
    );
}


void reducescatter_mean(float* full, float* shard, int shard_size, int world_size) {
    int shard_sizes[world_size];
    for (int i = 0; i < world_size; i++) {
        shard_sizes[i] = shard_size;
    }
    MPI_Reduce_scatter(full, shard, shard_sizes, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    for (int i = 0; i < shard_size; i++) {
        shard[i] = shard[i] / world_size;
    }
}



