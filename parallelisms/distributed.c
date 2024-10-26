#include <mpi.h>
#include <unistd.h>

#define rank0_printf(rank, ...) if (rank == 0) { printf(__VA_ARGS__); }


typedef struct {
    int tp_rank;
    int tp_size;
    MPI_Group tp_group;
    MPI_Comm tp_comm;

    int dp_rank;
    int dp_size;
    MPI_Group dp_group;
    MPI_Comm dp_comm;

    int pp_rank;
    int pp_size;
    MPI_Group pp_group;
    MPI_Comm pp_comm;

    int world_rank;
    int world_size; 
} Dist;



Dist* Dist_create(int tp_size, int dp_size, int pp_size) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (tp_size * dp_size * pp_size != world_size) {
        rank0_printf(
            world_rank, 
            "Invalid distributed environment: tp=%d * dp%d * pp=%d != world_size=%d\n", 
            tp_size, dp_size, pp_size, world_size
        );
        MPI_Finalize();
        exit(0);
    }
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group tp_group = NULL; MPI_Comm tp_comm = NULL;
    MPI_Group dp_group = NULL; MPI_Comm dp_comm = NULL;
    MPI_Group pp_group = NULL; MPI_Comm pp_comm = NULL;

    // Derive process group rank.
    int tp_rank = world_rank % tp_size;
    int dp_rank = (world_rank / tp_size) % dp_size;
    int pp_rank = world_rank / (tp_size * dp_size);

    if (tp_size > 1) {
        int tp_group_ranks[tp_size];
        for (int i = 0; i < tp_size; i++) {
            tp_group_ranks[i] = pp_rank * (tp_size * dp_size) + dp_rank * tp_size + i;
        }
        tp_group = malloc(sizeof(MPI_Group)); tp_comm = malloc(sizeof(MPI_Comm));
        MPI_Group_incl(world_group, tp_size, tp_group_ranks, &tp_group);
        MPI_Comm_create(MPI_COMM_WORLD, tp_group, &tp_comm);
   }

    if (dp_size > 1) {
        int dp_group_ranks[dp_size];
        for (int i = 0; i < dp_size; i++) {
            dp_group_ranks[i] = pp_rank * (tp_size * dp_size) + i * tp_size + tp_rank;
        }
        dp_group = malloc(sizeof(MPI_Group)); dp_comm = malloc(sizeof(MPI_Comm));
        MPI_Group_incl(world_group, dp_size, dp_group_ranks, &dp_group);
        MPI_Comm_create(MPI_COMM_WORLD, dp_group, &dp_comm);  
    }

    if (pp_size > 1) {
        int pp_group_ranks[pp_size];
        for (int i = 0; i < pp_size; i++) {
            pp_group_ranks[i] = i * (tp_size * dp_size) + dp_rank * tp_size + tp_rank;
        }
        pp_group = malloc(sizeof(MPI_Group)); pp_comm = malloc(sizeof(MPI_Comm));
        MPI_Group_incl(world_group, pp_size, pp_group_ranks, &pp_group);
        MPI_Comm_create(MPI_COMM_WORLD, pp_group, &pp_comm);
    }

    Dist* self = malloc(sizeof(Dist));
    self->tp_rank = tp_rank;
    self->tp_size = tp_size;
    self->tp_group = tp_group;
    self->tp_comm = tp_comm;
    self->dp_rank = dp_rank;
    self->dp_size = dp_size;
    self->dp_group = dp_group;
    self->dp_comm = dp_comm;
    self->pp_rank = pp_rank;
    self->pp_size = pp_size;
    self->pp_group = pp_group;
    self->pp_comm = pp_comm;
    self->world_rank = world_rank;
    self->world_size = world_size;
    return self;
}


void send(float* input, int input_size, int to_rank, MPI_Comm pg_comm) {
    MPI_Send(input, input_size, MPI_FLOAT, to_rank, 0, pg_comm);
}


void recv(float* output, int output_size, int from_rank, MPI_Comm pg_comm) {
    MPI_Status status;
    MPI_Recv(output, output_size, MPI_FLOAT, from_rank, 0, pg_comm, &status);
}


void allgather(float* shard, int shard_size, float* full, MPI_Comm pg_comm) {
    MPI_Allgather(
        shard, shard_size, MPI_FLOAT, full, shard_size, MPI_FLOAT, pg_comm 
    );
}


void reducescatter_mean(
    float* full, float* shard, int shard_size, MPI_Comm pg_comm, int pg_size
) {
    int shard_sizes[pg_size];
    for (int i = 0; i < pg_size; i++) {
        shard_sizes[i] = shard_size;
    }
    MPI_Reduce_scatter(full, shard, shard_sizes, MPI_FLOAT, MPI_SUM, pg_comm);
    for (int i = 0; i < shard_size; i++) {
        shard[i] = shard[i] / pg_size;
    }
}


void allreduce_mean(float* input, int size, MPI_Comm pg_comm, int pg_size) {
    MPI_Allreduce(MPI_IN_PLACE, input, size, MPI_FLOAT, MPI_SUM, pg_comm);
    for (int i = 0; i < size; i++) {
        input[i] = input[i] / pg_size;
    }
}
