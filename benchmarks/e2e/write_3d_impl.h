#ifndef IOSAGE_WRITE_3D_IMPL_H
#define IOSAGE_WRITE_3D_IMPL_H

#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <netcdf.h>
#include <netcdf_par.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define VARIABLE_COUNT 10
#define OUTPUT_PATH_CAPACITY 4096

static int parse_positive_int(const char *text, const char *name, int *value)
{
    char *end = NULL;
    long parsed;

    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
        fprintf(stderr, "%s must be an integer in [1, %d]: %s\n", name, INT_MAX, text);
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

static void abort_mpi(int rank, const char *operation, int status)
{
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;

    MPI_Error_string(status, message, &length);
    fprintf(stderr, "rank %d: %s failed: %.*s\n", rank, operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, status);
}

static void abort_netcdf(int rank, const char *operation, int status)
{
    fprintf(stderr, "rank %d: %s failed: %s\n", rank, operation, nc_strerror(status));
    MPI_Abort(MPI_COMM_WORLD, status);
}

#define CHECK_MPI(call) do { \
    int check_status = (call); \
    if (check_status != MPI_SUCCESS) { \
        abort_mpi(rank, #call, check_status); \
        return EXIT_FAILURE; \
    } \
} while (0)

#define CHECK_NC(call) do { \
    int check_status = (call); \
    if (check_status != NC_NOERR) { \
        abort_netcdf(rank, #call, check_status); \
        return EXIT_FAILURE; \
    } \
} while (0)

static int checked_product(size_t left, size_t right, size_t *result)
{
    if (left != 0 && right > SIZE_MAX / left) {
        return -1;
    }
    *result = left * right;
    return 0;
}

int main(int argc, char **argv)
{
    int npx, npy, npz, ndx, ndy, ndz;
    int rank = -1;
    int nprocs = 0;
    int ncid;
    int dimids[3];
    int var_ids[VARIABLE_COUNT];
    size_t global_dims[3];
    size_t starts[3];
    size_t counts[3];
    size_t local_elements;
    size_t global_elements;
    size_t grid_ranks;
    double *data;
    double start_time;
    double elapsed;
    double gib;
    char filename[OUTPUT_PATH_CAPACITY];
    MPI_Info info;
    static const char *names[VARIABLE_COUNT] = {
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J"
    };

    if (argc != 8) {
        fprintf(stderr, "usage: %s OUTPUT_PREFIX NPX NPY NPZ NDX NDY NDZ\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (parse_positive_int(argv[2], "NPX", &npx) ||
        parse_positive_int(argv[3], "NPY", &npy) ||
        parse_positive_int(argv[4], "NPZ", &npz) ||
        parse_positive_int(argv[5], "NDX", &ndx) ||
        parse_positive_int(argv[6], "NDY", &ndy) ||
        parse_positive_int(argv[7], "NDZ", &ndz)) {
        return EXIT_FAILURE;
    }
    {
        int path_length = snprintf(filename, sizeof(filename), "%s.nc4", argv[1]);
        if (path_length < 0 || path_length >= (int)sizeof(filename)) {
            fprintf(stderr, "output path is too long\n");
            return EXIT_FAILURE;
        }
    }
    if (checked_product((size_t)npx, (size_t)npy, &grid_ranks) ||
        checked_product(grid_ranks, (size_t)npz, &grid_ranks) ||
        grid_ranks > INT_MAX) {
        fprintf(stderr, "process grid size overflows the MPI rank limit\n");
        return EXIT_FAILURE;
    }

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init failed\n");
        return EXIT_FAILURE;
    }
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
    if (grid_ranks != (size_t)nprocs) {
        if (rank == 0) {
            fprintf(stderr, "process grid %dx%dx%d does not match %d MPI ranks\n",
                    npx, npy, npz, nprocs);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    if (checked_product((size_t)ndx, (size_t)ndy, &local_elements) ||
        checked_product(local_elements, (size_t)ndz, &local_elements)) {
        if (rank == 0) {
            fprintf(stderr, "local element count overflows size_t\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    if (local_elements > SIZE_MAX / sizeof(*data)) {
        if (rank == 0) {
            fprintf(stderr, "local allocation size overflows size_t\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    data = malloc(local_elements * sizeof(*data));
    if (data == NULL) {
        fprintf(stderr, "rank %d: cannot allocate %zu doubles\n", rank, local_elements);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    for (size_t index = 0; index < local_elements; ++index) {
        data[index] = (double)rank;
    }

    global_dims[0] = (size_t)npx * (size_t)ndx;
    global_dims[1] = (size_t)npy * (size_t)ndy;
    global_dims[2] = (size_t)npz * (size_t)ndz;
    starts[0] = (size_t)(rank % npx) * (size_t)ndx;
    starts[1] = (size_t)((rank / npx) % npy) * (size_t)ndy;
    starts[2] = ((size_t)rank / ((size_t)npx * (size_t)npy)) * (size_t)ndz;
    counts[0] = (size_t)ndx;
    counts[1] = (size_t)ndy;
    counts[2] = (size_t)ndz;

    CHECK_MPI(MPI_Info_create(&info));
    CHECK_MPI(MPI_Info_set(info, "cb_align", "2"));
    CHECK_MPI(MPI_Info_set(info, "romio_ds_write", "disable"));
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    start_time = MPI_Wtime();

    CHECK_NC(nc_create_par(filename, NC_NETCDF4 | NC_MPIIO | NC_CLOBBER,
                           MPI_COMM_WORLD, info, &ncid));
#if defined(IOSAGE_PATHOLOGICAL)
    CHECK_NC(nc_set_fill(ncid, NC_FILL, NULL));
#elif defined(IOSAGE_OPTIMIZED)
    CHECK_NC(nc_set_fill(ncid, NC_NOFILL, NULL));
    CHECK_NC(nc_set_chunk_cache(134217728, 1009, 0.75));
#endif

    CHECK_NC(nc_def_dim(ncid, "nx", global_dims[0], &dimids[0]));
    CHECK_NC(nc_def_dim(ncid, "ny", global_dims[1], &dimids[1]));
    CHECK_NC(nc_def_dim(ncid, "nz", global_dims[2], &dimids[2]));

    for (int index = 0; index < VARIABLE_COUNT; ++index) {
        CHECK_NC(nc_def_var(ncid, names[index], NC_DOUBLE, 3, dimids, &var_ids[index]));
#if defined(IOSAGE_PATHOLOGICAL)
        size_t chunks[3] = {1, 1, global_dims[2] < 64 ? global_dims[2] : 64};
        CHECK_NC(nc_def_var_chunking(ncid, var_ids[index], NC_CHUNKED, chunks));
#elif defined(IOSAGE_OPTIMIZED)
        CHECK_NC(nc_def_var_chunking(ncid, var_ids[index], NC_CHUNKED, counts));
#endif
    }
    CHECK_NC(nc_enddef(ncid));

    for (int index = 0; index < VARIABLE_COUNT; ++index) {
#if defined(IOSAGE_OPTIMIZED)
        CHECK_NC(nc_var_par_access(ncid, var_ids[index], NC_INDEPENDENT));
#else
        CHECK_NC(nc_var_par_access(ncid, var_ids[index], NC_COLLECTIVE));
#endif
        CHECK_NC(nc_put_vara_double(ncid, var_ids[index], starts, counts, data));
#if defined(IOSAGE_PATHOLOGICAL)
        CHECK_NC(nc_sync(ncid));
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
#endif
    }
    CHECK_NC(nc_close(ncid));
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    elapsed = MPI_Wtime() - start_time;

    if (checked_product(global_dims[0], global_dims[1], &global_elements) ||
        checked_product(global_elements, global_dims[2], &global_elements) ||
        global_elements > SIZE_MAX / VARIABLE_COUNT / sizeof(double)) {
        if (rank == 0) {
            fprintf(stderr, "global byte count overflows size_t\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    gib = ((double)VARIABLE_COUNT * (double)sizeof(double) *
           (double)global_elements) / (1024.0 * 1024.0 * 1024.0);
    if (rank == 0) {
        printf("%s %d %d %d %d %.9f %.9f %.9f\n",
               filename, nprocs, ndx, ndy, ndz, gib, elapsed,
               elapsed > 0.0 ? gib / elapsed : 0.0);
    }

    free(data);
    CHECK_MPI(MPI_Info_free(&info));
    {
        int final_status = MPI_Finalize();
        if (final_status != MPI_SUCCESS) {
            fprintf(stderr, "rank %d: MPI_Finalize failed\n", rank);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

#endif
