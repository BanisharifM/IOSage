# E2E NetCDF-4 write kernels (default / pathological / ultra_optimized).
# Build with the Cray wrapper so cray-mpich, cray-hdf5-parallel and cray-netcdf-hdf5parallel are
# linked from the currently loaded modules:  source modules.sh && make
CC      ?= cc
CFLAGS  ?= -O2
BINS     = write_3d_nc4 write_3d_nc4_pathological write_3d_nc4_ultra_optimized

all: $(BINS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
