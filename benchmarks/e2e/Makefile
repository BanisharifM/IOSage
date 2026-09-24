# E2E NetCDF-4 write kernels (default / pathological / ultra_optimized).
# Build with the Cray wrapper so cray-mpich, cray-hdf5-parallel and cray-netcdf-hdf5parallel are
# linked from the currently loaded modules:  source modules.sh && make
CC        ?= cc
CFLAGS    ?= -O2 -Wall -Wextra -Werror -std=c11
BUILD_DIR  = build
NAMES      = write_3d_nc4 write_3d_nc4_pathological write_3d_nc4_ultra_optimized
BINS       = $(NAMES:%=$(BUILD_DIR)/%)

all: $(BINS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%: %.c write_3d_impl.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $<

.PHONY: all
