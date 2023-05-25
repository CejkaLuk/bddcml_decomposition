#*********************************
# the whole link sequence for BDDC
#*********************************
BDDCML_LINK_SEQUENCE = $(LIBBDDCML) $(BDDCML_LIBMUMPS) $(BDDCML_LIBPARMETIS) $(BDDCML_LIBMETIS) $(BDDCML_LIBSCALAPACK) $(BDDCML_LIBBLACS) \
                       $(BDDCML_LIBPORD) $(BDDCML_LIBMAGMA) $(BDDCML_LIBLAPACK) $(BDDCML_LIBBLAS) $(BDDCML_BLOPEXLINK) \
	                    $(BDDCML_LIBTNL_BDDCML_INTERFACE) $(BDDCML_LIBOTHERS) 
BDDCML_INC           = -I$(BDDCML_ROOT)/src

#*********** Compilers
BDDCML_MPFC = mpif90
BDDCML_MPFFLAGS = -g -C -Wall -fbounds-check
#BDDCML_MPFFLAGS = -O3

BDDCML_MPCC = mpicc
BDDCML_MPCFLAGS = -g -Wall
#BDDCML_MPCFLAGS = -O3

#*********** Library tools
BDDCML_AR = ar vr
BDDCML_RANLIB = ranlib

#*********** Directories
# Project dir
PROJ_DIR = $(HOME)/github/sistek
DEPS_DIR = $(PROJ_DIR)/deps
MUMPS_DIR = $(DEPS_DIR)/MUMPS_5.5.1
MAGMA_DIR = $(DEPS_DIR)/magma-2.6.2
TNL_BDDCML_INTERFRACE_DIR= $(DEPS_DIR)/tnl_bddcml_interface

#*********** Includes
# ParMetis
BDDCML_INCPARMETIS = -I/usr/include
# Metis
BDDCML_INCMETIS = -I/usr/include
# MUMPS
BDDCML_INCMUMPS    = -I$(MUMPS_DIR)/include
# MAGMA - leave undefined if not needed
BDDCML_WITH_MAGMA = 1
#BDDCML_INCMAGMA    =
BDDCML_INCMAGMA    = -I$(MAGMA_DIR)/include
# TNL_BDDCML_INTERFACE
BDDCML_INCTNL_BDDCML_INTERFACE = -I$(TNL_BDDCML_INTERFRACE_DIR)

#*********** Libraries
# library being created
BDDCML_ROOT         = $(PROJ_DIR)/bddcml
BDDCML_LIB_DIR      = $(BDDCML_ROOT)/lib
LIBBDDCML           = $(BDDCML_LIB_DIR)/libbddcml.a
# MUMPS
BDDCML_LIBMUMPS     = -L$(MUMPS_DIR)/lib -ldmumps -lmumps_common -lpord
# ScaLAPACK
BDDCML_LIBSCALAPACK = -L/usr/lib -lscalapack-openmpi
# BLACS
BDDCML_LIBBLACS     = # -L/usr/lib -lblacs-openmpi -lblacsCinit-openmpi -lblacsF77init-openmpi -lblacs-openmpi
# BLAS
BDDCML_LIBBLAS      = -L/usr/lib -lblas
# LAPACK
BDDCML_LIBLAPACK    = -L/usr/lib -llapack
# MAGMA - leave undefined if not needed
#BDDCML_LIBMAGMA    = 
BDDCML_LIBMAGMA    = -L$(MAGMA_DIR)/lib -lmagma
# ParMetis
BDDCML_LIBPARMETIS  = -L/usr/lib -lparmetis
# Metis
BDDCML_LIBMETIS     = -L/usr/include -lmetis
# other needed libs ( e.g. -lmpi, -lpthread )
BDDCML_LIBOTHERS    = -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver -lstdc++
# TNL_BDDCML_INTERFACE
BDDCML_LIBTNL_BDDCML_INTERFACE = -L$(TNL_BDDCML_INTERFRACE_DIR) -ltnl_bddcml_interface

# BLOPEX objects
BDDCML_BLOPEXDIR = $(DEPS_DIR)/blopex
BDDCML_INCBLOPEX = \
   -I$(BDDCML_BLOPEXDIR)/blopex_serial_double/multivector \
   -I$(BDDCML_BLOPEXDIR)/blopex_serial_double/pcg_multi \
   -I$(BDDCML_BLOPEXDIR)/blopex_serial_double/matmultivec \
   -I$(BDDCML_BLOPEXDIR)/blopex_abstract/include
BDDCML_OBJBLOPEX = \
   $(BDDCML_BLOPEXDIR)/blopex_serial_double/multivector/multi_vector.o \
   $(BDDCML_BLOPEXDIR)/blopex_serial_double/pcg_multi/pcg_multi.o \
   $(BDDCML_BLOPEXDIR)/blopex_serial_double/matmultivec/matmultivec.o
BDDCML_LIBBLOPEX = -L$(BDDCML_BLOPEXDIR)/blopex_abstract/lib -lBLOPEX
BDDCML_BLOPEXLINK = $(BDDCML_OBJBLOPEX) $(BDDCML_LIBBLOPEX)

# MAGMA library
ifeq ($(strip $(BDDCML_WITH_MAGMA)), 1)
   BDDCML_DEFS += -DBDDCML_WITH_MAGMA
endif
