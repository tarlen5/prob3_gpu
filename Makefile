
SHELL = /bin/bash
NVCC = nvcc
EXEC = main

## Flags, includes, linker
# CUDA flags
CUDAFLAGS=-arch sm_20 -O3 --relocatable-device-code=true
# CPP flags
CXXCLAGS=-O3

#FOR DEBUGGING:
#CUDAFLAGS=-arch sm_20 -g -G
#CXXCLAGS=-O0 -g

# Verbose compiler
PTXFLAGS= --ptxas-options=-v -Xptxas -v

# Include files go here
INC =
# Third party libraries to link (-L and -l) go here
LINK =


## Some magic to find all source files
FIND_SRC_CU = $(wildcard $(dir)/*.cu)
FIND_SRC_CPP = $(wildcard $(dir)/*.cpp)

# List directories in which to find source files
DIRS := ./

# Find all CU and CPP files
SRC_CU := $(foreach dir,$(DIRS),$(FIND_SRC_CU))
SRC_CPP := $(foreach dir,$(DIRS),$(FIND_SRC_CPP))

## Object files are just the same as source files, but with .o extension instead
OBJCU= $(SRC_CU:.cu=.o)
OBJCPP= $(SRC_CPP:.cpp=.o)

## Makefile rules
all: $(EXEC)

# Linker -- create executable
$(EXEC): $(OBJCU) $(OBJCPP) 
	$(NVCC) -o $(EXEC) $^ $(CUDAFLAGS) $(LINK) $(PTXFLAGS)

# .cpp -> .o files
%.o: %.cpp
	$(NVCC) -o $@ -c $< $(INC) $(CXXFLAGS)

# .cu -> .o files
%.o: %.cu
	$(NVCC) -o $@ -c $< $(CUDAFLAGS) $(PTXFLAGS) $(INC)


#####
# Cleaning routines
####
.PHONY: clean
clean:
	@echo "Cleaning .o files ..."
	@rm -f $(OBJCU)
	@rm -f $(OBJCPP)
	@echo "...done"


.PHONY: mrpropre
mrpropre: clean
	rm $(EXEC)

