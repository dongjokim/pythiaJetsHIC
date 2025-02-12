# Compiler and base flags
CXX = g++
CXXFLAGS = -O2 -Wall -fPIC -std=c++17 -Wno-deprecated-declarations -D_GLIBCXX_USE_CXX11_ABI=0

# Add FastJet compatibility flags
COMPAT_FLAGS = -D_BACKWARD_BACKWARD_WARNING_H -DFASTJET_LEGACY_ALLOW_AUTOPTR -DFASTJET_ALLOW_DEPRECATED -I.

# ROOT flags and libs
ROOTCFLAGS    = $(shell root-config --cflags)
ROOTLIBS      = $(shell root-config --libs)
ROOTGLIBS     = $(shell root-config --glibs)

# FastJet configuration (using environment variable)
FASTJET_INCLUDE = $(FASTJET)/include
FASTJET_LIB = $(FASTJET)/lib
FASTJET_FLAGS = -I$(FASTJET_INCLUDE)
FASTJET_LIBS = -L$(FASTJET_LIB) -lfastjet

# Pythia8 configuration (using environment variable)
PYTHIA8_INCLUDE = $(PYTHIA8)/include
PYTHIA8_LIB = $(PYTHIA8)/lib
PYTHIA8_FLAGS = -I$(PYTHIA8_INCLUDE)
PYTHIA8_LIBS = -L$(PYTHIA8_LIB) -lpythia8

# Combine all flags
ALL_FLAGS = $(CXXFLAGS) $(COMPAT_FLAGS) $(ROOTCFLAGS) $(PYTHIA8_FLAGS) $(FASTJET_FLAGS)
LIBS = $(ROOTLIBS) $(PYTHIA8_LIBS) $(FASTJET_LIBS)

# Source files and targets
ANALYSIS_TARGET = pythiajets
ANALYSIS_SRCS = pythiajets.cpp

# Target rules
$(ANALYSIS_TARGET): $(ANALYSIS_SRCS)
	$(CXX) $(ALL_FLAGS) -o $@ $< $(LIBS) -DSTANDALONE_MODE

clean:
	rm -f $(ANALYSIS_TARGET)
	rm -f *.o
	rm -f *~
	rm -f *.d

.PHONY: all clean