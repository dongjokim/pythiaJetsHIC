#!/bin/bash

# need to have correct $ROOTSYS, $FASTJET and $PYTHIA8 are needed in Makefile
# In case you have installed all from Aliroot installation including fastjet packages based on https://dberzano.github.io/alice/install-aliroot/
# no need to setup $ROOTSYS, $FASTJET need to have only $PYTHIA8

# Source ROOT environment
source ~/softwares/root_install/bin/thisroot.sh
#source /usr/local/bin/thisroot.sh

# Set Pythia8 environment
export PYTHIA8=$HOME/softwares/pythia8312
export PATH=${PATH}:$PYTHIA8/bin

# Set FastJet environment
export FASTJET=$HOME/softwares/fastjet-3.4.0/insall
export PATH=$PATH:${FASTJET}/bin

# Set library paths without duplicates
# Create a function to add paths without duplicates
add_libpath() {
    if [ -z "$DYLD_LIBRARY_PATH" ]; then
        export DYLD_LIBRARY_PATH="$1"
    else
        echo "$DYLD_LIBRARY_PATH" | grep -q "$1" || export DYLD_LIBRARY_PATH="$1:$DYLD_LIBRARY_PATH"
    fi
}

# Add library paths
add_libpath "$PYTHIA8/lib"
add_libpath "$FASTJET/lib"
add_libpath "$ROOTSYS/lib"


