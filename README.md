# RDIR Compiler Plug-in

RDIR is a dataflow intermediate representation for optimizing
compilers for parallel programming models. This compiler plug-in adds
RDIR support to Regent, a compiler for task-based parallel programs,
with a translator (both AST->RDIR and RDIR->AST) and a number of
sample optimizations which demonstrate how to use the plug-in.

## Links
  * [Regent](https://github.com/StanfordLegion/legion/tree/regent-0.0/language)
  * [Legion](https://github.com/StanfordLegion/legion)
  * [Terra](https://github.com/zdevito/terra)

## Installation

 1. Install LLVM *with headers*. (Tested with LLVM 3.5.)
 2. Download and install Regent:

        git clone -b rdir --recursive https://github.com/StanfordLegion/legion.git
        cd legion/language
        ./install.py --debug

    Note: In some cases, Terra may fail to auto-detect CUDA. If so
    (and assuming you want to use CUDA), recompile Terra with CUDA
    enabled.

        cd terra
        make clean
        CUDA_HOME=.../path/to/cuda ENABLE_CUDA=1 make

## Usage

The following test is included which exercises the optimizations
included with RDIR. Both CPU and GPU code paths should be
working. (GPUs are enabled with the flag -ll:gpu N where N is the
number of GPUs to use.)

Running the following commands will produce output reporting bandwidth
and compute throughput on a simple streaming benchmark. In general,
inc1 and inc2 should achieve roughly the same bandwidth. When the RDIR
optimizations are functioning propertly, inc2 will achieve 2x the
compute throughput as inc1 (due to loop and task fusion).

    ./regent.py src/rdir/examples/inc.rg -fcuda 1 -ll:csize 4096
    ./regent.py src/rdir/examples/inc.rg -fcuda 1 -ll:gpu 1 -ll:csize 4096 -ll:fsize 4096

# Open Source License

The initial versions of files recorded in this repository are licensed
under the BSD license, copyright (c) 2015 NVIDIA Corporation.

All contributions following the commit below are dual-licensed under
the BSD and Apache version 2.0 licenses. Copyright holders are
recorded in each file.

commit 7871060eeba2726cd77605b39c2c6e18c582b092
Author: sidelnik <sidelnik@gmail.com>
Date:   Tue Oct 13 13:44:43 2015 -0700

Copies of the licenses respective are available in LICENSE_BSD.txt and
LICENSE_APACHE.txt.
