# MADP-Toolbox-Python3-Parser
# MADP Toolbox is an excellent C++ framework optimized for developing decentralized algorithms in control theory and reinforcement learning.
# The library in this repository is a Python 3 wrapper for the MADP Toolbox parser, that can allow to read problem files such as dectiger.dpomdp and many others from the multiagent control domain.

# To compile C++ and build the python bindings:
# sudo python3 setup.py develop

# The wrapper was tested with Python 3.5
# After the compilation, run: python3 test.py to run all the MADP problem reading interface functions

# The C++ version of the wrapper was taken from: https://github.com/laurimi/npgi
# Python bindings are created with the help of: https://github.com/pybind/pybind11
# Original and complete MADP is taken from: https://github.com/MADPToolbox/MADP