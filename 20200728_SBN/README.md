### 20200728_SBN

The notebooks contained in this directory are part of a spatial Bayesian network (SBN) that computes parcel-level resilience. There are three primary sets of notebooks, each organized into unique directories:

1. 1_DamageCodes - uses pyIncore to estiamte damage state probabilities for infrastructure components. The damage state probabilities are used to inform Monte-Carlo simulation (MCS), repair time estimates, and (where applicable) connectivity analyses.  
2. 2_CPTsH5 - used to extract results from the codes above and populate the SBN's conditional probability tables (CPTs). A single hdf5 file is generated for all parcels in Seaside. 
3. 3_SBN - a notebook that uses the SBN as a decision support tool. The SBN relies on the CPTs produced from the previous code. A copy of the pre-populated CPT.h5 file can be access at (...).


The notebooks rely on pyincore's building, electric, transportation, and water damage capabilities.

A conceptual framework of the


This notebook is used to demonstrate pyincore's multi-hazard damage capabilities. Computes building economic losses, risks, and building failure. Uses seismic-tsunami hazard at Seaside, Oregon as a testbed community. Demonstrates a variety of plotting methods including interactive spatial maps. 

