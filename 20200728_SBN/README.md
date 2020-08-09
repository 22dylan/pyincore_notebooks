### 20200728_SBN

The notebooks contained in this directory are part of a spatial Bayesian network (SBN) that computes parcel-level resilience. **If one  wants to run the SBN, then download this directory and CPT.h5 file at https://oregonstate.box.com/s/xzql2dhgzp7utqhcobgjyi9bitwnulq1. Ensure that the jupyter notebook (SBN.ipynb) and the CPT file (CPTs.h5) are in the same directory.** 

The entire process of performing the damage analysis, populating the CPT files and setting up the SBN is documented here. There are three primary sets of notebooks, organized into the following directories:

1. Damage infrastructure using pyIncore - Damage codes were developed to damage four infrastructure systems in Seaside (buildings, electric, transportation, and water). Where applicable, these damage codes used pyIncore.
2. Develop operability curves and populate conditional probability tables (CPTs) - The results from the damage analysis were used to generate operability curves. Statistics on the operability curves were extracted to populate the CPTs.
3. Generate the SBN - A Bayesian network was constructed at each parcel using the resulting CPTs.
4. Generate parcel-level maps of resilience - The SBN can then be used as a decision support tool to evaluate parcel-level resilience.

This entire process is outlined in the figure below. The notebook in this directory, however, demonstrates the final two steps highlighted in red. That is, a spatial Bayesian network is constructed and it is used to generate maps of resilience.


![framework](./images/framework.png)