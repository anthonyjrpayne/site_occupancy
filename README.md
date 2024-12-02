# site_occupancy

Code to determine the probability distribution of an ab initio molecular dynamics trajectory from a CASTEP '.md' file output.
Accounts for both the dynamic movement of the underlying surface atoms and the adsorbed molecule moving outside of the simulation cell.

The Example.py file illistrates an example of it's usage applied to the file 'Example.md'.
Funcitons are called from the 'site_utiliteis.py' file.
Atoms of interst and other veraibles are specified first before caculation.

Note much of the code is specific to hexagonal boron nitride unit cells.
Future improvements include applicability to a general unit cell.

