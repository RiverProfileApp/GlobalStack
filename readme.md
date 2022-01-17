GlobalStack is a set of routines for conditioning DEMs at the global scale for flow routing, and for calculating drainage area. At the moment the outputs of the routines and notebooks are specifically tailored to RiverProfileApp.  
In other words, any outputs will be grids that can be specifically used by the app. To get started, it is recommended to follow in "global_tiled_linear.ipynb" 

Anyone interested in building their own large scale flow routing and drainage accumulation grids should follow the steps outlined in 
 the global_tiled_linear notebook.  However keep in mind that the initial steps (formatting the receiver grid) assume you are using HydroSHEDS (or some other receiver grids) - so you already have some flow direction grid handy.
If that is not the case, you will need to run the create_pit_filled.py function, ideally on a relatively fast computer, in order to remove pits and conidition the DEM.

The routines outlined in global_stack_functions.py could be of universal use, however.  I added the sphinx documentation for them here:
 riverprofileapp.github.io/sphinx_index
