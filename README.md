# ATMOS Flux Data

This is a collection of tools for eddy covariance flux processing at the ATMOS prairie site at Argonne National Laboratory.

Raw flux processing is done in EddyPro.

Scripts 0-3 perform flux post processing on EddyPro outputs: 
- QA/QC
- UStar filtering
- MDS gap-filling
- NEE partitioning to GPP/Reco

R scripts based on code from Micrometeorology Lab at University of British Columbia <br/>
https://github.com/dng55/Manitoba_Flux_Processing/

4.methanegapfill (Irvin et al. 2021):
- predictor set selection
- prepares machine learning algorithms
- gap-fills methane flux data

atmos-FFP:
- applies the Kljun flux footprint model (Kljun et al. 2015)
- weighs fluxes by footprint following Rey-Sanchez et al. 2022
- matches footprint-weighed flux maps with ATMOS soil array

atmos-spatial matches footprint-weighed flux maps with satellite imagery (currently National Agriculture Imagery Program via Google Earth Engine)

<br/>

Irvin, Jeremy, Sharon Zhou, Gavin McNicol, Fred Lu, Vincent Liu, Etienne Fluet-Chouinard, Zutao Ouyang et al. "Gap-filling eddy covariance methane fluxes: Comparison of machine learning model predictions and uncertainties at FLUXNET-CH4 wetlands." _Agricultural and forest meteorology_ 308 (2021): 108528.

Kljun, Natascha, P. Calanca, M. W. Rotach, and Hans Peter Schmid. "A simple two-dimensional parameterisation for Flux Footprint Prediction (FFP)." _Geoscientific Model Development_ 8, no. 11 (2015): 3695-3713.

Rey‐Sanchez, Camilo, Ariane Arias‐Ortiz, Kuno Kasak, Housen Chu, Daphne Szutu, Joseph Verfaillie, and Dennis Baldocchi. "Detecting hot spots of methane flux using footprint‐weighted flux maps." _Journal of Geophysical Research: Biogeosciences_ 127, no. 8 (2022): e2022JG006977.

