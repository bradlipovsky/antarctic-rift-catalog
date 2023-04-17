# Antarctic Rift Cataloger
[![DOI](https://zenodo.org/badge/278439106.svg)](https://zenodo.org/badge/latestdoi/278439106)

Detect and measure rifts using ICESat-2 ATL06/SlideRule data product:

0. Download all of ATL06 locally (Integration with SlideRule is a work in progress)
1. get_file_list_icepyx.py to query metadata and select only ATL06 data from the ice shelves in Antarctica
2. make_catalog.py or test_the_detector.ipynb to run the rift measurement algorithm (from arc.py)
3. qc.ipynb to quality control the output, including plotting each rift detection/measurement on ICESat-2 heights and on Landsat imagery with a short temporal offsets, and plots to compare manual and automated rift detection/measurement success (manuscript table)
4. Analyze_rift_measurements.ipynb to do some calculations, including advection correction and comparison to a simple model of rift opening
5. Figures_1_3_S2_S3_S4.ipynb, Figures_2.ipynb, Figures_4_5_6_S1.ipynb, Figures_S15.ipynb and Figures_S7_to_S14 do a small additional data processing (e.g. processing Global Navigation Satellite System data, rotating rift measurements from satellite geometry to rift perpendicular geometry) and make manuscript and supplementary material figures.

Notebooks/scripts Milne.ipynb, analyze_rift_measurements.ipynb, analyze_rift_measurements_continent.ipynb, get_pts_from_geotiff.py, plot_moa.ipynb should be considered as work in progress or obsolete
