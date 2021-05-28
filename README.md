# Antarctic Rift Cataloger
Detect rifts using the ICESat-2 ATL06 data product using the following workflow:

0. Download all of ATL06 locally.  (Hopefully one day it will be on the cloud and this step wont be necessary.)
1. get_file_list_icepyx.py to query metadata and select only ATL06 data from the iceshelves in Antarctica.
2. make_catalog.py to run the rift detector
3. qc.ipynb to look at / quality check the output, make figures
4. analyze_rift_measurements.ipynb to do some calculations, make figures
