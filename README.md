Vineyard segmenter using high resolution images (25cm/px)
===============================================================

# Objective

The aim of this project is to train a semantic segmentation model that can detect vineyards in high-resolution aerial
images of 25cm from spanish [Plan Nacional de Ortofotografía Aérea (PNOA)](https://pnoa.ign.es/) project. To achieve
this, you will use a dataset of labeled images that includes both vineyard images and other types of crops and terrains.
Once the model is trained, it will be applied to the raster using window sliding
to obtain a mask with the probabilities of vineyard usage in the patch. After this, the masks will be vectorized to
obtain a shapefile, that will be filtered and simplified.

**This is just a prototype.**

# Dataset

* High resolution aerial images (25cm/px) from
  spanish [Plan Nacional de Ortofotografía Aérea (PNOA)](https://pnoa.ign.es/), a restricted área from the region of
  Castilla y León is used in this project to train the classifier. The images for year 2020 in the area of Castilla y
  León can be accessed [here](http://ftp.itacyl.es/cartografia/01_Ortofotografia/2020/).
* LPIS: Land parcel Information System, feature files containing parcels in the area of the images, these features are
  used to manually select the parcels with vineyeard usage to extract patches for each category (0-no vineyard
  1-vineyard). These files can be downloaded for the area of Castilla y León
  from [here](http://ftp.itacyl.es/cartografia/05_SIGPAC/2020_ETRS89/Parcelario_SIGPAC_CyL_Municipios/).



The input image size of the Unet is 256x256x3. To make sure avery detail 
