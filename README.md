Vineyard segmenter using high resolution images (25cm/px)
===============================================================

# Objective

The aim of this project is to train a semantic segmentation model that can detect vineyards in high-resolution aerial
images of 25cm from spanish [Plan Nacional de Ortofotografía Aérea (PNOA)](https://pnoa.ign.es/) project. To achieve
this, you will use a dataset of labeled images that includes both vineyard images and other types of crops and terrains.
Once the model is trained, it will be applied to the raster using window sliding to obtain a mask with the probabilities
of vineyard usage in the patch. After this, the masks will be vectorized to obtain a shapefile, that will be filtered
and simplified.

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

Pasos para la creación del dataset:

- Seleccionar un área de referencia y representarlo como un rectángulo en la capa resources/dataset/samples.extractions.
  Estos polígonos tienen como atributos el número de imágenes a samplear en la zona para las categorías 0 (background),
  1 (vineyard) y mixed (representación de ambas con un mínimo de pixels de cada categoría),
- Dentro de este polígono anotar las zonas de viñedo creando polígonos en la capa resources/dataset/samples.extractions.
- Descargar los rasters que contienen los ficheros de extracción utilizando el script data/run_download_sources.py. Este
  script va a consultar la capa `extractions` y descargar en el directorio PNOA_TILES los .tiff necesarios para extraer
  las imágenes.
- Crear los tiff de extracciones intersectando la capa samples.extractions con los rasters de referencia y almacenarlos
  en dataset_folder/extractions. En la capa de referencia se identificad el fichero raster original y el fichero de la
  extracción.
- De cada extracción se samplean tantas imágenes como se indique en la capa extractions.
- Crear el dataset con el script dataset/run.py

# Configurations

In the file vsegmenter/cfg.py hay que configurar estas variables:
CARTOGRAPHY_BASE_FOLDER: as DATASET_FOLDER PNOA_BASE_FOLDER

tematicos/figuras_calidad_vino/20200813_DO_VINO.shp"

### Iteration v1

Iteration 1, Unet Trained for 50 epochs using just 500 images with at least a 30% of each category. loss(
categorical_crossentropy)?=0.0378 val_mean_iou=0.938 Lets add more samples iwth images with just 0 category

- Model is performing suprisely well
- Trees have been unrepresented and crops with linear marks produced by tractor tracks. Add these images as 0-categories
  and create dataset v2

### Iteration v2

![](resources/assets/v2_eval1.png)

- Se ha añadido al dataset ejemplos totalmente positivos y totalmente negativos para represnetar zonas de cultivo y
  árboles. Los resultados bastante buenos, es ha aplicado el modelo en zonas no utilizadas para crear el dataset
  original y es capaz de segmentar el viñedo. Además mejora significativamente la predicción en las zonas en las que no
  hay presentación de viñedo.

![](resources/assets/v2_eval2.png)

- Si embargo tiene problemas en detectar tipos de viñedo no vistos, se han detectado tres problemas:
    - Viñedos con poca cobertura vegetal
    - Viñedos con estructrua poco regular, por ejemplo con sistemas de conducción diferenes a la espaldera.
    - Zonas con árboles y frutales
    - Viñedos con terreno más oscuro.

| <img src="resources/assets/v2_problems1.png" width="300"/>
| <img src="resources/assets/v2_problems2.png" width="300"/>
| <img src="resources/assets/v2_problems4.png" width="300"/> |

Para la tercera iteración se va a crear un dataset utilizando nuevos ejemplos de estas zonas y se va a re-entrenar el
modelo existente.

### Iteration v3

Creando un nuevo dataset que incluye las zonas no representadas mejora significativamente las prestaciones del modelo:

| <img src="resources/assets/v3_eval1_a.png" width="300"/> | <img src="resources/assets/v3_eval1_b.png" width="300"/> |
|-----|-----|
| <img src="resources/assets/v3_eval2_a.png" width="300"/> | <img src="resources/assets/v3_eval2_b.png" width="300"/> |
| <img src="resources/assets/v3_eval3_a.png" width="300"/> | <img src="resources/assets/v3_eval3_b.png" width="300"/> |

Los resultados son bastante buenos, pero [evaluando e inspeccionando los resultados ](notebooks/eval_model.ipynb) se
puede ver que el modelo presenta problemas con dos tipos de patrones:

| Zonas boscosas con patrones similares al viñedo |    <img src="/media/gus/workspace/wml/vineyard-segmenter/resources/assets/v3_eval4_a.png" width="300"/> |
|-------------------------------------------------|-----|
| Zonas de viñedo con poca vegetación             |    <img src="/media/gus/workspace/wml/vineyard-segmenter/resources/assets/v3_eval4_b.png" width="300"/> |

| epoch | categorical_accuracy | categorical_crossentropy | loss | mean_iou | val_categorical_accuracy | val_categorical_crossentropy | val_loss | val_mean_iou |
|-------|----------------------|---------------------------|------|----------|---------------------------|--------------------------------|----------|--------------|
| 197   | 0.9917               | 0.0213                   | 0.0213 | 0.8520   | 0.9931                    | 0.0210                         | 0.0210   | 0.8549       |

Dataset actual train = 1372 test = 343 con imágenes de 128x128

### Iteration v4

Creamos co En este punto, para mejorar el modelo

- Incrementar conjunto de entrenamiento, especialmente ejemplos negativos con tierras con poca vegetación.
- Añadir técnicas de augmención de datos
- Probar diferentes arquitecturas para el backbone
- Añadir capas de regularización

## Aumentar conjunto de datos sobre ejemplos con problemas

# References

@article{akeret2017radio, title={Radio frequency interference mitigation using deep convolutional neural networks},
author={Akeret, Joel and Chang, Chihway and Lucchi, Aurelien and Refregier, Alexandre}, journal={Astronomy and
Computing}, volume={18}, pages={35--39}, year={2017}, publisher={Elsevier} }