# Segment Anything Model - Mask Generator test

Este script de Python utiliza la librería SAM de Facebook para segmentar una imagen. SAM (Segment Anything with Masking) es una librería de código abierto que permite la segmentación de objetos en imágenes de forma automática. En este script se utiliza SAM para generar máscaras que identifican objetos en una imagen.

![Ejemplo: Faro de Navidad, Cartagena, España](example.png)

# Requerimientos:

matplotlib==3.7.1
numpy==1.24.2
opencv_python==4.7.0.72
segment_anything==1.0

Descargar los checkpoints de modelo de la página de GitHub de segment-anything:

https://github.com/facebookresearch/segment-anything#model-checkpoints

# Instrucciones de uso:

Colocar la imagen que se desea segmentar en la carpeta "image_examples" dentro del directorio donde se encuentra el script.

Establecer los parámetros deseados:
- model_size: Tamaño del modelo SAM que se desea utilizar (small, medium o large).
- device: Dispositivo donde se ejecutará el modelo (cuda:0 para GPU o cpu para CPU).
- image_name: Nombre de la imagen que se desea segmentar.

Ejecutar el script.

El script mostrará la imagen original junto con las máscaras que identifican los objetos en la imagen. También se puede descomentar la sección de código que muestra las máscaras por separado. El tiempo de ejecución se mostrará en la consola.

# License

This project is licensed under the Apache License 2.0. For more information, please refer to the [LICENSE](LICENSE) file in the root of this repository.
Apache License 2.0
Copyright (c) 2023 Alejandro Olivo
