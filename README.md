# Automated Image Processing for Scracth Detection on Specular Surfaces
![msc](https://user-images.githubusercontent.com/97564250/232253087-a540fb15-70ed-4099-b99a-9f2bd8c719c8.png)

This code belongs to a thesis work to detect scratch objects on a mirror-like metallic surfaces. The study targets to run on an actual product line in a dishwasher plant. As a results, the challenges are (1) operating in real-time and fast and (2) scrathes hard to spotted by naked eye. The project partner is [Arçelik](https://www.arcelikglobal.com/en/technology/rd/rd-and-design-centers/).

>**Paper** _Okbay, V., Akar, G., & Yaman, U., (2018). Automated Image Processing for Scratch Detection on Specular Surfaces._ International Conference and Exhibition on Digital Transformation and Smart Systems, Ankara, Turkey. Presented at [DTSS 2018](https://dtss.metu.edu.tr), Ankara, Turkey.

>**Note** Here is the URL of [full text](https://open.metu.edu.tr/handle/11511/27527). 

The work done can be divided into three:
| Approach | Directory | Comment |
| :--- | :----: | ---: |
| MATLAB model | [/src/matlab/](/src/matlab/) | Modeling low-cost methods |
| Raspberry Pi implementation | [/src/raspi/](/src/raspi/) | Low-cost method, Hough Transform as core algorithm|
| NVidia Jetson implementation | [/src/ml/](/src/ml/) | Mid-end method, a tiny custom CNN architecture|

## :books: Main Dependencies
PC:
- Tensorflow v1.6.0
- cuDNN v7.04
- OpenCV v3.4.2
- NumPy
- SciPy
- MATLAB v2016a

Raspberry:
- OpenCV v3.4.2
- Picamera v1.13
- NumPy

Jetson:
- [Tensorflow](https://github.com/NVIDIA-Jetson/tf_to_trt_image_classification)
- [OpenCV](https://github.com/jetsonhacks/buildOpenCVTX2)

## :wrench: Hardware
- Raspberry Pi 3B
  - Raspicam v1.2
- NVidia JETSON TX2
  - On-board camera module
