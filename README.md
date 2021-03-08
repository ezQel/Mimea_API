# Mimea_Server (A plant Species Identification Program.)
This program utilizes [tensorflow object detection](https://github.com/tensorflow/models/tree/master/research/object_detection) to recognize the species of plants when provided with an image of their leaves. It consists of a flask server that allows a user to POST an image of a plant and returns the details of the plant in json.

## Getting Started
1. Install the following python packages using `pip -r requirements.txt`:
    * Tensorflow 2.4.0
    * Opencv-python 4.5.1.48
    * flask 1.1.2
    * flask-sqlalchemy 2.4.4
    * flask-marshmallow 0.14.0
    * Numpy 1.19.2 

1. Follow the intructions [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) to install [Tensorflow Object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

1. Run server_script.py `python server_script.py`

1. The server will run locally at `localhost:500`. Test the program by uploading an image of a leaf from the web page at `localhost:5000`

### List of recognized plants
the program can identify the following plants:
*  mangifera indica
* rubus niveus
* caesalpinia decapetala
* psidium guajava
* eriobotrya japonica
* garcinia livingstonei
* monodora myristica
* ricinus communis
* tithonia diversifolia
