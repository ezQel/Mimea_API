
######## Tensorflow Image Object Detection #########
#
# Author: Ezekiel Kalama
# Date: 5/04/19
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the detected leaves in the image.
# The modified Image is sent back to an android front end/client with some details of the detected leaves

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py


# Import visualization and label_map utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


# REST and DB related imports
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow


# object detection related Imports
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since this script is stored in the object_detection folder.
sys.path.append("..")



def do_inference_on_image(IMAGE_NAME):

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'
	

	

    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'labelmap.pbtxt')

    
    # Number of classes the object detector can identify
    NUM_CLASSES = 9

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `1`, we know that this corresponds to `mangifera indica`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.70)

    detected = ([category_index.get(value) for index, value in enumerate(classes[0]) if scores[0, index] > 0.7])
  

    return image, detected



##### REST API#########
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, "db.sqlite")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# init Db
db = SQLAlchemy(app)

# init marshmallow
ma = Marshmallow(app)


# Species DB Table Class
class Species(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    latin_name = db.Column(db.String(100), unique=True)
    english_name = db.Column(db.String(100))
    description = db.Column(db.String(1000))

    def __init__(self, latin_name, english_name, description):

        self.latin_name = latin_name
        self.english_name = english_name
        self.description = description


# Species Schema
class SpeciesSchema(ma.Schema):
    class Meta:
        fields = ('id', 'latin_name', 'english_name', 'description')

# Init Schema
specie_schema = SpeciesSchema()
species_schema = SpeciesSchema(many=True)


# Add a Species to sqlite DB
@app.route('/addspecies', methods=['POST'])
def add_species():
    latin_name = request.json['latin_name']
    english_name = request.json['english_name']
    description = request.json['description']

    new_species = Species(latin_name, english_name, description)

    db.session.add(new_species)
    db.session.commit()

    return specie_schema.jsonify(new_species)


# Image upload html form for testing
@app.route('/')
def upload_file():
    return '''<html>
   <body>
      <form action = "/uploader" method = "POST" 
         enctype = "multipart/form-data">
         <input type = "file" name = "file" />
         <input type = "submit"/>
      </form>
   </body>
</html>'''



# image upload handler
@app.route('/uploader', methods=['POST'])
def upload_files():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename("upload.jpg"))
        img, det = do_inference_on_image("upload.jpg")
        cv2.imwrite("upload.jpg", img)

        if det.__len__()>0:
            plant = det[0]
            id1 = plant["id"]
            info = Species.query.get(id1)
            return specie_schema.jsonify(info)
        else:
            return jsonify({"english_name" : "no plant detected", "latin_name" : "null plantae", "description":"No plant has been detected. please try again"})
                     

# get all Plants
@app.route('/allspecies', methods=['GET'])
def get_all():
    all_species = Species.query.all()
    result = species_schema.dump(all_species)

    return jsonify(result.data)


# get single plant
@app.route('/allspecies/id=<id>', methods=['GET'])
def get_specie(id):
    species = Species.query.get(id)

    return specie_schema.jsonify(species)


# delete a plant
@app.route('/remove/id=<id>', methods=['DELETE'])
def delete_specie(id):

    plant = Species.query.get(id)

    db.session.delete(plant)
    db.session.commit()

    return "<html><b><h1>DELETED SUCCESSFULY</h1></b></html>"


@app.route('/upload.jpg', methods=['GET'])
def get_image():
    return send_file("upload.jpg", mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
