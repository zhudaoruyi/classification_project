# coding:utf-8

# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
#from keras.applications import ResNet50
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
from flask import request
import numpy as np
import flask
import io


def set_gpu(gpu_memory_frac=0.2):
    import tensorflow as tf
    import keras.backend as K

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_frac    # 不全部占满显存, 按给定值分配
    # config.gpu_options.allow_growth=True   # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    K.set_session(sess)


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
CLASS_INDEX = {"0": ["n6556", "LINE_CHART"], "1": ["n4066", "AREA_CHART"], "2": ["n1041", "BAR_CHART"],
               "3": ["n6966", "COLUMN_CHART"], "4": ["n1221", "PIE_CHART"], "5": ["n4838", "UNKNOWN"],
               "6": ["n1104", "GRID_TABLE"], "7": ["n0647", "LINE_TABLE"],"8": ["n0825", "QR_CODE"],
               "9": ["n1420", "INFO_GRAPH"], "10": ["n0335","TEXT"], "11": ["n3294","CANDLESTICK_CHART"],
               "12": ["n1759", "PHOTOS"], "13": ["n1061", "SCATTER_CHART"], "14": ["n0906", "RADAR_CHART"],
               "15": ["n0231", "DONUT_CHART"], "16": ["n0773", "LINE_POINT_CHART"], "17":["n1665", "DISCRETE_PLOT"]}


def get_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
        set_gpu()
        model = load_model('/home/abc/pzw/files/class_22/train0306/models_and_logs/m5032_dn169_v1_l.h5')
#	model = ResNet50(weights="imagenet")

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
        image = np.array(image)
#	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
#	image = imagenet_utils.preprocess_input(image)
        image = image / 255.
	# return the processed image
	return image


def decode_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 18:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 18)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        CLASS_INDEX = {"0": ["n6556", "LINE_CHART"], "1": ["n4066", "AREA_CHART"], "2": ["n1041", "BAR_CHART"],
                       "3": ["n6966", "COLUMN_CHART"], "4": ["n1221", "PIE_CHART"], "5": ["n4838", "UNKNOWN"],
                       "6": ["n1104", "GRID_TABLE"], "7": ["n0647", "LINE_TABLE"],"8": ["n0825", "QR_CODE"], 
                       "9": ["n1420", "INFO_GRAPH"], "10": ["n0335","TEXT"], "11": ["n3294","CANDLESTICK_CHART"],
                       "12": ["n1759", "PHOTOS"], "13": ["n1061", "SCATTER_CHART"], "14": ["n0906", "RADAR_CHART"], 
                       "15": ["n0231", "DONUT_CHART"], "16": ["n0773", "LINE_POINT_CHART"], "17":["n1665", "DISCRETE_PLOT"]}

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
#    im = misc.imread(f) 
#    img = im.reshape((1,784))  
#    clf = joblib.load('model/ok.m')
#    l = clf.predict(img)
    im = Image.open(f)
    im = im.convert("RGB")
    im = im.resize((224, 224))
    im = np.array(im)/255.
    im = np.expand_dims(im, axis=0)
    preds = model.predict(im)
    l = decode_predictions(preds)
#    resu = {}
#    resu["predictions"] = []
    resu = []
    for (_, cl, prob) in l[0]:
        r = {"label": cl, "score": prob}
#        resu["predictions"].append(r)
        resu.append(r)
    return '%s'%resu
#    return 'predictions: \n %s' % resu
#    return ' %s'%resu


@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <title> 图片分类器 | by zhwpeng </title>
    </head>
    <body>
    <h1>图片分类器</h1>
    <form action='/upload' method='post' enctype='multipart/form-data'>
        <input type='file' name='file'>
    <input type='submit' value='Upload'>
    </form>
    ''' 


@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			preds = model.predict(image)
			results = decode_predictions(preds)
			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			for (imagenetID, label, prob) in results[0]:
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	get_model()
	app.run(host="0.0.0.0", debug=False)

