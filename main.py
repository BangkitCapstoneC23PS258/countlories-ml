from flask import Flask, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from food_class import class_names
import numpy as np

app = Flask(__name__)

our_model = load_model('model.h5', compile = False)

def food_predict(model, images):
  img = image.load_img("static/"+images, target_size=(150, 150))
  img = image.img_to_array(img)                    
  img = np.expand_dims(img, axis=0)/255                                   
  pred = model.predict(img)
  index = np.argmax(pred)
  pred = list(class_names.keys())[list(class_names.values()).index(index)]
  return pred

@app.route("/get_predict", methods = ['POST'])
def get_predict():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename	
		img.save(img_path)
		result = food_predict(our_model, img.filename)

	response = {
			'payload': {
				'food_name': str(result)
			},
			'status' : 200
		}
	
	return response

if __name__ =='__main__':
	app.debug = True
	app.run(debug = True)