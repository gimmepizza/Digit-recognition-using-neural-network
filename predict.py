from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
def prep_image(img):
	#assumption we are getting a 28*28 img
	
	img = img.reshape(1,784).astype('float32')
	img=img/255
	return img


#load the model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
#compile the model
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


img=mpimg.imread('test.jpg')
img=prep_image(img)
answer=loaded_model.predict(img, batch_size=1, verbose=0, steps=None)

print (answer)
print(np.argmax(answer))

plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()
