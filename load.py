from keras.datasets import mnist # on running first time , downlooads the dataset
#import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#plt.subplot(221)
#plt.imshow(X_train[5], cmap=plt.get_cmap('gray'))
import numpy as np
import cv2
img=X_train[5]
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('test.jpg',img)
'''
plt.axis('off')
plt.savefig("test.png", bbox_inches='tight')
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.show()
'''