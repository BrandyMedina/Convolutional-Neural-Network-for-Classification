from tensorflow import keras
import cv2 as cv

lista = ['anger','contempt','disgust','fear','happy','neutral','sad','surprise']

model = keras.models.load_model('CNN_Expressions_4.h5')

img = cv.imread("imagenes/surprise.png")

face = cv.resize(img, (50,50))
conv = cv.cvtColor(face, cv.COLOR_BGR2RGB)     
patron = conv.reshape(1,50,50,3)        

pred = model.predict(patron);
max_value = max(pred[0])
clase = list(pred[0]).index(max_value)
print(lista[clase])