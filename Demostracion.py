"""
Medina Cadena Brandy Berlin
Actividad 7- Red neuronal convoluvional
14/Mayo/2023
"""


from tensorflow import keras
import cv2 as cv

lista = ['anger','contempt','disgust','fear','happy','neutral','sad','suprise']

model = keras.models.load_model('CNN_Expressions_4.h5')
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascPath)


if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60,60))
    
    #por cada cara detectada pintar un cuadro
    for (x, y, w, h) in faces:
        c = 100
        face = frame[y-c:y+h+c,x-c:x+w+c,:]
        #cv.imshow('Face', face)
      
        
      
        if face.shape[0]>50 and face.shape[1]>50:
            face = cv.resize(face, (50,50))
            conv = cv.cvtColor(face, cv.COLOR_BGR2RGB)     
            patron = conv.reshape(1,50,50,3)        
       
            pred = model.predict(patron);
            max_value = max(pred[0])
            clase = list(pred[0]).index(max_value)
            print(lista[clase])
            
            frame = cv.putText(frame,lista[clase], (x,y), cv.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),3)
        
            cv.imshow('face', face)

        frame = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        
    # Display the resulting frame
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()