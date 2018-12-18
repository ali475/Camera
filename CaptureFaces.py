import cv2
import cnn
import numpy as np

from read_data import generate_decoder


def add_to_data_set():
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    Id = input('id:')
    sampleNum = 0
    while (True):
        ret, img = cam.read()
        faces = detector.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 0, 0), 2)
            sampleNum = sampleNum + 1
            image = img[y - 20:y + h + 20, x - 20:x + w + 20]
            image = cv2.resize(image, (48, 48), cv2.INTER_AREA)
            cv2.imwrite("C:\\Users\\Ali Mohamed\\Desktop\\post\\" + Id + '.' + str(sampleNum) + ".jpg",image)
            cv2.imshow('frame', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sampleNum == 250:
            break
    cam.release()
    cv2.destroyAllWindows()


def recognize():
    data_set, encoder, encoder_length = cnn.create_train_data();
    decoder = generate_decoder(encoder)
    X_train, Y_train = cnn.pars_data_to_x_y(data_set)
    cluster_size = np.ceil(np.log2(encoder_length))
    convent = cnn.rnn_conv_lstm_model(48, 48, 3, cluster_size)
    model = cnn.start_model(X_train, Y_train, convent)

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while (True):
        ret, img = cam.read()
        faces = detector.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 0, 0), 2)
            image = img[y - 20:y + h + 20, x - 20:x + w + 20]
            image = cv2.resize(image, (48, 48), cv2.IMREAD_COLOR)
            id = cnn.predict(model, [image],0.9)
            id = str(id[0]).replace(",","")
            print(decoder[id])
            #print(model.predict([image]))
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

recognize()
