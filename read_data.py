import cv2
import os
import numpy


def read():
    # this function read the images from the folder and its names
    # thin return a list of names and the images
    images = []
    names = []
    for path, subdir, files in os.walk("C:\\Users\\Ali Mohamed\\Desktop\\post"):
        for filename in files:
            names.append(filename)
    for name in names:
        images.append([cv2.imread('C:\\Users\\Ali Mohamed\\Desktop\\post\\' + name, cv2.IMREAD_COLOR),
                       name.split('.')[0]])
    return images, names;


def generate_id(names):
    unique_names = {name.split('.')[0] for name in names}
    unique_names_length = len(unique_names)
    log_length = numpy.log2(unique_names_length);
    log_length = int(numpy.ceil(log_length))
    dec = {}
    i =0
    for name in unique_names:
        temp = str(bin(i))
        i = i + 1
        dec[name]=temp.replace('0b','')
    for name in unique_names:
        zeros = "0"
        dec[name] = (zeros * (log_length - len(dec[name]))) + dec[name]
        dec[name] = numpy.array([int(c) for c in dec[name]])
    return unique_names_length,dec


def generate_decoder(encoder):
    decoder={}
    for key  in encoder:
        temp = str(encoder[key])
        decoder[temp]= key
    return decoder
imag,name = read()
_,dc= generate_id(name)
print(dc)
print(generate_decoder(dc))