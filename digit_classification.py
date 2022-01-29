import struct
import numpy as numpy
import matplotlib as plt
import os


def load_data():
    with open("train-labels-idx1-ubyte", "rb") as labels:
        magic, n = struct.unpack(">II", labels.read(8))
        train_labels = np.fromfile(labels, dtype=np.uint8)


# import gzip
# f = gzip.open('train-images-idx3-ubyte.gz','r')

# image_size = 28
# num_images = 5

# import numpy as np
# f.read(16)
# buf = f.read(image_size * image_size * num_images)
# data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
# data = data.reshape(num_images, image_size, image_size, 1)

# import matplotlib.pyplot as plt
# image = np.asarray(data[2]).squeeze()
# plt.imshow(image)
# plt.show()