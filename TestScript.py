import sys
sys.path.append("deepsurv")
from matplotlib import pyplot as plt
import numpy
from keras.optimizers import SGD, RMSprop
import seaborn as sns
import pandas

import deepsurv
import deepsurv_keras


def generate_data(treatment_group = False):
    numpy.random.seed(123)
    sd = deepsurv.datasets.SimulatedData(5, num_features=9, treatment_group=treatment_group)
    train_data = sd.generate_data(5000)
    valid_data = sd.generate_data(2000)
    test_data = sd.generate_data(2000)
    return train_data, valid_data, test_data


train, valid, test = generate_data(treatment_group=True)

model = deepsurv_keras.build_model()

sgd = SGD(lr=1e-5, decay=0.01, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=1e-5, rho=0.9, epsilon=1e-8)
model.compile(loss=deepsurv_keras.negative_log_likelihood(train['e']), optimizer=sgd)

print('Training...')
history = model.fit(train['x'], train['y'], batch_size=324, epochs=1000, shuffle=False, verbose=False)  # Shuffle False --> Important!!

plt.plot(history.history['loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
