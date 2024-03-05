import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

num_classes = 6
classes = ['Normal', 'Sag', 'Swell', 'Interruption', 'Notch', 'Spike']

Sine_data = scipy.io.loadmat('Sine_data.mat')
X_Sine_train = Sine_data['sine_train']
X_Sine_val = Sine_data['sine_val']
X_Sine_test = Sine_data['sine_test']

Sine_label_data = scipy.io.loadmat('Sine_label_data.mat')
y_Sine_label_train = Sine_label_data['sine_label_train']
y_Sine_label_val = Sine_label_data['sine_label_val']
y_Sine_label_test = Sine_label_data['sine_label_test']

Sag_data = scipy.io.loadmat('Sag_data.mat')
X_Sag_train = Sag_data['sag_train']
X_Sag_val = Sag_data['sag_val']
X_Sag_test = Sag_data['sag_test']

Sag_label_data = scipy.io.loadmat('Sag_label_data.mat')
y_Sag_label_train = Sag_label_data['sag_label_train']
y_Sag_label_val = Sag_label_data['sag_label_val']
y_Sag_label_test = Sag_label_data['sag_label_test']

Swell_data = scipy.io.loadmat('Swell_data.mat')
X_Swell_train = Swell_data['swell_train']
X_Swell_val = Swell_data['swell_val']
X_Swell_test = Swell_data['swell_test']

Swell_label_data = scipy.io.loadmat('Swell_label_data.mat')
y_Swell_label_train = Swell_label_data['swell_label_train']
y_Swell_label_val = Swell_label_data['swell_label_val']
y_Swell_label_test = Swell_label_data['swell_label_test']

Interruption_data = scipy.io.loadmat('Interruption_data.mat')
X_Interruption_train = Interruption_data['interruption_train']
X_Interruption_val = Interruption_data['interruption_val']
X_Interruption_test = Interruption_data['interruption_test']

Interruption_label_data = scipy.io.loadmat('Interruption_label_data.mat')
y_Interruption_label_train = Interruption_label_data['interruption_label_train']
y_Interruption_label_val = Interruption_label_data['interruption_label_val']
y_Interruption_label_test = Interruption_label_data['interruption_label_test']

Notch_data = scipy.io.loadmat('Notch_data.mat')
X_Notch_train = Notch_data['notch_train']
X_Notch_val = Notch_data['notch_val']
X_Notch_test = Notch_data['notch_test']

Notch_label_data = scipy.io.loadmat('Notch_label_data.mat')
y_Notch_label_train = Notch_label_data['notch_label_train']
y_Notch_label_val = Notch_label_data['notch_label_val']
y_Notch_label_test = Notch_label_data['notch_label_test']

Spike_data = scipy.io.loadmat('Spike_data.mat')
X_Spike_train = Spike_data['spike_train']
X_Spike_val = Spike_data['spike_val']
X_Spike_test = Spike_data['spike_test']

Spike_label_data = scipy.io.loadmat('Spike_label_data.mat')
y_Spike_label_train = Spike_label_data['spike_label_train']
y_Spike_label_val = Spike_label_data['spike_label_val']
y_Spike_label_test = Spike_label_data['spike_label_test']

# Convert lists of arrays to numpy arrays
X_Sine_train = np.array([np.vstack(chunk) for chunk in X_Sine_train])
X_Sine_val = np.array([np.vstack(chunk) for chunk in X_Sine_val])
X_Sine_test = np.array([np.vstack(chunk) for chunk in X_Sine_test])

y_Sine_label_train = y_Sine_label_train.reshape(np.shape(X_Sine_train)[0])
y_Sine_label_test = y_Sine_label_test.reshape(np.shape(X_Sine_test)[0])
y_Sine_label_val = y_Sine_label_val.reshape(np.shape(X_Sine_val)[0])

# Convert lists of arrays to numpy arrays
X_Sag_train = np.array([np.vstack(chunk) for chunk in X_Sag_train])
X_Sag_val = np.array([np.vstack(chunk) for chunk in X_Sag_val])
X_Sag_test = np.array([np.vstack(chunk) for chunk in X_Sag_test])

y_Sag_label_train = y_Sag_label_train.reshape(np.shape(X_Sag_train)[0])
y_Sag_label_test = y_Sag_label_test.reshape(np.shape(X_Sag_test)[0])
y_Sag_label_val = y_Sag_label_val.reshape(np.shape(X_Sag_val)[0])

# Convert lists of arrays to numpy arrays
X_Swell_train = np.array([np.vstack(chunk) for chunk in X_Swell_train])
X_Swell_val = np.array([np.vstack(chunk) for chunk in X_Swell_val])
X_Swell_test = np.array([np.vstack(chunk) for chunk in X_Swell_test])

y_Swell_label_train = y_Swell_label_train.reshape(np.shape(X_Swell_train)[0])
y_Swell_label_test = y_Swell_label_test.reshape(np.shape(X_Swell_test)[0])
y_Swell_label_val = y_Swell_label_val.reshape(np.shape(X_Swell_val)[0])

# Convert lists of arrays to numpy arrays
X_Interruption_train = np.array([np.vstack(chunk) for chunk in X_Interruption_train])
X_Interruption_val = np.array([np.vstack(chunk) for chunk in X_Interruption_val])
X_Interruption_test = np.array([np.vstack(chunk) for chunk in X_Interruption_test])

y_Interruption_label_train = y_Interruption_label_train.reshape(np.shape(X_Interruption_train)[0])
y_Interruption_label_test = y_Interruption_label_test.reshape(np.shape(X_Interruption_test)[0])
y_Interruption_label_val = y_Interruption_label_val.reshape(np.shape(X_Interruption_val)[0])

# Convert lists of arrays to numpy arrays
X_Notch_train = np.array([np.vstack(chunk) for chunk in X_Notch_train])
X_Notch_val = np.array([np.vstack(chunk) for chunk in X_Notch_val])
X_Notch_test = np.array([np.vstack(chunk) for chunk in X_Notch_test])

y_Notch_label_train = y_Notch_label_train.reshape(np.shape(X_Notch_train)[0])
y_Notch_label_test = y_Notch_label_test.reshape(np.shape(X_Notch_test)[0])
y_Notch_label_val = y_Notch_label_val.reshape(np.shape(X_Notch_val)[0])

# Convert lists of arrays to numpy arrays
X_Spike_train = np.array([np.vstack(chunk) for chunk in X_Spike_train])
X_Spike_val = np.array([np.vstack(chunk) for chunk in X_Spike_val])
X_Spike_test = np.array([np.vstack(chunk) for chunk in X_Spike_test])

y_Spike_label_train = y_Notch_label_train.reshape(np.shape(X_Spike_train)[0])
y_Spike_label_test = y_Notch_label_test.reshape(np.shape(X_Spike_test)[0])
y_Spike_label_val = y_Notch_label_val.reshape(np.shape(X_Spike_val)[0])

X_train = np.concatenate([X_Sine_train, X_Sag_train, X_Swell_train, X_Interruption_train, X_Notch_train, X_Spike_train], axis = 0)
y_train = np.concatenate([y_Sine_label_train, y_Sag_label_train, y_Swell_label_train, y_Interruption_label_train, y_Notch_label_train, y_Spike_label_train], axis = 0)

X_test = np.concatenate([X_Sine_test, X_Sag_test, X_Swell_test, X_Interruption_test, X_Notch_test, X_Spike_test], axis = 0)
y_test = np.concatenate([y_Sine_label_test, y_Sag_label_test, y_Swell_label_test, y_Interruption_label_test, y_Notch_label_test, y_Spike_label_test], axis = 0)

X_val = np.concatenate([X_Sine_val, X_Sag_val, X_Swell_val, X_Interruption_val, X_Notch_val, X_Spike_val], axis = 0)
y_val = np.concatenate([y_Sine_label_val, y_Sag_label_val, y_Swell_label_val, y_Interruption_label_val, y_Notch_label_val, y_Spike_label_val], axis = 0)

y_train = to_categorical(y_train, num_classes = num_classes)
y_test = to_categorical(y_test, num_classes = num_classes)
y_val = to_categorical(y_val, num_classes = num_classes)

print('The shape of X_Sine_train is : ',np.shape(X_Sine_train))
print('The shape of X_Sag_train is : ',np.shape(X_Sag_train))
print('The shape of X_Swell_train is : ',np.shape(X_Swell_train))
print('The shape of X_Interruption_train is : ',np.shape(X_Interruption_train))
print('The shape of X_Notch_train is : ',np.shape(X_Notch_train))
print('The shape of X_Spike_train is : ',np.shape(X_Spike_train))

print('The shape of y_Sine_label_train is : ',np.shape(y_Sine_label_train))
print('The shape of y_Sag_label_train is : ',np.shape(y_Sag_label_train))
print('The shape of y_Swell_label_train is : ',np.shape(y_Swell_label_train))
print('The shape of y_Interruption_label_train is : ',np.shape(y_Interruption_label_train))
print('The shape of y_Notch_label_train is : ',np.shape(y_Notch_label_train))
print('The shape of y_Spike_label_train is : ',np.shape(y_Spike_label_train))

print("The shape of X_train is : ", np.shape(X_train))
print("The shape of y_train is : ", np.shape(y_train))

def build_model(input_shape, classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(classes, activation='softmax')  # Output layer with sigmoid activation for binary classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


model  = build_model(input_shape = (640, 1), classes = num_classes)
history = model.fit(X_train, y_train, epochs =2, batch_size = 32, validation_data = (X_val, y_val))

Sample_index = 420
single_sample = X_test[Sample_index]

ts = 1/3200;
t = np.arange(0,0.2, ts, dtype = float)
plt.plot(t, single_sample)
print(np.shape(single_sample))
single_sample = single_sample.reshape(1, single_sample.shape[0], 1)
print(np.shape(single_sample))
print('Time shape',np.shape(t))

"""loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)"""

predict_x=model.predict(single_sample)
classes_x=np.argmax(predict_x, axis=1)[0]
print('The disturbance is : ',classes[classes_x])

"""
print(history.history.keys())
plt.plot(history.history['val_accuracy'])
print(np.zeros(5))
print(np.shape(np.zeros(5)))
arr = np.full((5,),0, dtype = float)
print(arr)
print(np.shape(arr))

from tensorflow.keras.utils import to_categorical

classes = 3
y1 = np.zeros(5)
print(y1)
y2 = np.ones(5)
print(y2)
y3 = np.full((5,), 2)
print(y3)

y = np.concatenate([y1, y2, y3], axis = 0)
print(y)
y = to_categorical(y, num_classes = classes)
print(y)

Sine_label_data = scipy.io.loadmat('Sine_label_data.mat')
print(Sine_label_data.keys())

Sine_data = scipy.io.loadmat('Sine_data.mat')
print(Sine_data.keys())

"""

