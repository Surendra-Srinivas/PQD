import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

num_classes = 9
classes = ['Normal', 'Sag', 'Swell', 'Interruption', 'Notch', 'Spike', 'Flicker', 'Oscillatory Transients', 'Harmonics']

Sine_data = scipy.io.loadmat('Sine_data.mat')
X_Sine_train = Sine_data['sine_train']
X_Sine_val = Sine_data['sine_val']
X_Sine_test = Sine_data['sine_test']

Sine_label_data = scipy.io.loadmat('Sine_label_data.mat')
y_Sine_train = Sine_label_data['sine_label_train']
y_Sine_val = Sine_label_data['sine_label_val']
y_Sine_test = Sine_label_data['sine_label_test']

Sag_data = scipy.io.loadmat('Sag_data.mat')
X_Sag_train = Sag_data['sag_train']
X_Sag_val = Sag_data['sag_val']
X_Sag_test = Sag_data['sag_test']

Sag_label_data = scipy.io.loadmat('Sag_label_data.mat')
y_Sag_train = Sag_label_data['sag_label_train']
y_Sag_val = Sag_label_data['sag_label_val']
y_Sag_test = Sag_label_data['sag_label_test']

Swell_data = scipy.io.loadmat('Swell_data.mat')
X_Swell_train = Swell_data['swell_train']
X_Swell_val = Swell_data['swell_val']
X_Swell_test = Swell_data['swell_test']

Swell_label_data = scipy.io.loadmat('Swell_label_data.mat')
y_Swell_train = Swell_label_data['swell_label_train']
y_Swell_val = Swell_label_data['swell_label_val']
y_Swell_test = Swell_label_data['swell_label_test']

Interruption_data = scipy.io.loadmat('Interruption_data.mat')
X_Interruption_train = Interruption_data['interruption_train']
X_Interruption_val = Interruption_data['interruption_val']
X_Interruption_test = Interruption_data['interruption_test']

Interruption_label_data = scipy.io.loadmat('Interruption_label_data.mat')
y_Interruption_train = Interruption_label_data['interruption_label_train']
y_Interruption_val = Interruption_label_data['interruption_label_val']
y_Interruption_test = Interruption_label_data['interruption_label_test']

Notch_data = scipy.io.loadmat('Notch_data.mat')
X_Notch_train = Notch_data['notch_train']
X_Notch_val = Notch_data['notch_val']
X_Notch_test = Notch_data['notch_test']

Notch_label_data = scipy.io.loadmat('Notch_label_data.mat')
y_Notch_train = Notch_label_data['notch_label_train']
y_Notch_val = Notch_label_data['notch_label_val']
y_Notch_test = Notch_label_data['notch_label_test']

Spike_data = scipy.io.loadmat('Spike_data.mat')
X_Spike_train = Spike_data['spike_train']
X_Spike_val = Spike_data['spike_val']
X_Spike_test = Spike_data['spike_test']

Spike_label_data = scipy.io.loadmat('Spike_label_data.mat')
y_Spike_train = Spike_label_data['spike_label_train']
y_Spike_val = Spike_label_data['spike_label_val']
y_Spike_test = Spike_label_data['spike_label_test']

Flicker_data = scipy.io.loadmat('Flicker_data.mat')
X_Flicker_train = Sine_data['flicker_train']
X_Flicker_val = Sine_data['flicker_val']
X_Flicker_test = Sine_data['flicker_test']

Flicker_label_data = scipy.io.loadmat('Flicker_label_data.mat')
y_Flicker_train = Flicker_label_data['flicker_label_train']
y_Flicker_val = Flicker_label_data['flicker_label_val']
y_Flicker_test = Flicker_label_data['flicker_label_test']

Oscillatory_Transients_data = scipy.io.loadmat('Oscillatory_Transients_data.mat')
X_Oscillatory_Transients_train = Oscillatory_Transients_data['oscillatory_transients_train']
X_Oscillatory_Transients_val = Oscillatory_Transients_data['oscillatory_transients_val']
X_Oscillatory_Transients_test = Oscillatory_Transients_data['oscillatory_transients_test']

Oscillatory_Transients_label_data = scipy.io.loadmat('Oscillatory_Transients_label_data.mat')
y_Oscillatory_Transients_train = Oscillatory_Transients_label_data['Oscillatory_Transients_label_train']
y_Oscillatory_Transients_val = Oscillatory_Transients_label_data['Oscillatory_Transients_label_val']
y_Oscillatory_Transients_test = Oscillatory_Transients_label_data['Oscillatory_Transients_label_test']

Harmonics_data = scipy.io.loadmat('Harmonics_data.mat')
X_Harmonics_train = Harmonics_data['harmonics_train']
X_Harmonics_val = Harmonics_data['harmonics_val']
X_Harmonics_test = Harmonics_data['harmonics_test']

Harmonics_label_data = scipy.io.loadmat('Harmonics_label_data.mat')
y_Harmonics_train = Harmonics_label_data['harmonics_label_train']
y_Harmonics_val = Harmonics_label_data['harmonics_label_val']
y_Harmonics_test = Harmonics_label_data['harmonics_label_test']


# Convert lists of arrays to numpy arrays
X_Sine_train = np.array([np.vstack(chunk) for chunk in X_Sine_train])
X_Sine_val = np.array([np.vstack(chunk) for chunk in X_Sine_val])
X_Sine_test = np.array([np.vstack(chunk) for chunk in X_Sine_test])

y_Sine_train = y_Sine_train.reshape(np.shape(X_Sine_train)[0])
y_Sine_test = y_Sine_test.reshape(np.shape(X_Sine_test)[0])
y_Sine_val = y_Sine_val.reshape(np.shape(X_Sine_val)[0])

# Convert lists of arrays to numpy arrays
X_Sag_train = np.array([np.vstack(chunk) for chunk in X_Sag_train])
X_Sag_val = np.array([np.vstack(chunk) for chunk in X_Sag_val])
X_Sag_test = np.array([np.vstack(chunk) for chunk in X_Sag_test])

y_Sag_train = y_Sag_train.reshape(np.shape(X_Sag_train)[0])
y_Sag_test = y_Sag_test.reshape(np.shape(X_Sag_test)[0])
y_Sag_val = y_Sag_val.reshape(np.shape(X_Sag_val)[0])

# Convert lists of arrays to numpy arrays
X_Swell_train = np.array([np.vstack(chunk) for chunk in X_Swell_train])
X_Swell_val = np.array([np.vstack(chunk) for chunk in X_Swell_val])
X_Swell_test = np.array([np.vstack(chunk) for chunk in X_Swell_test])

y_Swell_train = y_Swell_train.reshape(np.shape(X_Swell_train)[0])
y_Swell_test = y_Swell_test.reshape(np.shape(X_Swell_test)[0])
y_Swell_val = y_Swell_val.reshape(np.shape(X_Swell_val)[0])

# Convert lists of arrays to numpy arrays
X_Interruption_train = np.array([np.vstack(chunk) for chunk in X_Interruption_train])
X_Interruption_val = np.array([np.vstack(chunk) for chunk in X_Interruption_val])
X_Interruption_test = np.array([np.vstack(chunk) for chunk in X_Interruption_test])

y_Interruption_train = y_Interruption_train.reshape(np.shape(X_Interruption_train)[0])
y_Interruption_test = y_Interruption_test.reshape(np.shape(X_Interruption_test)[0])
y_Interruption_val = y_Interruption_val.reshape(np.shape(X_Interruption_val)[0])

# Convert lists of arrays to numpy arrays
X_Notch_train = np.array([np.vstack(chunk) for chunk in X_Notch_train])
X_Notch_val = np.array([np.vstack(chunk) for chunk in X_Notch_val])
X_Notch_test = np.array([np.vstack(chunk) for chunk in X_Notch_test])

y_Notch_train = y_Notch_train.reshape(np.shape(X_Notch_train)[0])
y_Notch_test = y_Notch_test.reshape(np.shape(X_Notch_test)[0])
y_Notch_val = y_Notch_val.reshape(np.shape(X_Notch_val)[0])

# Convert lists of arrays to numpy arrays
X_Spike_train = np.array([np.vstack(chunk) for chunk in X_Spike_train])
X_Spike_val = np.array([np.vstack(chunk) for chunk in X_Spike_val])
X_Spike_test = np.array([np.vstack(chunk) for chunk in X_Spike_test])

y_Spike_train = y_Spike_train.reshape(np.shape(X_Spike_train)[0])
y_Spike_test = y_Spike_test.reshape(np.shape(X_Spike_test)[0])
y_Spike_val = y_Spike_val.reshape(np.shape(X_Spike_val)[0])

# Convert lists of arrays to numpy arrays
X_Flicker_train = np.array([np.vstack(chunk) for chunk in X_Flicker_train])
X_Flicker_val = np.array([np.vstack(chunk) for chunk in X_Flicker_val])
X_Flicker_test = np.array([np.vstack(chunk) for chunk in X_Flicker_test])

y_Flicker_train = y_Flicker_train.reshape(np.shape(X_Flicker_train)[0])
y_Flicker_test = y_Flicker_test.reshape(np.shape(X_Flicker_test)[0])
y_Flicker_val = y_Flicker_val.reshape(np.shape(X_Flicker_val)[0])

# Convert lists of arrays to numpy arrays
X_Oscillatory_Transients_train = np.array([np.vstack(chunk) for chunk in X_Oscillatory_Transients_train])
X_Oscillatory_Transients_val = np.array([np.vstack(chunk) for chunk in X_Oscillatory_Transients_val])
X_Oscillatory_Transients_test = np.array([np.vstack(chunk) for chunk in X_Oscillatory_Transients_test])

y_Oscillatory_Transients_train = y_Oscillatory_Transients_train.reshape(np.shape(X_Oscillatory_Transients_train)[0])
y_Oscillatory_Transients_test = y_Oscillatory_Transients_test.reshape(np.shape(X_Oscillatory_Transients_test)[0])
y_Oscillatory_Transients_val = y_Oscillatory_Transients_val.reshape(np.shape(X_Oscillatory_Transients_val)[0])

# Convert lists of arrays to numpy arrays
X_Harmonics_train = np.array([np.vstack(chunk) for chunk in X_Harmonics_train])
X_Harmonics_val = np.array([np.vstack(chunk) for chunk in X_Harmonics_val])
X_Harmonics_test = np.array([np.vstack(chunk) for chunk in X_Harmonics_test])

y_Harmonics_train = y_Harmonics_train.reshape(np.shape(X_Harmonics_train)[0])
y_Harmonics_test = y_Harmonics_test.reshape(np.shape(X_Harmonics_test)[0])
y_Harmonics_val = y_Harmonics_val.reshape(np.shape(X_Harmonics_val)[0])


X_train = np.concatenate([X_Sine_train, X_Sag_train, X_Swell_train, X_Interruption_train, X_Notch_train, X_Spike_train, X_Flicker_train, X_Oscillatory_Transients_train, X_Harmonics_train], axis = 0)
y_train = np.concatenate([y_Sine_train, y_Sag_train, y_Swell_train, y_Interruption_train, y_Notch_train, y_Spike_train, y_Flicker_train, y_Oscillatory_Transients_train, y_Harmonics_train], axis = 0)

X_test = np.concatenate([X_Sine_test, X_Sag_test, X_Swell_test, X_Interruption_test, X_Notch_test, X_Spike_test, X_Flicker_test, X_Oscillatory_Transients_test, X_Harmonics_test], axis = 0)
y_test = np.concatenate([y_Sine_test, y_Sag_test, y_Swell_test, y_Interruption_test, y_Notch_test, y_Spike_test, y_Flicker_test, y_Oscillatory_Transients_test, y_Harmonics_test], axis = 0)

X_val = np.concatenate([X_Sine_val, X_Sag_val, X_Swell_val, X_Interruption_val, X_Notch_val, X_Spike_val, X_Flicker_val, X_Oscillatory_Transients_val, X_Harmonics_val], axis = 0)
y_val = np.concatenate([y_Sine_val, y_Sag_val, y_Swell_val, y_Interruption_val, y_Notch_val, y_Spike_val, y_Flicker_val, y_Oscillatory_Transients_val, y_Harmonics_val], axis = 0)

y_train = to_categorical(y_train, num_classes = num_classes)
y_test = to_categorical(y_test, num_classes = num_classes)
y_val = to_categorical(y_val, num_classes = num_classes)

print('The shape of X_Sine_train is : ',np.shape(X_Sine_train))
print('The shape of X_Sag_train is : ',np.shape(X_Sag_train))
print('The shape of X_Swell_train is : ',np.shape(X_Swell_train))
print('The shape of X_Interruption_train is : ',np.shape(X_Interruption_train))
print('The shape of X_Notch_train is : ',np.shape(X_Notch_train))
print('The shape of X_Spike_train is : ',np.shape(X_Spike_train))
print('The shape of X_Flicker_train is : ',np.shape(X_Flicker_train))
print('The shape of X_Oscillatory_Transients_train is : ',np.shape(X_Oscillatory_Transients_train))
print('The shape of X_Harmonics_train is : ',np.shape(X_Harmonics_train))
print("\n")
print('The shape of y_Sine_train is : ',np.shape(y_Sine_train))
print('The shape of y_Sag_train is : ',np.shape(y_Sag_train))
print('The shape of y_Swell_train is : ',np.shape(y_Swell_train))
print('The shape of y_Interruption_train is : ',np.shape(y_Interruption_train))
print('The shape of y_Notch_train is : ',np.shape(y_Notch_train))
print('The shape of y_Spike_train is : ',np.shape(y_Spike_train))
print('The shape of y_Flicker_train is : ',np.shape(y_Flicker_train))
print('The shape of y_Oscillatory_Transients_train is : ',np.shape(y_Oscillatory_Transients_train))
print('The shape of y_Harmonics_train is : ',np.shape(y_Harmonics_train))
print("\n")
print("The shape of X_train is : ", np.shape(X_train))
print("The shape of y_train is : ", np.shape(y_train))
print("\n")
print("The shape of y_test is : ", np.shape(y_test))

def build_model(input_shape, classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.1, l2=0.01)),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.1, l2=0.01)),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.1, l2=0.01)),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.1, l2=0.01)),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.1, l2=0.01)),
        Dense(classes, activation='softmax')  # Output layer with sigmoid activation for binary classification
    ])
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


model  = build_model(input_shape = (640, 1), classes = num_classes)
history = model.fit(X_train, y_train, epochs =10, batch_size = 64, validation_data = (X_val, y_val))

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

