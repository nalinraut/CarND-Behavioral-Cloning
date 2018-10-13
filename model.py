
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model



def save_summary(summary):
    with open('modelsummary.txt', 'w+ ') as f:
        print(s)

def model(learning_rate = 0.001):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape= (160, 320, 3)))
    
    model.add(Cropping2D(cropping = ((65, 25),(0,0))))
    
    model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation = 'relu'))
    model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation = 'relu'))
    model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation = 'relu'))
    model.add(Convolution2D(64, 3, 3, subsample = (1,1), activation = 'relu'))
    model.add(Convolution2D(64, 3, 3, subsample = (1,1), activation = 'relu'))
    
    model.add(Flatten())
    
    model.add(Dense(1164))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    
    model.add(Dense(100))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.summary()
    
    model.compile(loss = 'mse', optimizer = Adam(learning_rate))
    
    model.save('model.h5')
    print('Model saved successfully')
    return model
    
    

