
# coding: utf-8

# In[ ]:


from model import *
from datamod import *
import matplotlib.pyplot as plt

SAMPLES_PER_EPOCH = 30000
EPOCHS = 8
VALID_SAMPLES = 6400
LEARNING_RATE = 0.001

def train(train_data_generator, validate_data_generator):
    model_ = model(learning_rate = LEARNING_RATE)
    train_history = model_.fit_generator(train_data_generator, 
                                samples_per_epoch= SAMPLES_PER_EPOCH, 
                                verbose=1, nb_epoch=EPOCHS,
                                validation_data = validate_data_generator, 
                                nb_val_samples=VALID_SAMPLES)
    model_.save('model.h5')
    return train_history

def plot_history(train_history):
    plt.plot(train_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.show()
    
    
if __name__ == '__main__':
    
    train_history = train(get_data(data_type='train'),get_data(data_type='valid'))
    plot_history(train_history)

