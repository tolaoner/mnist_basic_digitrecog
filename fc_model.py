import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
from tensorflow.keras.datasets import mnist
no_classes = 10
no_features = 784
learning_rate = 0.1
training_steps = 2000
batch_size = 256
display_step = 100
hidden_1 = 128
hidden_2 = 256
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test,np.float32)
x_train, x_test = x_train.reshape([-1,no_features]), x_test.reshape([-1, no_features])
x_train, x_test = x_train/255., x_test/255
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
#create TF model
class NeuralNet(Model):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = layers.Dense(hidden_1, activation=tf.nn.relu)
        self.fc2 = layers.Dense(hidden_2, activation=tf.nn.relu)
        self.out = layers.Dense(num_classes)
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        if not is_training:
            #tf cross entropy expect logits without softmax, so apply softmax only when not training
            x = tf.nn.softmax(x)
        return x
#build neural network model
neural_net = NeuralNet()
#cross_entropy loss
#note that this will apply 'softmax' to the logits.
def cross_entropy_loss(x, y):
    #convert labels to int 64 for tf cross entropy function
    y = tf.cast(y, tf.int64)
    #apply softmax to logits and compute cross entropy
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    #average loss across the batch
    return tf.reduce_mean(loss)
#accuracy metric
def accuracy(y_pred, y_true):
    right_pred = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf. reduce_mean(tf.cast(correct_prediction, tf.float32), axis = -1)
#stochastic gradient descent optimizer
gradient_optimizer = tf_optimizers.SGD(learning_rate)
def run_optimization(x, y):
    #wrap computation inside a GradientTape for automatic differentiation
    with tf.GradientTape() as g:
        #forward pass
        pred = neural_net(x, is_training = True)
        #compute loss
        loss = cross_entropy_loss(pred, y)
    #variables to update
    trainable_variables = neural_net.trainable_variables
    #compute gradients
    gradients = g.gradient(loss, trainable_variables)
    #update W and b following gradients
    optimizer.apply_gradients(zip(gradients, trainable_variables))
#run training for the given number of steps
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    #run optimization to update w and b values
    run_optimization(batch_x, batch_y)
    if step % display_step == 0:
        pred = neural_net (batch_x, is_training = True)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print('step: %i, loss: %f, accuracy: %f' % (step, loss, acc))
#test model on validation set
pred = neural_net(x_test, is_training = False)
print('Test Accuracy: %f' % accuracy(pred, y_test))
#visualize predictions
import matplotlib.pyplot as plt
#predict 10 images from validation set
no_images = 10
test_images = x_test[:no_images]
predictions = neural_net(test_images)
#display image and model prediction
for i in range(no_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print('Model prediction = %i' % np.argmax(predictions.numpy()[i]))
