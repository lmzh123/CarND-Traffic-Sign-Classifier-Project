# **Self-Driving Car Engineer Nanodegree** 

## Luis Miguel Zapata

---

**Traffic Sign Recognition Classifier**

This projects aims to develop a Deep Neural Network classifier for traffic signs from the German Traffic Sign Dataset.  

[image1]: ./screenshots/hist_training.png "Histogram Training"
[image2]: ./screenshots/hist_valid.png "Histogram Validation"
[image3]: ./screenshots/hist_test.png "Histogram Testing"
[image4]: ./screenshots/example_1.png "Example 1"
[image5]: ./screenshots/example_2.png "Example 2"
[image6]: ./screenshots/example_3.png "Example 3"
[image7]: ./screenshots/normal.jpg "Original"
[image8]: ./screenshots/gray.jpg "Gray"
[image9]: ./screenshots/equalized.jpg "Equalized"
[image10]: ./screenshots/normalized.png "Normalized"
[image11]: ./screenshots/test_1.png "Test 1"
[image12]: ./screenshots/test_2.png "Test 2"
[image13]: ./screenshots/test_3.png "Test 3"
[image14]: ./screenshots/test_4.png "Test 4"
[image15]: ./screenshots/test_5.png "Test 5"


### 1. Dataset.

The dataset consists of 3 different pickle files containing images and labels for the training, the validation and the testing.

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630

As it can be seen in the histogras shown below the amount of examples per class in every dataset show similar shapes and regardless from not being uniform distributions the amount of examples per class in the training set is proportional to the same class in validation and test datasets.

Training                   |  Validation               |  Testing
:-------------------------:|:-------------------------:|:-------------------------:
![][image1]                |  ![][image2]              |  ![][image3]

The images come in RGB format and are already contained in numpy arrays and the labels come as a list of integers.

* Image data shape = (32, 32, 3)
* Number of classes = 43

Every image has a corresponding label and these labels correspond to a category of traffic signal. These categories can be seen in the file `signnames.csv`.

Example 1                  |  Example 2                |  Example 3
:-------------------------:|:-------------------------:|:-------------------------:
![][image4]                |  ![][image5]              |  ![][image6]

### 2. Preprocessing
The pre-processing pipeline consists in three different steps. 

* Grayscale: The RGB channels disappear and only one channel corresponding to intensities remain.
```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
* Histogram equalzation: The contrast of the image is enhanced obtaining a more uniform histogram.
```
gray_equalized = cv2.equalizeHist(gray)
```

Original                   |  Grayscale                |  Histogram Equalization   
:-------------------------:|:-------------------------:|:-------------------------:
![][image7]                |  ![][image8]              |  ![][image9]              

* Normalization: The values of the intensity image no longer go from 0 to 255, but they range now from -1 to 1 in floating point format. Still using Matplotlib visualization the image looks identical to equalized one.

![][image10]  
```
norm_image = (gray_equalized - 128.0)/ 128.0
```
This procedure will enhance the results since the images itself and its information is more meaningful.

### 3. Model Architecture.
The model used here is inspired in the LeNet architecture which consists in the following:

* 2D Convolution: 6 Filters 5x5 + a Bias and stride of 1 by 1. Input = 32x32x1. Output = 28x28x6.
* ReLu activation. 
* Max Pooling: Filter 2x2. Stride 1 x 1. Input = 28x28x6. Output = 14x14x6.
* 2D Convolution: 16 Filters 5x5 + Bias. Stride 1 by 1. Input = 14x14x6. Output = 10x10x16.
* ReLu Activation.
* Max Pooling: Filter 2x2. Stride 1 x 1. Input = 10x10x16. Output = 5x5x16.
* Flatten: Input = 5x5x16. Output = 400.
* Multilayer perceptron: Input = 400. Output = 120.
* Multilayer perceptron: Input = 120. Output = 84.
* ReLu Activation.
* Multilayer perceptron (Output): Input = 84. Output = 43.

### 3. Model Training.
Placeholders for both the features and the labels are defined. The one-hot encoding is applied to the labels.

```
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
```

The learning rate is set to `0.001` experimentaly and the loss function to optimize using tehe Adam Optimization Algorithm will the the cross entropy between the one-hot encoded labels and the logits obtained by the model described before.

```
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

The accuracy of the model will be the mean of the correct predictions and this metric will be evaluated every epoch for every image of the batch. This procedure will be performed 100 times and for every time a set of 128 images will be used.

```
# A model to evaluate the accuracy is defined
EPOCHS = 100
BATCH_SIZE = 128

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```

The training consists then in optimize for every epoch the crossentropy as follows.

```
# The features and labels are shuffled every epoch
from sklearn.utils import shuffle

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
```

The resulting accurracy score for this configuration of hyper parameters and netwrok model is around `0.948` obtaining also a score of `0.932` for the testing set.

```
# Check performance in the test data set
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

### 4. New images.

A set of 5 new images is downloaded from the internet that do not belong to the dataset in order to test their performance. To feed these images trough the neural network the images have to be resized and the their labels are loaded into a python's list [31  4 26 17 22].

```
images.append(cv2.resize(np_image, (32, 32)))
```

Test 1                     |  Test 2                   |  Test 3   
:-------------------------:|:-------------------------:|:-------------------------:
![][image11]               |  ![][image12]             |  ![][image13]              

Test 4                     |  Test 5                   
:-------------------------:|:-------------------------:
![][image14]               |  ![][image15]             

Before making prediction on these images, these have to go trough the same pre-processing steps performed while the network was being trained and from here the predictions can be performed as follows:

```
# Preprocess the images
test_images_processed = preprocessing(test_images)

prediction = tf.argmax(logits, 1)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    results = sess.run(prediction, feed_dict={x:test_images_processed})
    print("Prediction = {}".format(results))   
```
