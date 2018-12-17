# Load pickled data
import pickle
import cv2
import matplotlib.pyplot as plt

def preprocessing(X):
    X_processed = []
    for i, img in enumerate(X):
        # equalize the histogram of the Y channel
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("preprocess/gray.jpg",gray)
        gray_equalized = cv2.equalizeHist(gray)
        cv2.imwrite("preprocess/equalized.jpg",gray)
        # Normalize image
        norm_image = (gray_equalized - 128.0)/ 128.0
        cv2.imwrite("preprocess/norm_image.jpg",norm_image)
        X_processed.append(norm_image)
    return X_processed
    # The set of images are stacked into a numpy array and its last dim expanded


# TODO: Fill this in based on where you saved the training and testing data

training_file = "dataset/train.p"
validation_file= "dataset/valid.p"
testing_file = "dataset/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']

X_train = preprocessing(X_train)

plt.imshow(X_train[-1], cmap='gray')
plt.show()