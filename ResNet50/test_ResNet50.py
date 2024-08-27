import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

model = load_model('weights/ResNet50_three_class_model.h5')
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    'Dataset/test2',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
loss, accuracy = model.evaluate(test_generator)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
for i, (true_label, predicted_label) in enumerate(zip(true_classes, predicted_classes)):
    print(f'Sample {i+1}: True Class: {true_label}, Predicted Class: {predicted_label}')


