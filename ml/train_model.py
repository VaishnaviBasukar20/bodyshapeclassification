import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import tensorflow as tf

# preprocessing function
def apply_canny(image):
    image = cv2.resize(image,(224,224))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)
    edges = cv2.Canny(blurred_image, 50, 150)
    edges_3_channel = cv2.merge([edges, edges, edges])
    return edges_3_channel / 255.0

# Define body shapes
shapes = ["apple", "pear", "rectangle", "inverted triangle", "hourglass"]

# Dictionary to map shapes to numerical labels
labels_annot = {shapes[i]: i for i in range(len(shapes))}

# Load data from CSV
df = pd.read_csv("utils\data\data.csv")

images = []
labels = []

# Sample and process images for each shape
for shape in shapes:
    sample = df.loc[df["shape"] == shape].sample(720)  # Sample 720 images per shape
    sample.reset_index(inplace=True)
    for i in sample.index:
        img = cv2.imread(sample.loc[i, 'img'])
        img = apply_canny(img)  # Apply preprocessing (e.g., Canny edge detection)
        cv2.imwrite(f"ml\\training_data\\{shape}{i}.png",img)
        images.append(img)
        labels.append(labels_annot[shape])

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Shuffle images and labels together
images, labels = shuffle(images, labels, random_state=42)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# # Data augmentation
# train_datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# val_datagen = ImageDataGenerator()

# train_generator = train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
# val_generator = val_datagen.flow(X_val, y_val, batch_size=32, shuffle=True)

# Load the ResNet50 model, excluding the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
predictions = Dense(5, activation='softmax')(x)  # 5 classes for body shapes

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(r'ml\model\best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model with callbacks
history = model.fit(X_train, y_train,
                    epochs=50,
                    validation_data=(X_val, y_val),
                    callbacks=[reduce_lr, early_stopping, model_checkpoint])

# Unfreeze some layers of the base model and fine-tune the model
for layer in base_model.layers[-10:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(X_train, y_train,
                         epochs=20,
                         validation_data=(X_val, y_val),
                         callbacks=[reduce_lr, early_stopping, model_checkpoint])

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

# Save the trained model
model.save(r'ml\model\resnet50_body_shape_classification_model_2.keras')