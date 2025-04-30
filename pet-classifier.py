import os
import random
from shutil import copyfile
import tensorflow as tf
from keras import Sequential, Input
from keras.src.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
class_names = ('Cat', 'Dog')

try:
    os.mkdir('dataset/tmp')
    os.mkdir('dataset/tmp/train')
    os.mkdir('dataset/tmp/dev')
    os.mkdir('dataset/tmp/test')
    os.mkdir('dataset/tmp/train/cats')
    os.mkdir('dataset/tmp/dev/cats')
    os.mkdir('dataset/tmp/test/cats')
    os.mkdir('dataset/tmp/train/dogs')
    os.mkdir('dataset/tmp/dev/dogs')
    os.mkdir('dataset/tmp/test/dogs')
except OSError as e:
    print(e)

CAT_IMAGES_DIR = r'dataset/original/cats'
DOG_IMAGES_DIR = r'dataset/original/dogs'
CAT_TRAIN_DIR = r'dataset/tmp/train/cats'
CAT_DEV_DIR = r'dataset/tmp/dev/cats'
CAT_TEST_DIR = r'dataset/tmp/test/cats'
DOG_TRAIN_DIR = r'dataset/tmp/train/dogs'
DOG_DEV_DIR = r'dataset/tmp/dev/dogs'
DOG_TEST_DIR = r'dataset/tmp/test/dogs'


def split_data(original_dir, train_dir, dev_dir, test_dir, split_size=0.9):
    files = []
    for file in os.listdir(original_dir):
        if os.path.getsize(os.path.join(original_dir, file)):
            files.append(file)
    shuffled = random.sample(files, len(files))
    split = int(split_size * len(shuffled))
    train_files = shuffled[:split]
    split_dev_test = int(split + (len(files) - split)/2)

    dev_files = shuffled[split:split_dev_test]
    test_files = shuffled[split_dev_test:]

    for file in train_files:
        copyfile(os.path.join(original_dir, file), os.path.join(train_dir, file))
    for file in dev_files:
        copyfile(os.path.join(original_dir, file), os.path.join(dev_dir, file))
    for file in test_files:
        copyfile(os.path.join(original_dir, file), os.path.join(test_dir, file))

    print('Split Complete!')


split_data(CAT_IMAGES_DIR, CAT_TRAIN_DIR, CAT_DEV_DIR, CAT_TEST_DIR, 0.9)
split_data(DOG_IMAGES_DIR, DOG_TRAIN_DIR, DOG_DEV_DIR, DOG_TEST_DIR, 0.9)

train_gen = ImageDataGenerator(rescale=1./255)
validation_gen = ImageDataGenerator(rescale=1./255.)
test_gen = ImageDataGenerator(rescale=1./255.)

train_generator = train_gen.flow_from_directory(
        'dataset/tmp/train',
        target_size=(150, 150),
        batch_size=64,
        class_mode='binary')
validation_generator = validation_gen.flow_from_directory(
        'dataset/tmp/dev',
        target_size=(150, 150),
        batch_size=64,
        class_mode='binary')
test_generator = test_gen.flow_from_directory(
    'dataset/tmp/test',
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary')


def plot_data(generator, n_images):
    """
    Plots random data from dataset
    Args:
    generator: a generator instance
    n_images : number of images to plot
    """
    i = 1
    images, labels = next(generator)
    labels = labels.astype('int32')

    plt.figure(figsize=(14, 15))

    for image, label in zip(images, labels):
        plt.subplot(4, 3, i)
        plt.imshow(image)
        plt.title(class_names[label])
        plt.axis('off')
        i += 1
        if i == n_images:
            break

    plt.show()


plot_data(train_generator, 7)
plot_data(validation_generator,7)

model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax'),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
r = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator)
model.evaluate(test_generator)
model.save('pet-classifier.h5')