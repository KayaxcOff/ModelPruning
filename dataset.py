import tensorflow as tf
from matplotlib import pyplot as plt

BATCH_SIZE = 64
IMAGE_SIZE = (32, 32)

DATA_DIR = 'archive/Sanskrit Mnist/images'

def load_datasets(data_dir = DATA_DIR):
    """
    Load an image from a file path, make it black-and-white, scale
    """
    print(f'Loading images from {data_dir}')

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.30,  # %70 for train
        subset="training",
        seed=123,  # Seed is IMPORTANT! Using the same seed will prevent data mix-ups.
        color_mode="grayscale",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.30,
        subset="validation",
        seed=123,  # It must be same seed
        color_mode="grayscale",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"\nNumber of Classes Found: {num_classes}")

    # --- DIVIDE 30% INTO TWO PARTS: VAL AND TEST ---
    # Find the total number of batches
    val_test_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_batches = val_test_batches // 2

    # The first half is Validation (15%), the remaining half is Testing (15%)
    val_ds = val_test_ds.take(val_batches)
    test_ds = val_test_ds.skip(val_batches)

    print(f"Training Batch Number: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Verification (VAL) Batch Count: {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"Number of Test Batches: {tf.data.experimental.cardinality(test_ds).numpy()}")

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    def preprocess(image, label):
        return normalization_layer(image), label

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, num_classes

def test_dataset():
    """
    Test for load_datasets()
    """
    train_ds, val_ds, test_ds, num_classes = load_datasets()

    for images, labels in train_ds.take(1):
        print(f"\nSize of a package: {images.shape} (Batch, Height, Width, Channel)")
        print(f"Pixel value range: Min={tf.reduce_min(images):.2f}, Max={tf.reduce_max(images):.2f}")

        plt.figure(figsize=(10, 2))
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            plt.imshow(images[i].numpy().squeeze(), cmap='gray')
            plt.title(f"Class: {labels[i].numpy()}")
            plt.axis("off")
        plt.show()

if __name__ == "__main__":
    test_dataset()