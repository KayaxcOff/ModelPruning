import tensorflow as tf
from dataset import load_datasets
from models import build_resnet

# HYPERPARAMETER
EPOCHS = 10  # 10 epoch for start
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "Models/original_resnet.keras"


def main():
    print("=== FAZ 1: ORIGINAL MODEL TRAINING ===")

    # 1. Upload datasets
    train_ds, val_ds, test_ds, num_classes = load_datasets()

    # 2. Build model
    model = build_resnet(num_classes=num_classes)
    model.summary()

    # 3. Compile
    # We use sparse_categorical because our tags are not one-hot encoded (they are directly 0, 1, 2...).
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # 4. Callbacks
    # ModelCheckpoint only registers the best model (the one with the lowest validation loss).
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )

    # 5. Training Loop
    print("\nTraining Begins...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, reduce_lr]
    )

    print(f"\n Training completed. The best model has been saved as '{MODEL_SAVE_PATH}'.")

    print("\nThe model is being evaluated on a test set..")
    # We are uploading the best model we have recorded.
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
    print(f"Original Model Test Accuracy: % {test_acc * 100:.2f}")

if __name__ == "__main__":
    main()