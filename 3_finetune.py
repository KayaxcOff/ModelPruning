import tensorflow as tf
from dataset import load_datasets
import os

PRUNED_MODEL_PATH = "Models/pruned_resnet.keras"
FINAL_MODEL_PATH = "Models/final_resnet_finetuned.keras"

FINE_TUNE_LR = 1e-4
EPOCHS = 5


def main():
    print("=== PHASE 3: SHOCK RECOVERY AND FINE-TUNING ===")

    train_ds, val_ds, test_ds, num_classes = load_datasets()

    print(f"\n'{PRUNED_MODEL_PATH}' download...")
    model = tf.keras.models.load_model(PRUNED_MODEL_PATH)

    # We need to compile the model because only the architecture and weights came from the file.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # 3. SHOCK TEST: How much did the success rate drop before fine-tuning?
    print("\n[SHOCK TEST] The trimmed model is tested DIRECTLY without any fine-tuning....")
    loss_before, acc_before = model.evaluate(test_ds, verbose=0)
    print(f"Post-Surgical (Uninformed) Test Success: % {acc_before * 100:.2f}")

    # 4. Fine-Tuning Training Cycle
    print(f"\nFine-tuning begins... ({EPOCHS} Epoch, LR={FINE_TUNE_LR})")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        FINAL_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )

    print("\n[FINAL TEST] The best model is evaluated after fine-tuning....")
    final_model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    final_loss, final_acc = final_model.evaluate(test_ds, verbose=0)

    print("\n" + "=" * 50)
    print("PROJECT FINAL REPORT")
    print("=" * 50)
    print(f"Original Success (704K Parameters)   : % 99.50 (Reference)")
    print(f"Shock Success (Immediately After Surgery): % {acc_before * 100:.2f}")
    print(f"FINAL SUCCESS (334K Parameters)    : % {final_acc * 100:.2f}")
    print("=" * 50)
    print(f"\nThe project has been successfully completed. The final model.: '{FINAL_MODEL_PATH}'")


if __name__ == "__main__":
    main()