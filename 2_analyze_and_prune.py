import tensorflow as tf
from models import build_resnet, transfer_weights
import numpy as np

MODEL_PATH = "Models/original_resnet.keras"
PRUNED_MODEL_PATH = "Models/pruned_resnet.keras"

def analyze_weights(model):
    """Calculates and plots the L2 norms and variances of the Conv2D layers in the Keras model."""
    print(f"{'Layer name':<25} | {'L2 Norm':<10} | {'Variance':<10} | {'Death Rate (<1e-4)':<15}")
    print("-" * 65)

    stats = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            weights = layer.get_weights()[0]  # [0] -> kernel (filter), [1] -> bias (we didn't use bias anyway)

            l2_norm = np.linalg.norm(weights)
            variance = np.var(weights)
            dead_ratio = np.mean(np.abs(weights) < 1e-4) * 100

            stats[layer.name] = {'variance': variance, 'l2': l2_norm}
            print(f"{layer.name:<25} | {l2_norm:<10.4f} | {variance:<10.6f} | % {dead_ratio:<13.2f}")

    return stats


def main():
    print("=== FAZ 2: WEIGHT ANALYSIS AND PHYSICAL PRUNING ===")

    # 1. Upload the Original Model
    print(f"'{MODEL_PATH}' download")
    original_model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Show Statistics (Here we can decide which blocks to prune)
    print("\n--- TRAINED LAYER STATISTICS ---")
    stats = analyze_weights(original_model)

    # Note: We should prune those with the lowest variance or L2 norm (usually the last layers). # For now, let's delete the last block and the second-to-last one, as in our JAX experiment.
    targets = ['stage3_block2', 'stage2_block2']
    print(f"\nTarget Pruning Blocks: {targets}")

    #3. Create the New (Incomplete) Model
    print("\nNew (Small) Architecture is Being Established...")
    pruned_model = build_resnet(blocks_to_skip=targets)

    print(f"\nOriginal Parameters : {original_model.count_params():,}")
    print(f"Pruned Parameters : {pruned_model.count_params():,}")
    print(
        f"The Contraction That Occurred   : % {((original_model.count_params() - pruned_model.count_params()) / original_model.count_params()) * 100:.2f}")

    # 4. Network Surgery (Copy Remaining Weights Only)
    pruned_model = transfer_weights(original_model, pruned_model)

    # 5. Save the new (smaller) model to disk.
    pruned_model.save(PRUNED_MODEL_PATH)
    print(f"\nThe lightweight model resulting from surgery was saved as '{PRUNED_MODEL_PATH}'.")

if __name__ == "__main__":
    main()