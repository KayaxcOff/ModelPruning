from tensorflow.keras import layers, models

def resnet_block(x, filters, stride=1, name='block'):
    """
    A named, skip-connection standard ResNet block.
    """

    shortcut = x

    # If the number of channels changes or the size decreases, we need to synchronize the shortcut as well.
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False, name=f'{name}_shortcut_conv')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)

    # 1. Convolution
    y = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False, name=f'{name}_conv1')(x)
    y = layers.BatchNormalization(name=f'{name}_bn1')(y)
    y = layers.Activation('relu', name=f'{name}_relu1')(y)

    # 2. Convolution
    y = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False, name=f'{name}_conv2')(y)
    y = layers.BatchNormalization(name=f'{name}_bn2')(y)

    # Skip Connection
    y = layers.Add(name=f'{name}_add')([shortcut, y])
    y = layers.Activation('relu', name=f'{name}_relu2')(y)

    return y

def build_resnet(input_shape=(32, 32, 1), num_classes=62, blocks_to_skip=None):
    """
    A model constructor that DOES NOT physically ADD blocks from the blocks_to_skip list to the graph.
    """

    if blocks_to_skip is None:
        blocks_to_skip = []

    inputs = layers.Input(shape=input_shape, name='input_layer')

    x = layers.Conv2D(32, 3, padding='same', use_bias=False, name='init_conv')(inputs)
    x = layers.BatchNormalization(name='init_bn')(x)
    x = layers.Activation('relu', name='init_relu')(x)

    # --- STAGE 1 (32 Filter) ---
    if 'stage1_block1' not in blocks_to_skip:
        x = resnet_block(x, 32, name='stage1_block1')
    if 'stage1_block2' not in blocks_to_skip:
        x = resnet_block(x, 32, name='stage1_block2')

    # --- STAGE 2 (64 Filter - Size 16x16) ---
    # Not: Because block 1 produces downsamples, pruning it is generally not recommended.
    if 'stage2_block1' not in blocks_to_skip:
        x = resnet_block(x, 64, stride=2, name='stage2_block1')
    if 'stage2_block2' not in blocks_to_skip:
        x = resnet_block(x, 64, name='stage2_block2')

    # --- STAGE 3 (128 Filter - Size 8x8') ---
    if 'stage3_block1' not in blocks_to_skip:
        x = resnet_block(x, 128, stride=2, name='stage3_block1')
    if 'stage3_block2' not in blocks_to_skip:
        x = resnet_block(x, 128, name='stage3_block2')

    # --- Exit (Classifier) ---
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='classifier')(x)

    model = models.Model(inputs, outputs, name="Sanskrit_ResNet")
    return model


def transfer_weights(original_model, pruned_model):
    """
    NETWORK SURGERY: Copies the weights from the original model to a trimmed-down model. Only layers with matching names (surviving layers) are transferred.
    """
    print("\nWeight transfer begins..")
    transfer_count = 0

    for layer in pruned_model.layers:
        # Only consider layers with weight (Conv2D, Dense, BN)
        if len(layer.weights) > 0:
            try:
                # Find the layer with the same name as the original model
                orig_layer = original_model.get_layer(layer.name)
                # Copy weights
                layer.set_weights(orig_layer.get_weights())
                transfer_count += 1
            except ValueError:
                # If no matching name is found, don't give an error, just skip it (this shouldn't happen in this scenario, but as a precaution)
                print(f"Warning: {layer.name} was not found in the original model.!")

    print(f"Success! The weight of {transfer_count} layers has been transferred to the new model.")
    return pruned_model


if __name__ == "__main__":
    # Install original model
    original = build_resnet()
    print(f"Original Model Parameter Count: {original.count_params():,}")

    # 2 Install the pruned block model.
    pruned = build_resnet(blocks_to_skip=['stage2_block2', 'stage3_block2'])
    print(f"Number of Trimmed Model Parameters: {pruned.count_params():,}")

    # Transfer simulation
    pruned = transfer_weights(original, pruned_model=pruned)