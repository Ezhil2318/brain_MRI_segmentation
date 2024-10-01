from tensorflow.keras import layers, models

def conv_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    return x

def unet_pp(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    conv1 = conv_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 512)

    # Decoder (with nested skip connections)
    up1 = layers.UpSampling2D(size=(2, 2))(conv4)
    up1 = layers.Concatenate()([up1, conv3])
    conv5 = conv_block(up1, 256)

    up2 = layers.UpSampling2D(size=(2, 2))(conv5)
    up2 = layers.Concatenate()([up2, conv2])
    conv6 = conv_block(up2, 128)

    up3 = layers.UpSampling2D(size=(2, 2))(conv6)
    up3 = layers.Concatenate()([up3, conv1])
    conv7 = conv_block(up3, 64)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv7)

    return models.Model(inputs=[inputs], outputs=[outputs])
