from tensorflow.keras import layers, models

def attention_block(x, g, inter_shape):
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(g)
    concat_xg = layers.add([theta_x, phi_g])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    upsample_psi = layers.UpSampling2D(size=(2, 2))(sigmoid_xg)
    return layers.Multiply()([upsample_psi, x])

def attention_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    conv1 = conv_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 512)

    # Decoder with attention
    up1 = layers.UpSampling2D(size=(2, 2))(conv4)
    att1 = attention_block(conv3, up1, 256)
    up1 = layers.Concatenate()([up1, att1])
    conv5 = conv_block(up1, 256)

    up2 = layers.UpSampling2D(size=(2, 2))(conv5)
    att2 = attention_block(conv2, up2, 128)
    up2 = layers.Concatenate()([up2, att2])
    conv6 = conv_block(up2, 128)

    up3 = layers.UpSampling2D(size=(2, 2))(conv6)
    att3 = attention_block(conv1, up3, 64)
    up3 = layers.Concatenate()([up3, att3])
    conv7 = conv_block(up3, 64)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv7)

    return models.Model(inputs=[inputs], outputs=[outputs])
