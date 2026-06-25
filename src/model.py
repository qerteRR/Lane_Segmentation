from tensorflow.keras import Model, layers


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)

    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 256)

    u1 = layers.UpSampling2D()(c3)
    u1 = layers.Concatenate()([u1, c2])
    c4 = conv_block(u1, 128)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c1])
    c5 = conv_block(u2, 64)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c5)

    return Model(inputs, outputs, name="unet_tusimple_lane_segmentation")
