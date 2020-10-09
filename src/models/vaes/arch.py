from tensorflow.keras import models
from tensorflow.keras import layers

EPS = 1.e-5


def encoder(input_shape, latent_dim):
    model = models.Sequential(name='encoder')

    assert input_shape[1] == input_shape[2]
    assert input_shape[1] in (64, 128)

    model.add(layers.Conv2D(kernel_size=4, filters=32, strides=(2, 2), activation='relu',
                            batch_input_shape=input_shape, padding='same'))
    model.add(layers.Conv2D(kernel_size=4, filters=32, strides=(2, 2), activation='relu', padding='same'))
    model.add(layers.Conv2D(kernel_size=4, filters=64, strides=(2, 2), activation='relu', padding='same'))
    model.add(layers.Conv2D(kernel_size=4, filters=64, strides=(2, 2), activation='relu', padding='same'))

    if input_shape[1] == 128:
        model.add(layers.Conv2D(kernel_size=4, filters=256, strides=(2, 2), activation='relu', padding='same'))

    model.add(layers.Conv2D(kernel_size=4, filters=256, strides=(2, 2), activation='relu', padding='valid'))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=2 * latent_dim, activation='linear'))

    return model


def decoder(z_shape, out_size):
    model = models.Sequential(name='decoder')

    assert out_size in (64, 128)

    model.add(layers.Dense(units=256, batch_input_shape=z_shape, activation='linear'))
    model.add(layers.Reshape((1, 1, 256)))

    model.add(layers.Conv2DTranspose(filters=64, kernel_size=4, activation='relu'))

    if out_size == 128:
        model.add(layers.Conv2DTranspose(filters=64, kernel_size=4, activation='relu', strides=(2, 2), padding='same'))

    model.add(layers.Conv2DTranspose(filters=64, kernel_size=4, activation='relu', strides=(2, 2), padding='same'))
    model.add(layers.Conv2DTranspose(filters=32, kernel_size=4, activation='relu', strides=(2, 2), padding='same'))
    model.add(layers.Conv2DTranspose(filters=32, kernel_size=4, activation='relu', strides=(2, 2), padding='same'))
    model.add(layers.Conv2DTranspose(filters=3, kernel_size=4, activation='linear', strides=(2, 2), padding='same'))

    return model


def create_arch_func(input_shape, latent_dim):
    enc_out_func = encoder(input_shape, latent_dim)
    dec_out_func = decoder((input_shape[0], latent_dim), input_shape[1])
    return enc_out_func, dec_out_func


def encoder_bigger(input_shape, latent_dim, trainable):
    model = models.Sequential(name='encoder')

    assert input_shape[1] == input_shape[2]
    assert input_shape[1] in (64, 128, 84, 96)

    model.add(layers.Conv2D(kernel_size=5, filters=128, strides=(2, 2), activation='linear',
                            batch_input_shape=input_shape, padding='same'))
    model.add(layers.BatchNormalization(trainable=trainable))
    model.add(layers.Activation('relu'))

    for filters_num in [256, 512, 1024]:
        model.add(layers.Conv2D(kernel_size=5, filters=filters_num, strides=(2, 2), activation='linear',
                                padding='same'))
        model.add(layers.BatchNormalization(trainable=trainable))
        model.add(layers.Activation('relu'))

    if input_shape[1] == 128:
        model.add(layers.Conv2D(kernel_size=5, filters=1024, strides=(2, 2), activation='relu', padding='same'))
        model.add(layers.BatchNormalization(trainable=trainable))
        model.add(layers.Activation('relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=2 * latent_dim, activation='linear'))

    return model


def decoder_bigger(z_shape, out_size, trainable):
    model = models.Sequential(name='decoder')

    assert out_size in (64, 128, 84, 96)

    if out_size in (64, 128):
        fc1_shape = (8, 8, 1024)
    elif out_size == 96:
        fc1_shape = (12, 12, 1024)
    else:
        fc1_shape = (10, 10, 1024)

    fc1_prod = fc1_shape[0] * fc1_shape[1] * fc1_shape[2]

    model.add(layers.Dense(units=fc1_prod, batch_input_shape=z_shape, activation='linear'))
    model.add(layers.Reshape(fc1_shape))

    if out_size == 128:
        model.add(layers.Conv2DTranspose(filters=512, kernel_size=5, strides=(2, 2), padding='same',
                                         activation='linear'))
        model.add(layers.BatchNormalization(trainable=trainable))
        model.add(layers.Activation('relu'))

    for filters_num in [512, 256, 128]:
        model.add(layers.Conv2DTranspose(filters=filters_num, kernel_size=5, strides=(2, 2), padding='same',
                                         activation='linear'))
        if filters_num == 512 and out_size == 84:
            model.add(layers.ZeroPadding2D(padding=((1, 0), (1, 0))))

        model.add(layers.BatchNormalization(trainable=trainable))
        model.add(layers.Activation('relu'))

    model.add(layers.Conv2DTranspose(filters=3, kernel_size=5, strides=(1, 1), padding='same', activation='linear'))

    return model


def create_bigger_arch_func(input_shape, latent_dim, trainable):
    enc_out_func = encoder_bigger(input_shape, latent_dim, trainable)
    dec_out_func = decoder_bigger((input_shape[0], latent_dim), input_shape[1], trainable)
    return enc_out_func, dec_out_func
