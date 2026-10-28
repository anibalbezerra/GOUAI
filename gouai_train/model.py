"""
model.py
--------
The network (identical for both collision systems and both input densities).

    image (280, 280, C)  -> ResNet-50 (random initialization, no ImageNet weights)
                         -> spatial attention (attention.CBAMBlock)
                         -> Conv2D(512, 3x3, dilation 2, ReLU) -> global average pooling
                         -> Dense(1024, ReLU) -> Dropout(0.3) -> Dense(128, ReLU)        "image features"
    scalars (n_scalars,) -> Dense(64) -> Dense(32) -> Dense(16)  (ReLU)                  "scalar features"
    concatenate -> Dense(128, ReLU)                                                       shared representation
    two heads, one per target:
        Dense(32, ReLU, L2) -> Dropout(0.4) -> Dense(1, linear, L2, ZERO-initialized)
    output (batch, 2) = [z_Nch, z_pT]: standardized residuals beyond the trend g(S) (targets.py)

With the v6 defaults C = 2 (x/255, x/S) and n_scalars = 1 (S/max S): this is exactly the published
architecture, so the published weights load into build_model(tcfg).

Zero initialization of the last layer of each head: the untrained network outputs z = 0, i.e. exactly
the trend prediction y = g(S) exp(mu). Training starts from "S alone" and only learns corrections.
(Step 04 checks this.)

Loss: Huber(z_Nch) + Huber(z_pT) with delta = 1: quadratic for small errors, linear for large ones,
so a few badly predicted events cannot dominate the gradient.
"""
import numpy as np

from . import gpu_env  # noqa: F401  (must precede tensorflow, see gpu_env.py)
import tensorflow as tf  # noqa: E402

from .attention import CBAMBlock  # noqa: E402


def build_model(tcfg, n_channels=2, n_scalars=1, grid_n=280):
    reg = tf.keras.regularizers.L2(tcfg['head_l2'])
    image_input = tf.keras.layers.Input(shape=(grid_n, grid_n, n_channels), name='image_input')
    sum_input = tf.keras.layers.Input(shape=(n_scalars,), name='sum_input')

    # image branch
    base = tf.keras.applications.ResNet50(weights=None, include_top=False, input_tensor=image_input)
    x = CBAMBlock()(base.output)
    x = tf.keras.layers.Conv2D(512, kernel_size=3, dilation_rate=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(tcfg['trunk_dropout'])(x)
    image_features = tf.keras.layers.Dense(128, activation='relu')(x)

    # scalar branch
    y = tf.keras.layers.Dense(64, activation='relu')(sum_input)
    y = tf.keras.layers.Dense(32, activation='relu')(y)
    sum_features = tf.keras.layers.Dense(16, activation='relu')(y)

    shared = tf.keras.layers.Dense(128, activation='relu')(
        tf.keras.layers.Concatenate()([image_features, sum_features]))

    # one residual head per target
    heads = []
    for name in ('z_nch', 'z_pt'):
        h = tf.keras.layers.Dense(tcfg['head_units'], activation='relu', kernel_regularizer=reg)(shared)
        h = tf.keras.layers.Dropout(tcfg['head_dropout'])(h)
        h = tf.keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros',
                                  kernel_regularizer=reg, name=name)(h)
        heads.append(h)
    output = tf.keras.layers.Concatenate(name='z')(heads)
    return tf.keras.Model(inputs={'image_input': image_input, 'sum_input': sum_input}, outputs=output)


def build_for_run(cfg):
    """The network for the channels and scalars of this experiment."""
    return build_model(cfg['training'], n_channels=len(cfg['features']['channels']),
                       n_scalars=len(cfg['features']['scalars']), grid_n=cfg['data']['grid_n'])


def make_loss(delta):
    huber = tf.keras.losses.Huber(delta=delta)

    def residual_loss(z_true, z_pred):
        return huber(z_true[:, 0], z_pred[:, 0]) + huber(z_true[:, 1], z_pred[:, 1])
    return residual_loss


def mae_z_nch(z_true, z_pred):
    return tf.reduce_mean(tf.abs(z_true[:, 0] - z_pred[:, 0]))


def mae_z_pt(z_true, z_pred):
    return tf.reduce_mean(tf.abs(z_true[:, 1] - z_pred[:, 1]))


def compile_model(model, tcfg):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tcfg['learning_rate']),
                  loss=make_loss(tcfg['huber_delta']), metrics=[mae_z_nch, mae_z_pt])
    return model


def predict_z(model, images, scalars, batch_size=32):
    """Raw network output z (N, 2) for prepared images and normalized scalars."""
    return model.predict({'image_input': images, 'sum_input': np.asarray(scalars, np.float32)},
                         batch_size=batch_size, verbose=0)
