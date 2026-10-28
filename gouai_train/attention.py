"""
attention.py
------------
Spatial-attention block (the spatial branch of CBAM, "Convolutional Block Attention Module",
Woo et al. 2018), placed after the ResNet-50 backbone.

For every position of the feature map it computes the mean and the maximum over all channels,
passes these two maps through a 7x7 convolution with a sigmoid, and multiplies the feature map by
the resulting weight (between 0 and 1). The network can thus emphasize the regions of the fireball
that matter (e.g. the dense core or the edges) and damp the empty corners of the 280 x 280 grid.

Identical to the block of the published v6 models (needed to load their weights).
"""
import tensorflow as tf
from keras import layers


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)   # (batch, h, w, 1)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)    # (batch, h, w, 1)
        attention = self.conv(tf.concat([avg_pool, max_pool], axis=-1))
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


class CBAMBlock(layers.Layer):
    """CBAM with the spatial attention only (no channel attention), as in the v6 models."""

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.spatial_attention = SpatialAttention(kernel_size=self.kernel_size)

    def call(self, inputs):
        return self.spatial_attention(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config
