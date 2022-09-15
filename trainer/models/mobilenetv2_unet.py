import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

class NormInputImage(tf.keras.layers.Layer):
    def __init__(self):
        super(NormInputImage, self).__init__()
    def call(self, inputs):
        return inputs/255

class ReshapeInputImage(tf.keras.layers.Layer):
    def __init__(self,target_image_size=(128, 128)):
        super(ReshapeInputImage, self).__init__()
        self.target_image_size = target_image_size
    def call(self, inputs):
        return tf.image.resize(inputs, self.target_image_size)

def unet_model(output_channels,input_shape):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False
    #
    up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]
    #
    _inputs = tf.keras.Input(shape=[None,None,3],name="image")##<---- change
    norm_in = NormInputImage()(_inputs)
    inputs = ReshapeInputImage(input_shape[:2])(norm_in)

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

  # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128
    x = last(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    return tf.keras.Model(inputs=_inputs, outputs=x)

def dice_coef(y_true, y_pred, smooth=100):
    tp_raw = tf.math.multiply(y_true,y_pred)
    tp = tf.math.reduce_sum(tp_raw)
    #
    neg_y_pred = (y_pred - 1)*-1
    neg_y_true = (y_true - 1)*-1
    #
    fp_raw = tf.math.multiply(neg_y_true,y_pred)
    fp = tf.math.reduce_sum(fp_raw)
    #
    fn_raw = tf.math.multiply(y_true,neg_y_pred)
    fn = tf.math.reduce_sum(fn_raw)
    #
    dice = (2*tp)/(2*tp+fp+fn)
    return dice

#2 TP / (2 TP + FP + FN)
class CustomDiceLoss(tf.keras.losses.Loss):
    def __init__(self,smooth=100):
        super().__init__()
        self.smooth=smooth
    def call(self, y_true, y_pred):
        tp_raw = tf.math.multiply(y_true,y_pred)
        tp = tf.math.reduce_sum(tp_raw)
        #
        neg_y_pred = (y_pred - 1)*-1
        neg_y_true = (y_true - 1)*-1
        #
        fp_raw = tf.math.multiply(neg_y_true,y_pred)
        fp = tf.math.reduce_sum(fp_raw)
        #
        fn_raw = tf.math.multiply(y_true,neg_y_pred)
        fn = tf.math.reduce_sum(fn_raw)
        #
        dice = (2*tp)/(2*tp+fp+fn)
        return 1 - dice
