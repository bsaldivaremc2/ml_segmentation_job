#@tf.function
import tensorflow as tf
def get_dataset_from_tfrecord(tf_pattern):
    AUTO = tf.data.experimental.AUTOTUNE
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False
    train_filenames = tf.io.gfile.glob(tf_pattern)
    ds = tf.data.TFRecordDataset(train_filenames, num_parallel_reads=AUTO)
    ds = ds.with_options(option_no_order)
    return ds

def read_tfrecord(example,target_size=(224,224)):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "mask": tf.io.FixedLenFeature([], tf.string)  # tf.string = bytestring (not text string)
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    mask = tf.image.decode_jpeg(example['mask'], channels=1)
    image = tf.cast(image,tf.float32)
    mask = tf.cast(mask,tf.float32)
    image = tf.image.resize(image, (target_size))
    mask = tf.image.resize(mask, (target_size))
    return image,mask

def random_flip(x,y):
    image,mask = x,y
    #mask = y
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask
