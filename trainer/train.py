import os
import argparse
from tqdm import tqdm
import tensorflow as tf
from train_funcs import *
from models.mobilenetv2_unet import *

parser = argparse.ArgumentParser(description='Generate tf records for anemia dataset')

parser.add_argument('-tf_dir', type=str,default='/home/bsaldivar/ml/TFrecords/',
                    dest="tf_dir",help='Directory where the tf records data is located')
parser.add_argument('-val_data_groups', type=str,default="c4anemia",dest="val_data_groups",help='group label as validation set. comma , separated')
parser.add_argument('-test_data_groups', type=str,default="c5anemia",dest="test_data_groups",help='group label as test set. comma , separated')
parser.add_argument('-train_data_groups', type=str,default="c1anemia,no_vars,c2anemia,hnino",dest="train_data_groups",help='group label as training set. comma separated')
parser.add_argument('-batch_size', type=int ,default=32,dest="batch_size",help='batch size')
parser.add_argument('-buffer_size', type=int ,default=32,dest="buffer_size",help='buffer size')
parser.add_argument('-target_resize', type=str ,default="(224,224)",dest="target_resize",help='target resize (224,224)')
parser.add_argument('-output_dir', type=str,default='/home/bsaldivar/ml/output_artifacts/',
                    dest="output_dir",help='Directory where to store the models and checkpoints')
parser.add_argument('-epochs', type=int ,default=10,dest="epochs",help='epochs')
parser.add_argument('-steps_per_epoch', type=int ,default=10,dest="steps_per_epoch",help='steps_per_epoch')
parser.add_argument('-validation_steps', type=int ,default=1,dest="validation_steps",help='validation_steps')

args = parser.parse_args()

def main(args):
    train_pattern = args.tf_dir + "train*.tfrec"
    val_pattern = args.tf_dir + "val*.tfrec"
    test_pattern = args.tf_dir + "test*.tfrec"
    train_ds = get_dataset_from_tfrecord(train_pattern)
    test_ds = get_dataset_from_tfrecord(test_pattern)
    val_ds = get_dataset_from_tfrecord(val_pattern)
    BATCH_SIZE = args.batch_size
    BUFFER_SIZE = args.buffer_size
    TARGET_SIZE = eval(args.target_resize)
    EPOCHS = args.epochs
    STEPS_PER_EPOCH = args.steps_per_epoch
    VALIDATION_STEPS = args.validation_steps
    DATA_DIR = args.output_dir
    #tr_ds = train_ds.map(read_tfrecord).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE).repeat()
    tr_ds = train_ds.map(lambda x: read_tfrecord(x,TARGET_SIZE)).map(random_flip).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    vl_ds = val_ds.map(lambda x: read_tfrecord(x,TARGET_SIZE)).map(random_flip).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    unet = unet_model(output_channels=1,input_shape=[224,224,3])
    unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=CustomDiceLoss(),
              metrics=[dice_coef])
              #tf.keras.utils.plot_model(unet, show_shapes=True)
    checkpoint_filepath = '{}checkpoints/'.format(DATA_DIR)
    os.makedirs(checkpoint_filepath,exist_ok=True)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_dice_coef',
        mode='max',
        save_best_only=True)
    model_history = unet.fit(tr_ds.repeat(), epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=vl_ds.repeat(),
                        callbacks=[model_checkpoint_callback])
    unet.load_weights(checkpoint_filepath)
    val_loss, val_metric = unet.evaluate(vl_ds)
    val_metric = round(val_metric,3)
    model_export_path = "{}models/anemia_segmentation/val_dice_coef_{}/".format(DATA_DIR,val_metric)
    os.makedirs(model_export_path,exist_ok=True)
    tf.saved_model.save(unet, model_export_path)
    print("saved model",model_export_path)

if __name__ =="__main__":
    main(args)
