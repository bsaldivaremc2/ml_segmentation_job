import os
import numpy
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.image as mpimg
from tensorflow_examples.models.pix2pix import pix2pix

def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def to_tfrecord_image_and_mask(tfrec_filewriter, img_bytes, mask_bytes):
    feature = {
      "image": _bytestring_feature([img_bytes]), # one image in the list
      "mask": _bytestring_feature([mask_bytes]), # one image in the list
      }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def to_tfrecord_image_and_polygon(tfrec_filewriter, img_bytes, polygon):
    feature = {
      "image": _bytestring_feature([img_bytes]), # one image in the list
      "polygon": _bytestring_feature([polygon]), # one image in the list
      }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def get_image_and_mask_points(ifile,imask_list):
    img = mpimg.imread(ifile)
    mask = np.vstack(imask_list)
    maskx, masky = mask[:,0], mask[:,1]
    return img,maskx,masky

def resize_img_and_mask(filename,polygon,resize_target=None):

    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits)
    image = tf.cast(image, tf.uint8)
    #
    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    #
    mask = create_mask_from_coordinates(polygon,h,w,return_3d=False)
    #
    if type(resize_target)==type(()):
        RESIZE_W, RESIZE_H = resize_target
        image = tf.image.resize(image, [RESIZE_W,RESIZE_H])
        mask = tf.image.resize(mask, [RESIZE_W,RESIZE_H])
    image = tf.cast(image, tf.uint8)
    mask = tf.cast(mask, tf.uint8)
    return image, mask

def create_tfrecords_image_and_mask(idf,prefix="train",output_tfrecords_dir="./tf_records/",
                                            batch_size=32,
                                            bar_or_print="bar"):
    os.makedirs(output_tfrecords_dir,exist_ok=True)
    N = idf.shape[0]
    batches = N//batch_size
    steps = tqdm(range(batches+1))
    if bar_or_print!="bar":
        steps = range(batches+1)
    for batch in steps:
        start=batch*batch_size
        end = start + batch_size
        tdf = idf.iloc[start:end,:]
        data_size = tdf.shape[0] ###not used
        ofilename =  "{}{}-{:02d}-{}.tfrec".format(output_tfrecords_dir,prefix,batch, batch_size)
        with tf.io.TFRecordWriter(ofilename) as out_file:
            for rowi in range(tdf.shape[0]):
                rowx = idf.iloc[rowi]
                filename = rowx['path_image']
                polygon = rowx['Coordinates_mask']
                i, m = resize_img_and_mask(filename,polygon,resize_target=None)
                example = to_tfrecord_image_and_mask(out_file,
                            tf.image.encode_jpeg(i, optimize_size=True, chroma_downsampling=False).numpy(),
                            tf.image.encode_jpeg(m, optimize_size=True, chroma_downsampling=False).numpy())
                out_file.write(example.SerializeToString())
            if bar_or_print!="bar":
                print("Wrote file {} containing {} records".format(ofilename, batch_size))
    if bar_or_print=="bar":
        print("Wrote file {} containing {} records".format(ofilename, batch_size))




def create_tfrecords_image_and_mask_reshape(idf,prefix="train",output_tfrecords_dir="./tf_records/",
                                            batch_size=32,
                                            reshape_target=(255,255),bar_or_print="bar"):
    os.makedirs(output_tfrecords_dir,exist_ok=True)
    N = idf.shape[0]
    batches = N//batch_size
    steps = tqdm(range(batches+1))
    if bar_or_print!="bar":
        steps = range(batches+1)
    for batch in steps:
        start=batch*batch_size
        end = start + batch_size
        tdf = idf.iloc[start:end,:]
        data_size = tdf.shape[0] ###not used
        ofilename =  "{}{}-{:02d}-{}.tfrec".format(output_tfrecords_dir,prefix,batch, batch_size)
        with tf.io.TFRecordWriter(ofilename) as out_file:
            for rowi in range(tdf.shape[0]):
                rowx = idf.iloc[rowi]
                filename = rowx['path_image']
                polygon = rowx['Coordinates_mask']
                i, m = resize_img_and_mask(filename,polygon)
                example = to_tfrecord_image_and_mask(out_file,
                            tf.image.encode_jpeg(i, optimize_size=True, chroma_downsampling=False).numpy(),
                            tf.image.encode_jpeg(m, optimize_size=True, chroma_downsampling=False).numpy())
                out_file.write(example.SerializeToString())
            if bar_or_print!="bar":
                print("Wrote file {} containing {} records".format(ofilename, batch_size))
    if bar_or_print=="bar":
        print("Wrote file {} containing {} records".format(ofilename, batch_size))


def test_tensorflow_reshape(tf_image,tf_mask):
    i, m = tf_image,tf_mask
    ii = i.numpy().astype(np.uint8)
    mm = m.numpy().astype(np.uint8)
    im = ii*mm
    mm = np.dstack([mm for _ in range(3)])
    mmm = mm*255
    o = np.hstack([ii,mmm,im])
    plt.imshow(o)
    plt.show()

def get_image_and_mask_points(ifile,imask_list):
    img = mpimg.imread(ifile)
    mask = np.vstack(imask_list)
    maskx, masky = mask[:,0], mask[:,1]
    return img,maskx,masky

def create_mask_from_coordinates(ipolygon=[(10,10),(10,20),(20,20),(20,10)],width=30,height=30,
                                return_3d=False):
    """
    polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
    width = 200
    height = 200
    """
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(ipolygon, outline=1, fill=1)
    mask = numpy.array(img)
    mask = np.expand_dims(mask,-1)
    if return_3d:
        mask = np.dstack([mask for _ in range(3)])
    return mask.copy()


def plot_sample_with_landmarks(idf,samples=(2,3),figsize=(12,4),dpi=80):
    pics = 1
    for _ in samples:
        pics = pics*_
    #
    df_sample = idf.sample(pics)
    files = df_sample['path_image'].values
    masks = df_sample['Coordinates_mask'].values

    fig, axs = plt.subplots(*samples,figsize=figsize,dpi=dpi)
    for xi,ax in enumerate(axs.flatten()):
        img,xs, ys = get_image_and_mask_points(files[xi],masks[xi])
        #
        height, width, _ = img.shape
        mask = create_mask_from_coordinates(masks[xi],width,height, return_3d=True)
        img_binary = mask*img
        #
        img = np.hstack([img,img_binary])
        #
        ax.imshow(img)
        ax.scatter(xs,ys,marker="x",color="r")
        #ax.set_xticks([],[])
        #ax.set_yticks([],[])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    plt.show()

def get_image_and_mask(filename,polygon,is_mask_binary=True):
    img,xs, ys = get_image_and_mask_points(filename,polygon)
    height, width, _ = img.shape
    mask = create_mask_from_coordinates(polygon,width,height, return_3d=False)
    if not is_mask_binary:
        mask = mask*img
    return img.copy(),mask.copy()


def get_train_val_test_dfs(dataset_file_csv,data_dir,
                           train_groups=['c1anemia', 'no_vars', 'c2anemia', 'hnino'],
                           val_groups = ["c4anemia"],
                           test_groups=["c5anemia"]
                          ):
    #read
    dfx = pd.read_csv(dataset_file_csv)
    valid_cols = "path_image,Coordinates_mask,Coordinates_detection,ane_glo".split(",")
    dfx = dfx[dfx['condition']==1]
    dfx[valid_cols[0]] = dfx[valid_cols[0]].apply(lambda x: x.replace("Data",data_dir))
    for c in "Coordinates_mask,Coordinates_detection".split(","):
        dfx[c] = dfx[c].apply(lambda x: eval(x))
    dfx = dfx[valid_cols]
    tr_df = dfx[dfx['path_image'].apply(lambda x: x.split("/")[-2] in train_groups)]
    val_df = dfx[dfx['path_image'].apply(lambda x: x.split("/")[-2] in val_groups)]
    test_df = dfx[dfx['path_image'].apply(lambda x: x.split("/")[-2] in test_groups)]
    for _df, l in zip([tr_df,val_df,test_df],"train val test".split(" ")):
        print("{} set size: {}".format(l,_df.shape))
    return tr_df.copy(),val_df.copy(),test_df.copy()
