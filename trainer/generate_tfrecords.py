import os
import sys
import copy
import argparse
from tqdm import tqdm
from data_prep_funcs import *

parser = argparse.ArgumentParser(description='Generate tf records for anemia dataset')

parser.add_argument('-data_dir', type=str,default='/home/bsaldivar/ml/',
                    dest="data_dir",help='Directory where the data is located')
parser.add_argument('-val_data_groups', type=str,default="c4anemia",dest="val_data_groups",help='group label as validation set. comma , separated')
parser.add_argument('-test_data_groups', type=str,default="c5anemia",dest="test_data_groups",help='group label as test set. comma , separated')
parser.add_argument('-train_data_groups', type=str,default="c1anemia,no_vars,c2anemia,hnino",dest="train_data_groups",help='group label as training set. comma separated')
parser.add_argument('-shard_size', type=int ,default=128,dest="shard_size",help='group data in packs of shard_size')

args = parser.parse_args()

def main(args):
    DATA_DIR = args.data_dir
    DATASET_FILE_CSV = "{}dataset_anemia_upch/data_anemia.csv".format(DATA_DIR)
    OUTPUT_TFRECORDS_DIR = "{}TFrecords/".format(DATA_DIR)
    VAL_DATA_GROUPS = args.val_data_groups
    TEST_DATA_GROUPS = args.test_data_groups
    TRAIN_DATA_GROUPS =args.train_data_groups
    #https://stackoverflow.com/questions/52191167/optimal-size-of-a-tfrecord-file
    SHARD_SIZE=args.shard_size

    #D = "/home/bsaldivar/ml/dataset_anemia_upch/"
    #print(os.listdir(D))

    tr_df,val_df,test_df = get_train_val_test_dfs(DATASET_FILE_CSV,DATA_DIR,
                                train_groups=TRAIN_DATA_GROUPS.split(","),
                               val_groups = VAL_DATA_GROUPS.split(","),
                               test_groups=TEST_DATA_GROUPS.split(","))

    create_tfrecords_image_and_mask(val_df,prefix="val",
                                    output_tfrecords_dir=OUTPUT_TFRECORDS_DIR,
                                                batch_size=SHARD_SIZE,
                                           bar_or_print="bar")


    create_tfrecords_image_and_mask(test_df,prefix="test",
                                    output_tfrecords_dir=OUTPUT_TFRECORDS_DIR,
                                                batch_size=SHARD_SIZE,
                                           bar_or_print="bar")

    create_tfrecords_image_and_mask(tr_df,prefix="train",
                                    output_tfrecords_dir=OUTPUT_TFRECORDS_DIR,
                                                batch_size=SHARD_SIZE,
                                           bar_or_print="bar")


if __name__ =="__main__":
    main(args)
