python3 train.py -epochs=2 -batch_size=64 -steps_per_epoch=30

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
