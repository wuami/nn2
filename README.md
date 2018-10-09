# nn2
A framework for modeling nucleic acid nearest neighbor motif energies with a CNN

```
usage: train_model.py [-h] [-o OPTIMIZER] [-e EPOCHS] [-b BATCH_SIZE]
                      [-p KEEPPROB] [-z REGULARIZE] [-l LEARNING_RATE]
                      [-m] [--batch_norm] [-r RESTORE] [-s] [--sterr]
                      [-g NUM_GPUS] [-t TESTFILE] [-c SCALE] [-n NORM]
                      [-d SEED] [-a ALPHA] [--linear] [--dilate]
                      filename n_units

train a CNN for nearest neighbor parameters

positional arguments:
  filename              name of parameter file
  n_units               numbers of units in hidden layers

optional arguments:
  -h, --help            show this help message and exit
  -o OPTIMIZER, --optimizer OPTIMIZER
                        optimizer type (either "descent", "adam", "adagrad",
                        or "rmsprop")
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        number of data points in each training batch
  -p KEEPPROB, --keepprob KEEPPROB
                        probability to keep a node in dropout
  -z REGULARIZE, --regularize REGULARIZE
                        regularization parameter for motifs
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate for model training
  -m, --low_mem         low memory mode
  --batch_norm          enable batch normalization
  -r RESTORE, --restore RESTORE
                        file to restore weights from
  -s, --save            save log files and models
  --sterr               whether not to use standard error in loss function
  -g NUM_GPUS, --num_gpus NUM_GPUS
                        number of gpus
  -t TESTFILE, --testfile TESTFILE
                        test dataset
  -c SCALE, --scale SCALE
                        scale factor between dG and dH
  -n NORM, --norm NORM  order of norm for loss function
  -d SEED, --seed SEED  random seed
  -a ALPHA, --alpha ALPHA
                        regularization parameter for weights
  --linear              linear least squares fit
  --dilate              use dilated convlutions
```
