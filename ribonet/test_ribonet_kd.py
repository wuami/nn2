import argparse
import model_utils as m
import ensemble as e
import settings as s
import re
import os

def parse_args():
    parser = argparse.ArgumentParser(description='train a CNN for '
                                                 'nearest neighbor parameters')
    parser.add_argument('filename', help='name of test file')
    parser.add_argument('restore', help='file to restore weights from')
    parser.add_argument('-n', '--n_units', help='numbers of units in hidden '
                        'layers')
    parser.add_argument('-m', '--low_mem', help='low memory mode',
                        action='store_false')
    parser.add_argument('--sterr', help='whether not to use standard error in '
                        'loss function', action='store_true')
    parser.add_argument('-g', '--num_gpus', help='number of gpus', type=int,
                        default=1)
    parser.add_argument('--dilate', help='use dilated convlutions',
                        action='store_true')
    return parser.parse_args()


def main():
    """ test existing model on given data """
    args = parse_args()
    if args.filename.endswith('kd'):
        data = e.KdDataset([args.filename])
    elif args.filename.endswith('melt'):
        data = e.MeltDataset([args.filename])
    elif args.filename.endswith('dG'):
        data = e.dGDataset(args.filename)

    if args.n_units is None:
        match = re.search('([0-9]+x[0-9]+)layer', args.restore)
        try:
            args.n_units = match.group(1)
        except:
            raise ValueError('unable to parse number of units from model name')
    
    if args.filename.endswith('dG'):
        cnn = m.MeltTrainer(data, args.n_units, num_gpus=args.num_gpus,
                            low_mem=args.low_mem, dilate=args.dilate)
    else:
        cnn = m.RiboNetTrainer(data, args.n_units, num_gpus=args.num_gpus,
                               low_mem=args.low_mem, dilate=args.dilate)

    cnn.restore(args.restore)
    print 'rmse: %s' % cnn.get_loss(sterr=args.sterr)
    print 'r2: %s' % cnn.get_r2()
    cnn.write_results_file('%s/%s_%s.txt' % (s.RESULTS_DIR, os.path.basename(args.restore),
                                             os.path.basename(args.filename)))
    print 'results written to %s/%s_%s.txt' % (s.RESULTS_DIR, os.path.basename(args.restore),
                                               os.path.basename(args.filename))


if __name__ == '__main__':
    main()
