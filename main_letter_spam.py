# main_letter_spam.py (Updated for TF 2.x compatibility)

import argparse
import numpy as np

# Load updated modules
from data_loader import data_loader
from gain import gain  # Make sure gain.py is updated for TF 2.x
from utils import rmse_loss

def main(args):
    """Main function for UCI letter and spam datasets."""
    # Parameters
    data_name = args.data_name
    miss_rate = args.miss_rate

    gain_parameters = {
        'batch_size': args.batch_size,
        'hint_rate': args.hint_rate,
        'alpha': args.alpha,
        'iterations': args.iterations
    }

    # Load data and create missing data
    ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)

    # Impute missing data using GAIN
    imputed_data_x = gain(miss_data_x, gain_parameters)

    # Compute RMSE
    rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)

    print(f"\n✅ RMSE Performance: {rmse:.4f}")
    return imputed_data_x, rmse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', choices=['letter', 'spam'], default='spam', type=str)
    parser.add_argument('--miss_rate', help='missing data probability', default=0.2, type=float)
    parser.add_argument('--batch_size', help='mini-batch size', default=128, type=int)
    parser.add_argument('--hint_rate', help='hint rate', default=0.9, type=float)
    parser.add_argument('--alpha', help='hyperparameter', default=100, type=float)
    parser.add_argument('--iterations', help='training iterations', default=10000, type=int)

    args = parser.parse_args()
    imputed_data, rmse = main(args)

