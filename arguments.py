import argparse


def get_args():
    parser = argparse.ArgumentParser(description='iCarl2.0')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=2.0, metavar='LR',
                        help='learning rate (default: 2.0). Note that lr is decayed by args.gamma parameter args.schedule ')
    parser.add_argument('--schedule', type=int, nargs='+', default=[45, 60, 68],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seeds values to be used; seed introduces randomness by changing order of classes')
    parser.add_argument('--decay', type=float, default=0.00005, help='Weight decay (L2 penalty).')
    parser.add_argument('--step-size', type=int, default=10, help='How many classes to add in each increment')
    parser.add_argument('--memory-budget', type=int, default=2000,
                        help='How many images can we store at max. 0 will result in fine-tuning')
    parser.add_argument('--epochs-class', type=int, default=70, help='Number of epochs for each increment')
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--dataset', default="CIFAR100", help='Dataset to be used; example CIFAR, Imagenet')

    args = parser.parse_args()
    return args