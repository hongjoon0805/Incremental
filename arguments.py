import argparse


def get_args():
    parser = argparse.ArgumentParser(description='iCarl2.0')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 2.0). Note that lr is decayed by args.gamma parameter args.schedule ')
    parser.add_argument('--alpha', type=float, default=10, help='KD convex combination.')
    parser.add_argument('--ratio', type=float, default=1/512, help='variance ratio')
    parser.add_argument('--beta', type=float, default=1e-4, help='beta')
    parser.add_argument('--schedule', type=int, nargs='+', default=[40,80],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seeds values to be used; seed introduces randomness by changing order of classes')
#     parser.add_argument('--decay', type=float, default=0.003, help='Weight decay (L2 penalty).')
    parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    parser.add_argument('--step-size', type=int, default=100, help='How many classes to add in each increment')
    parser.add_argument('--memory-budget', type=int, default=20000,
                        help='How many images can we store at max. 0 will result in fine-tuning')
    parser.add_argument('--epochs-class', type=int, default=100, help='Number of epochs for each increment')
    parser.add_argument('--base-classes', type=int, default=100, help='Number of base classes')
    parser.add_argument('--sample', type=int, default=10, help='Number of samples in BNN')
    parser.add_argument('--factor', type=int, default=4, help='Number of samples in BNN')
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--dataset', default='', type=str, required=True,
                        choices=['CIFAR100', 
                                 'Imagenet'], 
                        help='(default=%(default)s)')
    parser.add_argument('--trainer', default='', type=str, required=True,
                        choices=['lwf', 
                                 'er', 
                                 'bayes',
                                 'gda',
                                 'coreset'], 
                        help='(default=%(default)s)')
    parser.add_argument('--strategy', default='', type=str, required=True,
                        choices=['Reservior', 
                                 'RingBuffer',
                                 'Weighted'], 
                        help='(default=%(default)s)')
    
    args = parser.parse_args()
    return args