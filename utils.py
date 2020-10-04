import torch
import numpy

def load_models(args, model, t):
    
    if t==0:
        name = 'models/trained_model/{}_step_{}_nepochs_{}_{}'.format(args.dataset, 
                                                                      args.base_classes, 
                                                                      args.nepochs, 
                                                                      args.trainer)
        if args.trainer == 'il2m':
            name = 'models/trained_model/{}_{}_{}.pt'.format(args.dataset, args.base_classes, 'ft')
    elif t>0:
        trainer = args.trainer if args.trainer != 'il2m' else 'ft'
        
        name = 'models/trained_model/CVPR_{}_{}_{}_memsz_{}_base_{}_step_{}_batch_{}_epoch_{}'.format(
            args.dataset, 
            trainer, 
            args.seed, 
            args.memory_budget, 
            args.base_classes, 
            args.step_size, 
            args.batch_size, 
            args.nepochs)
        
        if args.trainer == 'ssil':
            name += '_replay_{}'.format(args.replay_batch_size)
        
        if args.trainer == 'ssil' or 'ft' in  args.trainer or args.trainer == 'il2m':
            name += '_factor_{}'.format(args.factor)
    flag = 0
    
    try:
        state_dict = torch.load(name + '_task_{}.pt'.format(t))
        model.load_state_dict(state_dict)
        flag = 1
    except:
        print('Failed to load Pre-trained model')
        print('Model training start')
        flag = 0 
    
    return flag
