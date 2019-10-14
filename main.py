import argparse

import torch
torch.backends.cudnn.benchmark=True
import torch.utils.data as td
import numpy as np

import data_handler
import experiment as ex
import networks
import trainer
import arguments

torch.set_default_tensor_type('torch.cuda.FloatTensor')

args = arguments.get_args()

log_name = '{}_{}_{}_{}_memsz_{}_alpha_{}_s_{}_beta_{}_lr_{}_batch_{}_epoch_{}'.format(args.date,
                                                                                       args.dataset,
                                                                                       args.trainer,
                                                                                       args.seed,
                                                                                       args.memory_budget,
                                                                                       args.alpha,
                                                                                       args.sample,
                                                                                       args.beta,
                                                                                       args.lr,
                                                                                       args.batch_size,
                                                                                       args.epochs_class)


dataset = data_handler.DatasetFactory.get_dataset(args.dataset)
seed = args.seed
m = args.memory_budget

# Fix the seed.
args.seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Loader used for training data
train_dataset_loader = data_handler.IncrementalLoader(dataset.train_data.train_data,
                                                      dataset.train_data.train_labels,
                                                      dataset.classes,
                                                      args.step_size,
                                                      args.memory_budget,
                                                      'train',
                                                      args.batch_size,
                                                      transform=dataset.train_transform,
                                                      )

# Loader for test data.
test_dataset_loader = data_handler.IncrementalLoader(dataset.test_data.test_data,
                                                     dataset.test_data.test_labels,
                                                     dataset.classes,
                                                     args.step_size,
                                                     args.memory_budget,
                                                     'test',
                                                     args.batch_size,
                                                     transform=dataset.test_transform
                                                     )

kwargs = {'num_workers': 1, 'pin_memory': True} 

# Iterator to iterate over training data.
train_iterator = torch.utils.data.DataLoader(train_dataset_loader,
                                             batch_size=args.batch_size, shuffle=True, **kwargs)

# Iterator to iterate over test data
test_iterator = torch.utils.data.DataLoader(test_dataset_loader,
                                            batch_size=args.batch_size, shuffle=False, **kwargs)

# Get the required model
myModel = networks.ModelFactory.get_model(args.dataset, args.ratio, args.trainer).cuda()
# myModel = networks.ModelFactory.get_model(args.dataset).cuda()

# Define the optimizer used in the experiment
optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
                            weight_decay=args.decay, nesterov=True)

# Trainer object used for training
myTrainer = trainer.TrainerFactory.get_trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer)

# Initilize the evaluators used to measure the performance of the system.
t_classifier = trainer.EvaluatorFactory.get_evaluator("trainedClassifier")

results = np.zeros(dataset.classes // args.step_size)

# Loop that incrementally adds more and more classes
for t in range(dataset.classes//args.step_size):
    print("SEED:", seed, "MEMORY_BUDGET:", m, "tasknum:", t)
    # Add new classes to the train, and test iterator
    myTrainer.update_frozen_model()
    epoch = 0

    # Running epochs_class epochs
    for epoch in range(0, args.epochs_class):
        myTrainer.update_lr(epoch)
        myTrainer.train(epoch)
        # print(my_trainer.threshold)
        if epoch % 5 == (5 - 1):
            TrainError = t_classifier.evaluate(myTrainer.model, train_iterator, t, args.step_size, 'train')
            TestError = t_classifier.evaluate(myTrainer.model, test_iterator, t, args.step_size, 'test')
            print("*********CURRENT EPOCH********** : %d"%epoch)
            print("Train Classifier: %0.2f"%TrainError)
            print("Test Classifier: %0.2f"%TestError)

    
    # Evaluate the learned classifier

    TestError = t_classifier.evaluate(myTrainer.model, test_iterator, t, args.step_size, 'test')
    print("Test Classifier Final: %0.2f"%TestError)
    
    myTrainer.setup_training()
    results[t] = TestError
    np.savetxt('./result_data/'+log_name+'.txt', results, '%.4f')
    torch.save(myModel.state_dict(), './models/trained_model/' + log_name + '_task_{}.pt'.format(t))


