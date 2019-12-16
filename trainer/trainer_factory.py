class TrainerFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer):
        
        if args.trainer == 'lwf':
            import trainer.lwf as trainer
        elif args.trainer == 'er':
            import trainer.er as trainer
        elif args.trainer == 'bayes':
            import trainer.bayes as trainer
        elif args.trainer == 'gda':
            import trainer.GDA as trainer
        elif args.trainer == 'coreset':
            import trainer.coreset as trainer
        elif args.trainer == 'ood':
            import trainer.ood as trainer
        return trainer.Trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer)