import time
import torch
import os
import argument_parser

opt = argument_parser.parser()
from trainer import Trainer

def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = Trainer(opt)
    trainer.build_dataset()
    trainer.build_network()
    trainer.build_losses()
    trainer.build_optimizer()
    trainer.start_train_time = time.time()

    for epoch in range(opt.nepoch):
        # if opt.training:
        #     trainer.train_epoch()
        # if opt.test and trainer.epoch >= opt.tune_AE:
        with torch.no_grad():
            # if not opt.voting:
            trainer.test_epoch_timing()
            # else:
            #     trainer.test_voting()
        # if opt.training:
        #     trainer.save_network()
        trainer.increment_epoch()

    if opt.test_final:
        print(trainer.result.mean())

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    main()
