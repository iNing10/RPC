import os
import random
import numpy as np
import torch
from common.data import dataset_slug
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import DEFAULT_EXP_DIR, create_exp_dirs
from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params
from utils.scoring_utils import score_dataset
from utils.train_utils import calc_num_of_params


def main(argv=None):
    parser = init_parser()
    args = parser.parse_args(argv)

    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
    else:
        # Improve CUDA reproducibility for deterministic kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    args, model_args = init_sub_args(args)
    pretrained = vars(args).get('checkpoint', None)
    if pretrained and os.path.abspath(args.exp_dir) == os.path.abspath(DEFAULT_EXP_DIR):
        args.exp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "eval_outputs", "stgnf")
    run_dir = create_exp_dirs(args.exp_dir, dirmap=dataset_slug(args.dataset))
    args.exp_dir = run_dir
    args.ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)

    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained is not None))

    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    
    calc_num_of_params(model)
    trainer = Trainer(args, model, loader['train'], loader['test'],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    if pretrained:
        trainer.load_checkpoint(pretrained)
    else:
        trainer.train(dataset=dataset)
        dump_args(args, args.exp_dir)
    
    normality_scores = trainer.test()
    auc, scores = score_dataset(normality_scores, dataset["test"].metadata, args=args)

    # Logging and recording results
    print("\nEvaluation complete")
    print(f"  Dataset: {args.dataset}")
    print(f"  Frame AUC: {auc:.6f}")
    print(f"  Frames: {scores.shape[0]}")
    print()


if __name__ == '__main__':
    main()
