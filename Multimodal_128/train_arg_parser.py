import argparse
import json
import logging

logger = logging.getLogger(__name__)


def get_args_parser():
    # parser = argparse.ArgumentParser("Image dataset training", add_help=False)

    parser = argparse.ArgumentParser(
        "Image dataset training",
        add_help=False,
        fromfile_prefix_chars="@"  # enables config file with @filename
    )


    parser.add_argument(
        "--dataset_name",
        default="Laion_400M",
        help="name of the dataset to use",
    )

    parser.add_argument(
        "--image_model_name",
        default="facebook/dinov2-small",
        help="name of the image model to use",
    )

    parser.add_argument(
        "--text_model_name",
        default="distilbert-base-uncased",
        help="name of the text model to use",
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=921, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Optimizer parameters model  --------------------------------------------------
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="learning rate (model lr)",
    )

    parser.add_argument(
        "--lr_layer",
        type=float,
        default=0.0001,
        help="learning rate (Layer lr)",
    )

    parser.add_argument(
        "--lr_restriction_maps",
        type=float,
        default=0.0001,
        help="learning rate (restriction map lr)",
    )

    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.0001,
        help="learning rate (restriction map lr)",
    )
    # ----------------------------------------------------------------------------------------------------------------------

       # Optimizer parameters model  --------------------------------------------------
    parser.add_argument(
        "--lr_2",
        type=float,
        default=0.0001,
        help="learning rate (model lr)",
    )

    parser.add_argument(
        "--lr_layer_2",
        type=float,
        default=0.0001,
        help="learning rate (Layer lr)",
    )

    parser.add_argument(
        "--lr_restriction_maps_2",
        type=float,
        default=0.0001,
        help="learning rate (restriction map lr)",
    )

    parser.add_argument(
        "--lambda_reg_2",
        type=float,
        default=0.0001,
        help="learning rate (restriction map lr)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0001,
        help="variance regularization weight",
    )

    parser.add_argument(
        "--VAR_THRESHOLD",
        type=float,
        default=1e-5,
        help="variance regularization weight",
    )


    # -----------------------------------------------------------------------------------------------------------------------------------------





    parser.add_argument(
        "--optimizer_betas",
        nargs="+",
        type=float,
        default=[0.9, 0.95],
        help="learning rate (absolute lr",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Numerical stability for Adam optimizer",
    )

    parser.add_argument(
        "--weight_decay",
        nargs="+",
        type=float,
        default=1e-4,
        help="important: prevents overfitting",
    )



    parser.add_argument(
        "--decay_lr",
        action="store_true",
        help="Adds a linear decay to the lr during training.",
    )


    parser.add_argument(
        "--latent_dim",
        type = int,
        default = 128, 
        help = "Dimensionality of the restriction maps.",
    )

    parser.add_argument(
        "--patience",
        type = int,
        default = 50, 
        help = "Set patience .",
    )

  
    # parser.add_argument(
    #     "--use_ema",
    #     action="store_true",
    #     help="When evaluating, use the model Exponential Moving Average weights.",
    # )

    # # Dataset parameters
    # # CIFAR-10
    # parser.add_argument(
    #     "--dataset_1",
    #     default=list(MODEL_CONFIGS)[0],
    #     type=str,
    #     choices=list(MODEL_CONFIGS),
    #     help="Dataset to use.",
    # )


    # parser.add_argument(
    #     "--data_path",
    #     default="./data/image_generation",
    #     type=str,
    #     help="imagenet root folder with train, val and test subfolders",
    # )

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
 
  
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="start epoch (used when resumed from checkpoint)",
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="No training, only run evaluation"
    )
  ##
   
   
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Only run one batch of training and evaluation."
    )
   
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Load previous models to fine tune"
    )



    parser.add_argument(
        "--model1_path",
        default="./saved_model/best_model1.pth",
        help="path to load model previously saved"
    )

    parser.add_argument(
        "--model2_path",
        default="./saved_model/best_model2.pth",
        help="path to load model previously saved"
    )

    parser.add_argument(
        "--monitor",
        default="loss",
        choices=["loss", "acc"],
        help="Metric to monitor for saving checkpoints and early stopping (loss or acc).",
    )


    return parser