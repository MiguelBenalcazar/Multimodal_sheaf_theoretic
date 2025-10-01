import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F
# from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser

from training import distributed_mode
from training.evaluate import evaluate_model, evaluate_model_once


from utils.save_load import save_checkpoint, load_checkpoint
from utils.logs import setup_run



from utils.dataset import dataset_class

from models.models import RestrictionMap

import random
import gc


from transformers import AutoImageProcessor, AutoModel,  DistilBertTokenizer, DistilBertModel

# Load model and processor dinov2

logger = logging.getLogger(__name__)

def main(args):

    logger, writer, run_dir = setup_run(args)
    logger.info(f"Everything is being logged in {run_dir}")
    logger.info("Starting training loop...")


    curretn_folder = os.getcwd()
    dataset_base = os.path.abspath(os.path.join(curretn_folder,"..", "..","data","COCO"))

    file_train = os.path.join(dataset_base,"joint","coco_train_captions.csv")
    file_test = os.path.join(dataset_base,"joint","coco_test_captions.csv")

    if not os.path.exists(file_train) or not os.path.exists(file_test):
        raise ("There is no train or test file")
    
    logger.info(f"Train file: {file_train}")
    logger.info(f"Test file: {file_test}")
    
    distributed_mode.init_distributed_mode(args)



    seed = args.seed + distributed_mode.get_rank()
     # Are you setting these identically?
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

    device = torch.device(args.device)

    
    
    img_processor = AutoImageProcessor.from_pretrained(args.image_model_name, use_fast = False)
    img_model = AutoModel.from_pretrained(args.image_model_name).to(device)

    text_tokenizer = DistilBertTokenizer.from_pretrained(args.text_model_name)
    text_model = DistilBertModel.from_pretrained(args.text_model_name).to(device)

    # freezing all parameters in the model 
    for param in img_model.parameters():
        param.requires_grad = False

    for param in text_model.parameters():
        param.requires_grad = False



    logger.info(f"Initializing Dataset: {args.dataset_name}")
    train_loader, test_loader = dataset_class(
        args=args, 
        img_processor= img_processor, 
        tokenizer = text_tokenizer,
        file_train=file_train, 
        file_test=file_test
        )
    
    

    ''' 
        Creating restricion maps Pij
    '''
    d_i_j = 768
    logger.info(f" Models dimension d_i d_j: {d_i_j} --> dimension restriction maps: {args.latent_dim} ")

    P12 = RestrictionMap(di_dj=d_i_j, dij=args.latent_dim)
    P21 = RestrictionMap(di_dj=d_i_j, dij=args.latent_dim)

    P12.Pij.to(device)
    P21.Pij.to(device)


    logger.info(f"P12 dimension: {P12.get_size()}, P21 dimension: {P21.get_size()}")

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )

    logger.info(f"model1: {args.lr} model1: {args.lr_2}")
    logger.info(f"alpha1: {args.lr_layer} alpha2: {args.lr_layer_2}")
    logger.info(f"restriction map: eta1: {args.lr_restriction_maps} eta2: {args.lr_restriction_maps_2}")
    logger.info(f"restriction map: Lambda1: {args.lambda_reg} Lambda2: {args.lambda_reg_2}" )
    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")



    theta_1 = torch.randn(args.batch_size, d_i_j,  device=device) # img
    theta_2 = torch.randn(args.batch_size, d_i_j,  device=device) # text

    best_model = float("inf")


    if args.finetune:
        ckpt, theta_1, theta_2 = load_checkpoint("./saved_restrictions/best_restriction_maps.pth", P12=P12, P21=P21)
        best_model= ckpt['discrepancy']
        logger.info(f"Loaded saved_restrictions/best_restriction_maps.pth with best discrepancy: {best_model}")




    wait = 0
    stop_train = False

    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        img_model.eval()
        text_model.eval()

        total_sheaf = 0.0

        logger.info(f"-------TRAINING Epoch: {epoch} -------")

        img_model.eval()
        text_model.eval()

        for i, (imgs, encodings) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            encodings = {key: val.to(device, non_blocking=True) for key, val in encodings.items()}

            with torch.no_grad():
                embeddings_imgs = img_model(**imgs).last_hidden_state[:, 0, :]
                embeddings_text = text_model(**encodings).last_hidden_state[:, 0, :]

            # detach to break graph and start new gradients
            theta1 = embeddings_imgs.detach().clone().requires_grad_(True)
            theta2 = embeddings_text.detach().clone().requires_grad_(True)
            

           

            # Assume P12.Pij is nn.Linear
            theta_proj_discrepancy = P12.Pij(theta1) -  P21.Pij(theta2) 

            # compute scalar for backward (sheaf regularization)
            # sheaf_loss = (theta_proj_discrepancy ** 2).sum()  # or simply theta_proj_discrepancy.sum() if you want
            # sheaf_loss = torch.norm(theta_proj_discrepancy, p=2) ** 2 
            sheaf_loss = torch.mean(theta_proj_discrepancy ** 2)

            # compute gradients w.r.t theta1, theta2 without building a full graph
            grads_theta1, grads_theta2 = torch.autograd.grad(sheaf_loss, [theta1, theta2], retain_graph=False)


        
            with torch.no_grad():
                theta1 -= args.lr_layer * (grads_theta1 + args.lambda_reg * (P12.Pij.weight.T @ theta_proj_discrepancy.T).T)
                theta2 -= args.lr_layer_2 * (grads_theta2 + args.lambda_reg_2 * (P21.Pij.weight.T @ -theta_proj_discrepancy.T).T)

                P12.step(P21.Pij, theta1, theta2, args.lr_restriction_maps, args.lambda_reg)
                P21.step(P12.Pij, theta2, theta1, args.lr_restriction_maps_2, args.lambda_reg_2)

            total_sheaf += sheaf_loss.item()


            # Logging every few batches
            if i % 10 == 0:
                avg_loss = total_sheaf / (i + 1)
                logger.info(f"[Epoch {epoch+1}, Batch {i+1}] Avg Sheaf Loss: {avg_loss}")


            if args.test_run:
                logger.info(f"[Epoch {epoch+1}, Batch {i+1}/{args.epochs}] RestrictionMaps/Discrepancy: {total_sheaf}")
                logger.info("Stop training because test run on")
                break


            del embeddings_imgs, embeddings_text, theta1, theta2, grads_theta1, grads_theta2
            gc.collect()
            torch.cuda.empty_cache()
            

        avg_train_total_sheaf = total_sheaf / len(test_loader)
        logger.info(f"[Epoch {epoch+1}/{args.epochs}] RestrictionMaps/Discrepancy: {avg_train_total_sheaf}")
        writer.add_scalar("Train/Loss_Sheaf", avg_train_total_sheaf, epoch)
            
            
        

        with torch.no_grad():

            total_sheaf = evaluate_model(img_model, text_model, test_loader, P12, P21, epoch,  device, args)

            
            avg_train_total_sheaf = total_sheaf / len(test_loader)
            logger.info(f"[Epoch {epoch+1}/{args.epochs}] RestrictionMaps/Discrepancy: {avg_train_total_sheaf}")
            writer.add_scalar("Test/Loss_Sheaf", avg_train_total_sheaf, epoch)


            if not stop_train:
                metric_model = avg_train_total_sheaf
                best_metric_model = best_model
                is_better = metric_model < best_metric_model

                if is_better:
                    best_metric_model = metric_model
                    wait = 0
                    save_checkpoint(
                        P12=P12, 
                        P21=P21,
                        theta_1=theta_1,
                        theta_2=theta_2,
                        batch=epoch, 
                        monitor_value= best_metric_model, 
                        keep_last=2,
                        is_best=True 
                    )
                    logger.info(f"Discrepancy: {metric_model}")
                else:
                    wait += 1
                    logger.info(f"Model discrepancy no improvement ({wait}/{args.patience}")

                if wait >= args.patience:
                    logger.info("Stopping training (patience reached).")
                    stop_train = True

        if args.test_run:
            logger.info("Stop testing because test run on")
            break

        if stop_train:
            logger.info("Ending training loop.")
            break

    writer.close()

            

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)