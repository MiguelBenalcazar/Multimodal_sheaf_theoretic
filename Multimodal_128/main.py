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

from training import distributed_mode, metrics
from training.evaluate import evaluate_model, reduce_tensor


from utils.save_load import save_checkpoint, load_checkpoint
from utils.logs import setup_run



from utils.dataset import dataset_class

from models.models import RestrictionMap

import random
import gc


from transformers import AutoImageProcessor, AutoModel,  DistilBertTokenizer, DistilBertModel
# Parallel process
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist



# Load model and processor dinov2

logger = logging.getLogger(__name__)

def reduce_scalar(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor


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
        
    logger.info(distributed_mode.get_rank()) 

    seed = args.seed + distributed_mode.get_rank()
     # Are you setting these identically?
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False

    # set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f"[Rank {args.rank}] Using device {device}")

   

    # device = torch.device(args.device)

    
    
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
    d_i_j = 384
    d_j_i = 768
    logger.info(f" Models dimension d_i d_j: {d_i_j} --> dimension restriction maps: {args.latent_dim} ")

    P12 = RestrictionMap(di_dj=d_i_j, dij=args.latent_dim, requires_grad=False)
    P21 = RestrictionMap(di_dj=d_j_i, dij=args.latent_dim, requires_grad=False)

    P12.Pij.to(device)
    P21.Pij.to(device)

  

    logger.info(f"P12 dimension: {P12.get_size()}, P21 dimension: {P21.get_size()}")

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )


    logger.info(f"alpha1: {args.lr_layer} alpha2: {args.lr_layer_2}")
    logger.info(f"restriction map: eta1: {args.lr_restriction_maps} eta2: {args.lr_restriction_maps_2}")
    logger.info(f"restriction map: Lambda1: {args.lambda_reg} Lambda2: {args.lambda_reg_2}" )
    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")



    theta_1 = torch.randn(args.batch_size, d_i_j,  device=device) # img
    theta_2 = torch.randn(args.batch_size, d_j_i,  device=device) # text

    best_model = float("inf")


    if args.finetune and distributed_mode.is_main_process():
        ckpt, theta_1, theta_2 = load_checkpoint("./saved_restrictions/best_restriction_maps.pth", P12=P12, P21=P21)
        best_model= ckpt['discrepancy']
        logger.info(f"Loaded saved_restrictions/best_restriction_maps.pth with best discrepancy: {best_model}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(P12.Pij.weight.data, src=0)
        torch.distributed.broadcast(P21.Pij.weight.data, src=0)


    mse_loss = nn.MSELoss()

    wait = 0
    stop_train = False

    if distributed_mode.is_main_process():
        logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        img_model.eval()
        text_model.eval()

        total_sheaf = torch.tensor(0.0, device=device)
        total_discrepancy_norm =torch.tensor(0.0, device=device)
        total_cosine_similarity = torch.tensor(0.0, device=device)
        total_var_theta1 = torch.tensor(0.0, device=device)
        total_var_theta2 = torch.tensor(0.0, device=device)
        

        if distributed_mode.is_main_process():
            logger.info(f"-------TRAINING Epoch: {epoch} -------")


        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)


        for i, (imgs, encodings) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            encodings = {key: val.to(device, non_blocking=True) for key, val in encodings.items()}

            with torch.no_grad():
                embeddings_imgs = img_model(**imgs).last_hidden_state[:, 0, :]
                embeddings_text = text_model(**encodings).last_hidden_state[:, 0, :]

            # detach to break graph and start new gradients
            theta1 = embeddings_imgs.detach().clone().requires_grad_(True)
            theta2 = embeddings_text.detach().clone().requires_grad_(True)

        
            

            theta_proj_discrepancy, sheaf_loss, discrepancy_norm, cos_sim, var_theta1, var_theta2 = metrics.evaluation(P12, P21, theta1, theta2)
        
   

            # # Assume P12.Pij is nn.Linear
            # theta_proj_discrepancy = P12.Pij(theta1) -  P21.Pij(theta2) 

            # # compute scalar for backward (sheaf regularization)
            # sheaf_loss = torch.mean(theta_proj_discrepancy ** 2)

            # loss = mse_loss(P12.Pij(theta1), P21.Pij(theta2))

            loss_sheaf_reg = sheaf_loss - args.alpha * (var_theta1 + var_theta2)

            # compute gradients w.r.t theta1, theta2 without building a full graph
            grads_theta1, grads_theta2 = torch.autograd.grad(loss_sheaf_reg, [theta1, theta2], retain_graph=False)

            # discrepancy_norm = torch.norm(theta_proj_discrepancy, p=2) / theta_proj_discrepancy.numel()
            # cos_sim = torch.nn.functional.cosine_similarity(P12.Pij(theta1), P21.Pij(theta2), dim=-1).mean()

           
            with torch.no_grad():
                theta1 -= args.lr_layer * (grads_theta1 + args.lambda_reg * (P12.Pij.weight.T @ theta_proj_discrepancy.T).T)
                theta2 -= args.lr_layer_2 * (grads_theta2 + args.lambda_reg_2 * (P21.Pij.weight.T @ -theta_proj_discrepancy.T).T)


                # Manually update restriction maps locally
                P12.step(P21.Pij, theta1, theta2, args.lr_restriction_maps, args.lambda_reg)
                P21.step(P12.Pij, theta2, theta1, args.lr_restriction_maps_2, args.lambda_reg_2)
              
                #  Optional: synchronize restriction maps across GPUs
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(P12.Pij.weight.data, op=torch.distributed.ReduceOp.SUM)
                    P12.Pij.weight.data /= torch.distributed.get_world_size()

                    torch.distributed.all_reduce(P21.Pij.weight.data, op=torch.distributed.ReduceOp.SUM)
                    P21.Pij.weight.data /= torch.distributed.get_world_size()

      
            total_sheaf += sheaf_loss.item()
            total_discrepancy_norm += discrepancy_norm.item()
            total_cosine_similarity += cos_sim.item()
            total_var_theta1 += var_theta1.item()
            total_var_theta2 += var_theta2.item()
            num_batches = i + 1

            # Logging every few batches
            if i % 10 == 0 and distributed_mode.is_main_process():
                logger.info(
                    f"[Epoch {epoch+1}, Batch {num_batches}] "
                    f"Avg Sheaf Loss: {total_sheaf/num_batches:.8e} "
                    f"Discrepancy Norm: {total_discrepancy_norm/num_batches:.8e} "
                    f"Cosine Sim: {total_cosine_similarity/num_batches:.6f} "
                    f"Avg Var Theta1: {total_var_theta1/num_batches:.8e} "
                    f"Avg Var Theta2: {total_var_theta2/num_batches:.8e}"
                )
               

            if args.test_run:
                if distributed_mode.is_main_process():
                    logger.info(f"[Epoch {epoch+1}, Batch {num_batches}/{args.epochs}] RestrictionMaps/Discrepancy: {total_sheaf}")
                    logger.info("Stop training because test run on")
                break


            del embeddings_imgs, embeddings_text, theta1, theta2, grads_theta1, grads_theta2
            gc.collect()
            torch.cuda.empty_cache()

        # Reduce across GPUs if distributed
        if args.distributed:
            total_sheaf = reduce_tensor(total_sheaf)
            total_discrepancy_norm = reduce_tensor(total_discrepancy_norm)
            total_cosine_similarity = reduce_tensor(total_cosine_similarity)

        
        num_batches = len(train_loader)
        avg_train_total_sheaf = total_sheaf / num_batches
        avg_train_discrepancy = total_discrepancy_norm / num_batches
        avg_train_cosine_sim = total_cosine_similarity / num_batches
        avg_var_theta1 = total_var_theta1 / num_batches
        avg_var_theta2 = total_var_theta2 / num_batches

        if distributed_mode.is_main_process():
            logger.info(
                f"[Epoch {epoch+1}/{args.epochs}] "
                f"Sheaf Loss: {avg_train_total_sheaf:.8e}, "
                f"Discrepancy Norm: {avg_train_discrepancy:.8e}, "
                f"Cosine Sim: {avg_train_cosine_sim:.8f}, "
                f"Avg Var Theta1: {avg_var_theta1:.8e}, "
                f"Avg Var Theta2: {avg_var_theta2:.8e}"
            )
            writer.add_scalar("Train/Loss_Sheaf", avg_train_total_sheaf, epoch)
            writer.add_scalar("Train/Discrepancy", avg_train_discrepancy, epoch)
            writer.add_scalar("Train/CosineSim", avg_train_cosine_sim, epoch)
            writer.add_scalar("Train/Var_Theta1", avg_var_theta1, epoch)
            writer.add_scalar("Train/Var_Theta2", avg_var_theta2, epoch)
        
      
 
        if distributed_mode.is_main_process():
            logger.info(f"-------TESTING Epoch: {epoch + 1} -------")
        
        with torch.no_grad():

            # total_sheaf = evaluate_model(img_model, text_model, test_loader, P12, P21, epoch,  device, args)
            avg_train_total_sheaf, avg_train_discrepancy, avg_train_cosine_sim, avg_var_theta1, avg_var_theta2 = \
                evaluate_model(img_model, text_model, test_loader, P12, P21, device, args)

            if distributed_mode.is_main_process():
                logger.info(
                    f"[Epoch {epoch+1}/{args.epochs}] "
                    f"Sheaf Loss: {avg_train_total_sheaf:.8e}, "
                    f"Discrepancy Norm: {avg_train_discrepancy:.8e}, "
                    f"Cosine Sim: {avg_train_cosine_sim:.8f}, "
                    f"Avg Var Theta1: {avg_var_theta1:.8e}, "
                    f"Avg Var Theta2: {avg_var_theta2:.8e}"
                )
                writer.add_scalar("Test/Loss_Sheaf", avg_train_total_sheaf, epoch)
                writer.add_scalar("Test/Discrepancy", avg_train_discrepancy, epoch)
                writer.add_scalar("Test/CosineSim", avg_train_cosine_sim, epoch)
                writer.add_scalar("Test/Var_Theta1", avg_var_theta1, epoch)
                writer.add_scalar("Test/Var_Theta2", avg_var_theta2, epoch)
            
            
            # avg_train_total_sheaf, avg_train_discrepancy, avg_train_cosine_sim, avg_var_theta1, avg_var_theta2 =  evaluate_model(img_model, text_model, test_loader, P12, P21,  device, args)

            # if distributed_mode.is_main_process():
            #     logger.info(f"[Epoch {epoch+1}/{args.epochs}] "
            #         f"Sheaf Loss: {avg_train_total_sheaf:.8e}, "
            #         f"Discrepancy Norm: {avg_train_discrepancy:.8e}, "
            #         f"Cosine Sim: {avg_train_cosine_sim:.8f}")
            #     writer.add_scalar("Test/Loss_Sheaf", avg_train_total_sheaf, epoch)
            #     writer.add_scalar("Test/Discrepancy", avg_train_discrepancy, epoch)
            #     writer.add_scalar("Test/CosineSim", avg_train_cosine_sim, epoch)
            
            # avg_train_total_sheaf = total_sheaf / len(test_loader)
            # if distributed_mode.is_main_process():
            #     logger.info(f"[Epoch {epoch+1}/{args.epochs}] RestrictionMaps/Discrepancy: {avg_train_total_sheaf}")
            #     writer.add_scalar("Test/Loss_Sheaf", avg_train_total_sheaf, epoch)


            if not stop_train and not args.test_run:
                metric_model = avg_train_total_sheaf
                best_metric_model = best_model
                # is_better = metric_model < best_metric_model

                # Check if embeddings are collapsed
                embeddings_collapsed = (avg_var_theta1 < args.VAR_THRESHOLD) or (avg_var_theta2 < args.VAR_THRESHOLD)
                is_better = (metric_model < best_metric_model) and not embeddings_collapsed

                if embeddings_collapsed and distributed_mode.is_main_process():
                    logger.info("Warning: Embeddings have collapsed. Not saving the model.")
                    stop_train = True

                if distributed_mode.is_main_process() and is_better:
                    best_model = metric_model
                    wait = 0
                    save_checkpoint(
                        P12=P12, 
                        P21=P21,
                        theta_1=theta_1,
                        theta_2=theta_2,
                        batch=epoch+1, 
                        monitor_value= best_metric_model, 
                        keep_last=2,
                        is_best=True 
                    )
                    if distributed_mode.is_main_process():
                        logger.info(f"Discrepancy: {metric_model}")
                else:
                    wait += 1
                    if distributed_mode.is_main_process():
                        logger.info(f"Model discrepancy no improvement ({wait}/{args.patience}")

                if wait >= args.patience:
                    logger.info("Stopping training (patience reached).")
                    stop_train = True

        if args.test_run and distributed_mode.is_main_process():
            if distributed_mode.is_main_process():
                logger.info("Stop testing because test run on")
            break

        if stop_train and distributed_mode.is_main_process():
            if distributed_mode.is_main_process():
                logger.info("Ending training loop.")
            break

    
    if distributed_mode.is_main_process():
        writer.close()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Training time {total_time_str}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    gc.collect()




if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)