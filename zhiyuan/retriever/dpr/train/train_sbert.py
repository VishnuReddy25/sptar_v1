# '''
# This examples show how to train a basic Bi-Encoder for any BEIR dataset without any mined hard negatives or triplets.

# The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
# These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

# For training, we use MultipleNegativesRankingLoss. There, we pass pairs in the format:
# (query, positive_passage). Other positive passages within a single batch becomes negatives given the pos passage.

# We do not mine hard negatives or train triplets in this example.

# Running this script:
# python train_sbert.py
# '''

# from sentence_transformers import losses, models, SentenceTransformer, evaluation
# from beir import util, LoggingHandler
# from beir.datasets.data_loader import GenericDataLoader
# from beir.retrieval.train import TrainRetriever
# import pathlib, os, gzip
# import logging
# import sys
# import random
# import argparse
# from os.path import join
# import torch
# import torch.nn as nn
# from torch.nn import functional as F

# #### Just some code to print debug information to stdout
# cwd = os.getcwd()
# if join(cwd, "zhiyuan") not in sys.path:
#     sys.path.append(join(cwd, "zhiyuan"))
#     sys.path.append(join(cwd, "xuyang"))
# from weak_data_loader import WeakDataLoader
# data_dir = join(cwd, "zhiyuan", "datasets")
# raw_dir = join(data_dir, "raw")
# weak_dir = join(data_dir, "weak")
# beir_dir = join(raw_dir, "beir")
# xuyang_dir = join(cwd, "xuyang", "data")

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_name', required=False, default="scifact", type=str)
# parser.add_argument('--num_epochs', required=False, default=2, type=int)
# parser.add_argument('--train_num', required=False, default=100, type=int)
# parser.add_argument('--weak_num', required=False, default="5000", type=str)
# parser.add_argument('--product', required=False, default="cos_sim", type=str)
# parser.add_argument('--exp_name', required=False, default="no_aug", type=str)
# parser.add_argument('--model_name', required=False, default="bert-base-uncased", type=str)
# parser.add_argument('--version', required=False, default="v1", type=str)
# parser.add_argument(
#     "--loss_fn",
#     type=str,
#     default="listwise_softmax",
#     help="Loss function to use. Options: listwise_softmax, pairwise_hinge",
# )
# args = parser.parse_args()

# #### Provide model save path
# model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", args.exp_name, str(args.train_num), args.model_name + '-' + args.version + '-' + args.dataset_name)
# os.makedirs(model_save_path, exist_ok=True)

# #### Just some code to print debug information to stdout
# fh = logging.FileHandler(join(model_save_path, "log.txt"))
# ch = logging.StreamHandler(sys.stdout)
# logging.basicConfig(format='%(asctime)s - %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     level=logging.INFO,
#                     handlers=[fh, ch])
# ####

# #### Data Loading
# if args.exp_name == "no_aug":
#     corpus, queries, qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv")).load_custom()
# else:
#     weak_query_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_queries_{args.train_num}_{args.exp_name}.jsonl")
#     weak_qrels_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_train_{args.train_num}_{args.exp_name}.tsv")
#     corpus, queries, qrels = WeakDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv"), weak_query_file=weak_query_file, weak_qrels_file=weak_qrels_file).load_weak_custom()

# dev_corpus, dev_queries, dev_qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(beir_dir, args.dataset_name, "qrels", "dev.tsv")).load_custom()

# ####
# train_retriever = TrainRetriever(model=None, batch_size=32)
# train_samples = train_retriever.load_train(corpus, queries, qrels)
# logging.info("Loaded {} training pairs.".format(len(train_samples)))
# logging.info("dev set contains {} documents and {} queries".format(len(dev_corpus), len(dev_queries)))

# #### Model Setup
# word_embedding_model = models.Transformer(args.model_name, max_seq_length=350)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# #### Loss Function Setup
# if args.loss_fn == "listwise_softmax":
#     # The standard MultipleNegativesRankingLoss, which is a listwise softmax loss.
#     train_loss = losses.MultipleNegativesRankingLoss(model=model)
# elif args.loss_fn == "pairwise_hinge":
#     # Fixed Pairwise Hinge Loss to avoid in-place operation error
#     # A custom pairwise hinge loss.
#     # We need to define a custom loss function for this.
#     # sentence-transformers doesn't have a built-in pairwise hinge loss that works directly with MultipleNegativesRankingLoss data format.
#     # So we will implement it within the training loop logic if needed, or use a different data loader.
#     # For simplicity, let's stick with what's available or requires minimal changes.
#     # MultipleNegativesRankingLoss is the most common and effective for this setup.
#     # Let's define a custom loss that can be used with the existing data loader.
#     class PairwiseHingeLoss(nn.Module):
#         def __init__(self, model, margin=1.0):
#             super(PairwiseHingeLoss, self).__init__()
#             self.model = model
#             self.margin = margin

#         def forward(self, sentence_features, labels):
#             reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
#             embeddings_a = reps[0]
#             embeddings_b = torch.cat(reps[1:], dim=0)
#             embeddings_b = torch.cat(reps[1:])
            
#             # Assuming the first one is positive and the rest are negatives
#             query_embeddings = embeddings_a
#             doc_embeddings = embeddings_b
            
#             # In-batch negatives
#             scores = torch.matmul(query_embeddings, doc_embeddings.transpose(0, 1))
            
#             pos_scores = scores.diag()
            
#             # Use clamp instead of in-place relu to avoid the RuntimeError
#             # Pairwise Hinge Loss calculation
#             margin = self.margin
#             # Use clamp instead of in-place relu to avoid potential RuntimeError
#             losses = torch.clamp(margin - (pos_scores.unsqueeze(1) - scores), min=0.0)
            
#             # Zero out the diagonal safely
#             mask = torch.eye(losses.size(0), device=losses.device)
#             losses = losses * (1 - mask)
#             # Zero out the diagonal (query vs. positive doc) safely
#             mask = torch.eye(losses.size(0), device=losses.device, dtype=torch.bool)
#             losses.masked_fill_(mask, 0)
            
#             loss = losses.mean()
#             return loss

#     train_loss = PairwiseHingeLoss(model=model)
# else:
#     raise ValueError(f"Unsupported loss function: {args.loss_fn}. Choose 'listwise_softmax' or 'pairwise_hinge'.")


# #### Evaluator Setup
# dev_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, 
#                                                          dev_corpus, 
#                                                          dev_qrels, 
#                                                          name=args.dataset_name, 
#                                                          main_score_function=args.product)

# #### Create DataLoader from train samples
# from torch.utils.data import DataLoader
# train_dataloader = DataLoader(train_samples, batch_size=32, shuffle=True)

# #### Clear GPU cache and prepare for training
# import gc
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     gc.collect()
#     logging.info(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# #### Train the model
# logging.info("Starting to Train...")
# model.fit(train_objectives=[(train_dataloader, train_loss)],
#           evaluator=dev_evaluator,
#           epochs=args.num_epochs,
#           output_path=model_save_path,
#           warmup_steps=100,
#           use_amp=False,
#           checkpoint_path=model_save_path,
#           checkpoint_save_steps=len(train_dataloader),
#           evaluation_steps=1000,
#           save_best_model=True)

# #### Save the final model to the main path
# logging.info("Training complete. Saving final model to {}...".format(model_save_path))
# # Ensure the output directory exists
# os.makedirs(model_save_path, exist_ok=True)
# model.save(model_save_path)

# # Verify that required files exist
# config_file = os.path.join(model_save_path, "config.json")
# if os.path.exists(config_file):
#     logging.info("Model config.json saved successfully.")
# else:
#     logging.warning("config.json not found in model directory. This may cause issues during evaluation.")
#     # Try to create a minimal config.json if it doesn't exist
#     try:
#         # Get the config from the underlying transformer model
#         if hasattr(model._modules['0'], 'auto_model') and hasattr(model._modules['0'].auto_model, 'config'):
#             config = model._modules['0'].auto_model.config
#             config.save_pretrained(model_save_path)
#             logging.info("Created config.json from model configuration.")
#     except Exception as e:
#         logging.warning(f"Could not create config.json: {e}")

# logging.info("Final model saved successfully.")


# hybrid  



# """
# Train a Bi-Encoder on a BEIR dataset using a combination of:
# 1) Listwise loss (MultipleNegativesRankingLoss)
# 2) Pairwise hard-negative hinge loss
# """
# import os
# from os.path import join
# import sys
# cwd = os.getcwd()
# if join(cwd, "zhiyuan") not in sys.path:
#     sys.path.append(join(cwd, "zhiyuan"))
#     sys.path.append(join(cwd, "xuyang"))
# from weak_data_loader import WeakDataLoader
# from sentence_transformers import losses, models, SentenceTransformer, evaluation
# from beir.datasets.data_loader import GenericDataLoader
# from beir.retrieval.train import TrainRetriever
# from weak_data_loader import WeakDataLoader
# import pathlib, os, logging, argparse, sys, torch
# from torch import nn
# from torch.utils.data import DataLoader

# # Paths


# cwd = os.getcwd()
# if os.path.join(cwd, "zhiyuan") not in sys.path:
#     sys.path.append(os.path.join(cwd, "zhiyuan"))
#     sys.path.append(os.path.join(cwd, "xuyang"))

# data_dir = os.path.join(cwd, "zhiyuan", "datasets")
# raw_dir = os.path.join(data_dir, "raw")
# weak_dir = os.path.join(data_dir, "weak")
# beir_dir = os.path.join(raw_dir, "beir")
# xuyang_dir = os.path.join(cwd, "xuyang", "data")

# # Argument parser
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_name', type=str, default="scifact")
# parser.add_argument('--num_epochs', type=int, default=2)
# parser.add_argument('--train_num', type=int, default=100)
# parser.add_argument('--weak_num', type=str, default="5000")
# parser.add_argument('--exp_name', type=str, default="no_aug")
# parser.add_argument('--model_name', type=str, default="bert-base-uncased")
# parser.add_argument('--version', type=str, default="v1")
# parser.add_argument('--product', type=str, default="cos_sim")
# parser.add_argument('--alpha', type=float, default=0.5, help="Weight for pairwise loss")
# args = parser.parse_args()

# # Model save path
# model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(),
#                                "output", args.exp_name, str(args.train_num),
#                                f"{args.model_name}-{args.version}-{args.dataset_name}")
# os.makedirs(model_save_path, exist_ok=True)

# # Logging
# fh = logging.FileHandler(os.path.join(model_save_path, "log.txt"))
# ch = logging.StreamHandler(sys.stdout)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[fh, ch])

# # Data loading
# if args.exp_name == "no_aug":
#     corpus, queries, qrels = GenericDataLoader(
#         corpus_file=os.path.join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
#         query_file=os.path.join(beir_dir, args.dataset_name, "queries.jsonl"),
#         qrels_file=os.path.join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv")
#     ).load_custom()
# else:
#     weak_query_file = os.path.join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num,
#                                    f"weak_queries_{args.train_num}_{args.exp_name}.jsonl")
#     weak_qrels_file = os.path.join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num,
#                                    f"weak_train_{args.train_num}_{args.exp_name}.tsv")
#     corpus, queries, qrels = WeakDataLoader(
#         corpus_file=os.path.join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
#         query_file=os.path.join(beir_dir, args.dataset_name, "queries.jsonl"),
#         qrels_file=os.path.join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv"),
#         weak_query_file=weak_query_file,
#         weak_qrels_file=weak_qrels_file
#     ).load_weak_custom()

# # Dev set
# dev_corpus, dev_queries, dev_qrels = GenericDataLoader(
#     corpus_file=os.path.join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
#     query_file=os.path.join(beir_dir, args.dataset_name, "queries.jsonl"),
#     qrels_file=os.path.join(beir_dir, args.dataset_name, "qrels", "dev.tsv")
# ).load_custom()

# # TrainRetriever & samples
# train_retriever = TrainRetriever(model=None, batch_size=32)
# train_samples = train_retriever.load_train(corpus, queries, qrels)
# logging.info(f"Loaded {len(train_samples)} training pairs.")
# logging.info(f"Dev set: {len(dev_corpus)} docs, {len(dev_queries)} queries")

# # Model setup
# word_embedding_model = models.Transformer(args.model_name, max_seq_length=350)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# # Listwise loss
# listwise_loss = losses.MultipleNegativesRankingLoss(model=model)

# # Pairwise hard-negative loss
# class PairwiseHingeHardNegLoss(nn.Module):
#     def __init__(self, model, margin=1.0):
#         super().__init__()
#         self.model = model
#         self.margin = margin

#     def forward(self, sentence_features, labels):
#         reps = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
#         query_embeddings = reps[0]
#         doc_embeddings = torch.cat(reps[1:], dim=0)
#         scores = torch.matmul(query_embeddings, doc_embeddings.T)
#         pos_scores = scores.diag()
#         losses = torch.clamp(self.margin - (pos_scores.unsqueeze(1) - scores), min=0.0)
#         mask = torch.eye(losses.size(0), device=losses.device)
#         losses = losses * (1 - mask)
#         losses.masked_fill_(mask.bool(), 0)
#         return losses.mean()

# pairwise_loss = PairwiseHingeHardNegLoss(model=model)

# # Combined loss
# class CombinedLoss(nn.Module):
#     def __init__(self, listwise_loss, pairwise_loss, alpha=0.5):
#         super().__init__()
#         self.listwise_loss = listwise_loss
#         self.pairwise_loss = pairwise_loss
#         self.alpha = alpha

#     def forward(self, sentence_features, labels):
#         lw = self.listwise_loss(sentence_features, labels)
#         pw = self.pairwise_loss(sentence_features, labels)
#         return lw + self.alpha * pw

# train_loss = CombinedLoss(listwise_loss, pairwise_loss, alpha=args.alpha)

# # Evaluator
# dev_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, dev_corpus, dev_qrels,
#                                                          name=args.dataset_name,
#                                                          main_score_function=args.product)

# # Dataloader
# train_dataloader = DataLoader(train_samples, batch_size=32, shuffle=True)

# # GPU cleanup
# import gc
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     gc.collect()
#     logging.info(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# # Training
# logging.info("Starting training...")
# model.fit(train_objectives=[(train_dataloader, train_loss)],
#           evaluator=dev_evaluator,
#           epochs=args.num_epochs,
#           output_path=model_save_path,
#           warmup_steps=100,
#           use_amp=False,
#           checkpoint_path=model_save_path,
#           checkpoint_save_steps=len(train_dataloader),
#           evaluation_steps=1000,
#           save_best_model=True)

# logging.info("Training complete. Saving final model...")
# model.save(model_save_path)
# logging.info("Final model saved successfully.")


#combined given by claude 


# """
# Complete Bi-Encoder Training Script with Optimized Combined Loss
# Includes: Hard negative mining, combined loss, extensive arguments
# """

# import os
# from os.path import join
# import sys
# import pathlib
# import logging
# import argparse
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# import gc

# # Path setup
# cwd = os.getcwd()
# if join(cwd, "zhiyuan") not in sys.path:
#     sys.path.append(join(cwd, "zhiyuan"))
#     sys.path.append(join(cwd, "xuyang"))

# from weak_data_loader import WeakDataLoader
# from sentence_transformers import losses, models, SentenceTransformer, evaluation, InputExample
# from beir.datasets.data_loader import GenericDataLoader
# from beir.retrieval.train import TrainRetriever
# from beir.retrieval.search.dense import DenseRetrievalExactSearch
# from beir.retrieval.evaluation import EvaluateRetrieval

# # Directory paths
# data_dir = join(cwd, "zhiyuan", "datasets")
# raw_dir = join(data_dir, "raw")
# weak_dir = join(data_dir, "weak")
# beir_dir = join(raw_dir, "beir")
# xuyang_dir = join(cwd, "xuyang", "data")


# # ==================== Argument Parser ====================
# parser = argparse.ArgumentParser(description="Train Bi-Encoder with Combined Loss")

# # Dataset arguments
# parser.add_argument('--dataset_name', type=str, default="scifact", 
#                     help="BEIR dataset name")
# parser.add_argument('--train_num', type=int, default=100, 
#                     help="Number of training samples")
# parser.add_argument('--weak_num', type=str, default="5000", 
#                     help="Weak supervision sample count")
# parser.add_argument('--exp_name', type=str, default="no_aug", 
#                     help="Experiment name (no_aug or augmentation type)")

# # Model arguments
# parser.add_argument('--model_name', type=str, default="bert-base-uncased",
#                     help="Pretrained model name")
# parser.add_argument('--max_seq_length', type=int, default=350,
#                     help="Maximum sequence length")
# parser.add_argument('--version', type=str, default="v1",
#                     help="Model version identifier")

# # Training arguments
# parser.add_argument('--num_epochs', type=int, default=10,
#                     help="Number of training epochs")
# parser.add_argument('--batch_size', type=int, default=64,
#                     help="Training batch size")
# parser.add_argument('--learning_rate', type=float, default=2e-5,
#                     help="Learning rate")
# parser.add_argument('--warmup_steps', type=int, default=1000,
#                     help="Number of warmup steps")
# parser.add_argument('--use_amp', action='store_true',
#                     help="Use automatic mixed precision")
# parser.add_argument('--evaluation_steps', type=int, default=500,
#                     help="Evaluate every N steps")

# # Loss arguments
# parser.add_argument('--loss_type', type=str, default="combined",
#                     choices=["listwise", "pairwise", "combined"],
#                     help="Loss function type")
# parser.add_argument('--alpha', type=float, default=0.5,
#                     help="Weight for pairwise loss in combined loss")
# parser.add_argument('--margin', type=float, default=1.0,
#                     help="Margin for pairwise hinge loss")
# parser.add_argument('--scale', type=float, default=20.0,
#                     help="Temperature scaling for listwise loss")

# # Hard negative mining arguments
# parser.add_argument('--use_hard_negatives', action='store_true',
#                     help="Use hard negative mining")
# parser.add_argument('--hard_neg_top_k', type=int, default=100,
#                     help="Top-K documents to retrieve for hard negative mining")
# parser.add_argument('--num_hard_negatives', type=int, default=5,
#                     help="Number of hard negatives per query")
# parser.add_argument('--mine_hard_negs_before_training', action='store_true',
#                     help="Mine hard negatives before training starts")

# # Evaluation arguments
# parser.add_argument('--product', type=str, default="cos_sim",
#                     choices=["cos_sim", "dot"],
#                     help="Similarity function for evaluation")

# # System arguments
# parser.add_argument('--seed', type=int, default=42,
#                     help="Random seed")

# args = parser.parse_args()


# # ==================== Set Random Seeds ====================
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(args.seed)


# # ==================== Model Save Path ====================
# model_save_path = join(pathlib.Path(__file__).parent.absolute(),
#                        "output", args.exp_name, str(args.train_num),
#                        f"{args.model_name}-{args.version}-{args.dataset_name}-{args.loss_type}")
# os.makedirs(model_save_path, exist_ok=True)


# # ==================== Logging Setup ====================
# fh = logging.FileHandler(join(model_save_path, "log.txt"))
# ch = logging.StreamHandler(sys.stdout)
# logging.basicConfig(level=logging.INFO, 
#                    format='%(asctime)s - %(message)s',
#                    datefmt='%Y-%m-%d %H:%M:%S',
#                    handlers=[fh, ch])

# logging.info("=" * 80)
# logging.info("Training Configuration")
# logging.info("=" * 80)
# for arg, value in vars(args).items():
#     logging.info(f"{arg}: {value}")
# logging.info("=" * 80)


# # ==================== Optimized Combined Loss ====================
# class OptimizedCombinedLoss(nn.Module):
#     def __init__(self, model, margin=1.0, alpha=0.5, scale=20.0):
#         """
#         Combined loss that computes embeddings once and applies both losses.
        
#         Args:
#             model: SentenceTransformer model
#             margin: Margin for pairwise hinge loss
#             alpha: Weight for pairwise loss (final_loss = listwise + alpha * pairwise)
#             scale: Temperature scaling for listwise softmax loss
#         """
#         super().__init__()
#         self.model = model
#         self.margin = margin
#         self.alpha = alpha
#         self.scale = scale
#         self.cross_entropy_loss = nn.CrossEntropyLoss()
#         self.step_count = 0

#     def forward(self, sentence_features, labels):
#         # Compute embeddings ONCE for all inputs
#         reps = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
#         query_embeddings = reps[0]  # [batch_size, embedding_dim]
#         doc_embeddings = torch.cat(reps[1:], dim=0)  # [batch_size, embedding_dim]
        
#         # Compute similarity scores (used by both losses)
#         scores = torch.matmul(query_embeddings, doc_embeddings.T)  # [batch_size, batch_size]
        
#         # === Listwise Softmax Loss (MultipleNegativesRankingLoss equivalent) ===
#         scores_scaled = scores * self.scale
#         labels_listwise = torch.arange(scores.size(0), device=scores.device)
#         listwise_loss = self.cross_entropy_loss(scores_scaled, labels_listwise)
        
#         # === Pairwise Hard-Negative Hinge Loss ===
#         pos_scores = scores.diag().unsqueeze(1)  # [batch_size, 1]
#         pairwise_losses = torch.clamp(self.margin - (pos_scores - scores), min=0.0)
        
#         # Zero out diagonal (don't penalize positive pairs)
#         mask = torch.eye(scores.size(0), device=scores.device, dtype=torch.bool)
#         pairwise_losses.masked_fill_(mask, 0)
#         pairwise_loss = pairwise_losses.mean()
        
#         # === Combined Loss ===
#         total_loss = listwise_loss + self.alpha * pairwise_loss
        
#         # Periodic logging
#         self.step_count += 1
#         if self.step_count % 100 == 0:
#             logging.info(f"Step {self.step_count} - Listwise: {listwise_loss.item():.4f}, "
#                         f"Pairwise: {pairwise_loss.item():.4f}, "
#                         f"Total: {total_loss.item():.4f}")
        
#         return total_loss


# class PairwiseHingeLoss(nn.Module):
#     """Standalone pairwise hinge loss"""
#     def __init__(self, model, margin=1.0):
#         super().__init__()
#         self.model = model
#         self.margin = margin

#     def forward(self, sentence_features, labels):
#         reps = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
#         query_embeddings = reps[0]
#         doc_embeddings = torch.cat(reps[1:], dim=0)
#         scores = torch.matmul(query_embeddings, doc_embeddings.T)
#         pos_scores = scores.diag().unsqueeze(1)
#         losses = torch.clamp(self.margin - (pos_scores - scores), min=0.0)
#         mask = torch.eye(scores.size(0), device=scores.device, dtype=torch.bool)
#         losses.masked_fill_(mask, 0)
#         return losses.mean()


# # ==================== Hard Negative Mining ====================
# def mine_hard_negatives(model, corpus, queries, qrels, top_k=100, num_hard_negs=5):
#     """
#     Mine hard negatives using the current model.
    
#     Returns:
#         dict: {query_id: [list of hard negative doc_ids]}
#     """
#     logging.info("Mining hard negatives...")
    
#     model.eval()
#     retriever = DenseRetrievalExactSearch(model, batch_size=128)
#     results = retriever.search(corpus, queries, top_k=top_k, score_function=args.product)
    
#     hard_negatives = {}
#     for qid in queries:
#         if qid in results and qid in qrels:
#             positives = set(qrels[qid].keys())
#             # Get top-k documents that are NOT positives
#             hard_negs = [doc_id for doc_id, score in 
#                         sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
#                         if doc_id not in positives][:num_hard_negs]
#             if hard_negs:
#                 hard_negatives[qid] = hard_negs
    
#     logging.info(f"Mined hard negatives for {len(hard_negatives)} queries")
#     model.train()
#     return hard_negatives


# def create_train_samples_with_hard_negs(corpus, queries, qrels, hard_negatives):
#     """
#     Create training samples including hard negatives.
    
#     Returns:
#         list of InputExample objects
#     """
#     train_samples = []
    
#     for qid in queries:
#         query_text = queries[qid]
        
#         if qid not in qrels:
#             continue
            
#         # Add positive pairs
#         for doc_id in qrels[qid]:
#             if doc_id in corpus:
#                 pos_text = corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")
#                 train_samples.append(InputExample(texts=[query_text, pos_text]))
        
#         # Add hard negative pairs if available
#         if qid in hard_negatives:
#             for hard_neg_id in hard_negatives[qid]:
#                 if hard_neg_id in corpus:
#                     neg_text = corpus[hard_neg_id].get("title", "") + " " + corpus[hard_neg_id].get("text", "")
#                     # Add as a separate example - the loss will handle it as a negative
#                     train_samples.append(InputExample(texts=[query_text, neg_text], label=0.0))
    
#     return train_samples


# # ==================== Data Loading ====================
# logging.info("Loading data...")

# if args.exp_name == "no_aug":
#     corpus, queries, qrels = GenericDataLoader(
#         corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
#         query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
#         qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", 
#                        f"prompt_tuning_{args.train_num}.tsv")
#     ).load_custom()
# else:
#     weak_query_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", 
#                           args.weak_num, f"weak_queries_{args.train_num}_{args.exp_name}.jsonl")
#     weak_qrels_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", 
#                           args.weak_num, f"weak_train_{args.train_num}_{args.exp_name}.tsv")
#     corpus, queries, qrels = WeakDataLoader(
#         corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
#         query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
#         qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", 
#                        f"prompt_tuning_{args.train_num}.tsv"),
#         weak_query_file=weak_query_file,
#         weak_qrels_file=weak_qrels_file
#     ).load_weak_custom()

# # Load dev set
# dev_corpus, dev_queries, dev_qrels = GenericDataLoader(
#     corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
#     query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
#     qrels_file=join(beir_dir, args.dataset_name, "qrels", "dev.tsv")
# ).load_custom()

# logging.info(f"Training: {len(corpus)} docs, {len(queries)} queries")
# logging.info(f"Dev: {len(dev_corpus)} docs, {len(dev_queries)} queries")


# # ==================== Model Setup ====================
# logging.info(f"Loading model: {args.model_name}")
# word_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_length)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# # ==================== Initial Training Samples ====================
# train_retriever = TrainRetriever(model=None, batch_size=args.batch_size)
# train_samples = train_retriever.load_train(corpus, queries, qrels)
# logging.info(f"Loaded {len(train_samples)} initial training pairs")


# # ==================== Hard Negative Mining (Optional) ====================
# if args.use_hard_negatives and args.mine_hard_negs_before_training:
#     hard_negatives = mine_hard_negatives(
#         model, corpus, queries, qrels, 
#         top_k=args.hard_neg_top_k, 
#         num_hard_negs=args.num_hard_negatives
#     )
#     # Recreate training samples with hard negatives
#     train_samples = create_train_samples_with_hard_negs(corpus, queries, qrels, hard_negatives)
#     logging.info(f"Training samples with hard negatives: {len(train_samples)}")


# # ==================== Loss Function Setup ====================
# logging.info(f"Setting up {args.loss_type} loss function")

# if args.loss_type == "listwise":
#     train_loss = losses.MultipleNegativesRankingLoss(model=model)
#     logging.info("Using MultipleNegativesRankingLoss (listwise softmax)")
    
# elif args.loss_type == "pairwise":
#     train_loss = PairwiseHingeLoss(model=model, margin=args.margin)
#     logging.info(f"Using PairwiseHingeLoss (margin={args.margin})")
    
# elif args.loss_type == "combined":
#     train_loss = OptimizedCombinedLoss(
#         model=model, 
#         margin=args.margin, 
#         alpha=args.alpha, 
#         scale=args.scale
#     )
#     logging.info(f"Using OptimizedCombinedLoss (alpha={args.alpha}, margin={args.margin}, scale={args.scale})")


# # ==================== Evaluator Setup ====================
# dev_evaluator = evaluation.InformationRetrievalEvaluator(
#     dev_queries, 
#     dev_corpus, 
#     dev_qrels, 
#     name=args.dataset_name,
#     show_progress_bar=True,
#     main_score_function=args.product
# )


# # ==================== DataLoader ====================
# train_dataloader = DataLoader(train_samples, batch_size=args.batch_size, shuffle=True)
# logging.info(f"DataLoader created: {len(train_dataloader)} batches per epoch")


# # ==================== GPU Memory Cleanup ====================
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     gc.collect()
#     logging.info(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
#     logging.info(f"GPU device: {torch.cuda.get_device_name(0)}")


# # ==================== Training ====================
# logging.info("=" * 80)
# logging.info("Starting Training")
# logging.info("=" * 80)

# model.fit(
#     train_objectives=[(train_dataloader, train_loss)],
#     evaluator=dev_evaluator,
#     epochs=args.num_epochs,
#     output_path=model_save_path,
#     warmup_steps=args.warmup_steps,
#     optimizer_params={'lr': args.learning_rate},
#     use_amp=args.use_amp,
#     checkpoint_path=model_save_path,
#     checkpoint_save_steps=len(train_dataloader),
#     evaluation_steps=args.evaluation_steps,
#     save_best_model=True,
#     show_progress_bar=True
# )


# # ==================== Save Final Model ====================
# logging.info("=" * 80)
# logging.info("Training Complete")
# logging.info("=" * 80)
# logging.info(f"Saving final model to {model_save_path}")

# model.save(model_save_path)

# # Verify saved files
# config_file = join(model_save_path, "config.json")
# if os.path.exists(config_file):
#     logging.info("✓ Model config.json saved successfully")
# else:
#     logging.warning("✗ config.json not found - attempting to create")
#     try:
#         if hasattr(model._modules['0'], 'auto_model') and hasattr(model._modules['0'].auto_model, 'config'):
#             config = model._modules['0'].auto_model.config
#             config.save_pretrained(model_save_path)
#             logging.info("✓ Created config.json from model configuration")
#     except Exception as e:
#         logging.warning(f"✗ Could not create config.json: {e}")

# logging.info("=" * 80)
# logging.info(f"Model saved to: {model_save_path}")
# logging.info("=" * 80)


# # ==================== Final GPU Memory Report ====================
# if torch.cuda.is_available():
#     logging.info(f"GPU memory after training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
#     logging.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

# logging.info("Training script completed successfully!")

#chatgpt


"""
Train a Bi-Encoder on a BEIR dataset with optional hybrid loss (Listwise + Pairwise Hinge).
Supports weak data augmentation and automatic corpus selection based on --weak_num.
"""

import os
import sys
import pathlib
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import models, SentenceTransformer, losses, evaluation
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever

# ------------------- PATH SETUP -------------------
cwd = os.getcwd()
# Add zhiyuan and xuyang to sys.path
for p in ["zhiyuan", "xuyang"]:
    full_path = os.path.join(cwd, p)
    if full_path not in sys.path:
        sys.path.append(full_path)

# Import weak data loader
from weak_data_loader import WeakDataLoader

# ------------------- ARGUMENTS -------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default="scifact", type=str)
parser.add_argument('--num_epochs', default=2, type=int)
parser.add_argument('--train_num', default=100, type=int)
parser.add_argument('--weak_num', default="5000", type=str)
parser.add_argument('--product', default="cos_sim", type=str)
parser.add_argument('--exp_name', default="no_aug", type=str)
parser.add_argument('--model_name', default="bert-base-uncased", type=str)
parser.add_argument('--version', default="v1", type=str)
parser.add_argument('--loss_fn', type=str, default="listwise_softmax",
                    help="Options: listwise_softmax, pairwise_hinge, hybrid_listwise_hinge")
args = parser.parse_args()

# ------------------- MODEL SAVE PATH -------------------
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(),
                               "output", args.exp_name, str(args.train_num),
                               args.model_name + '-' + args.version + '-' + args.dataset_name)
os.makedirs(model_save_path, exist_ok=True)

# ------------------- LOGGING -------------------
fh = logging.FileHandler(os.path.join(model_save_path, "log.txt"))
ch = logging.StreamHandler(sys.stdout)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[fh, ch])

# ------------------- DATA PATHS -------------------
data_dir = os.path.join(cwd, "zhiyuan", "datasets")
raw_dir = os.path.join(data_dir, "raw")
beir_dir = os.path.join(raw_dir, "beir")
xuyang_dir = os.path.join(cwd, "xuyang", "data")

# ------------------- LOAD DATA -------------------
if args.exp_name == "no_aug":
    corpus_file = os.path.join(beir_dir, args.dataset_name,
                               f"corpus_{args.weak_num}_reduced_ratio_20.jsonl")
    query_file = os.path.join(beir_dir, args.dataset_name, "queries.jsonl")
    qrels_file = os.path.join(xuyang_dir,
                              f"{args.dataset_name}_{args.train_num}",
                              f"prompt_tuning_{args.train_num}.tsv")
    corpus, queries, qrels = GenericDataLoader(
        corpus_file=corpus_file, query_file=query_file, qrels_file=qrels_file
    ).load_custom()
else:
    weak_query_file = os.path.join(xuyang_dir,
                                   f"{args.dataset_name}_{args.train_num}",
                                   args.weak_num,
                                   f"weak_queries_{args.train_num}_{args.exp_name}.jsonl")
    weak_qrels_file = os.path.join(xuyang_dir,
                                   f"{args.dataset_name}_{args.train_num}",
                                   args.weak_num,
                                   f"weak_train_{args.train_num}_{args.exp_name}.tsv")
    corpus_file = os.path.join(beir_dir, args.dataset_name,
                               f"corpus_{args.weak_num}_reduced_ratio_20.jsonl")
    query_file = os.path.join(beir_dir, args.dataset_name, "queries.jsonl")
    qrels_file = os.path.join(xuyang_dir,
                              f"{args.dataset_name}_{args.train_num}",
                              f"prompt_tuning_{args.train_num}.tsv")
    corpus, queries, qrels = WeakDataLoader(
        corpus_file=corpus_file,
        query_file=query_file,
        qrels_file=qrels_file,
        weak_query_file=weak_query_file,
        weak_qrels_file=weak_qrels_file
    ).load_weak_custom()

# Dev set
dev_corpus_file = os.path.join(beir_dir, args.dataset_name,
                               f"corpus_{args.weak_num}_reduced_ratio_20.jsonl")
dev_query_file = os.path.join(beir_dir, args.dataset_name, "queries.jsonl")
dev_qrels_file = os.path.join(beir_dir, args.dataset_name, "qrels", "dev.tsv")
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(
    corpus_file=dev_corpus_file,
    query_file=dev_query_file,
    qrels_file=dev_qrels_file
).load_custom()

# ------------------- TRAIN SAMPLES -------------------
train_retriever = TrainRetriever(model=None, batch_size=32)
train_samples = train_retriever.load_train(corpus, queries, qrels)
logging.info(f"Loaded {len(train_samples)} training pairs.")
logging.info(f"Dev set contains {len(dev_corpus)} documents and {len(dev_queries)} queries")

# ------------------- MODEL -------------------
word_embedding_model = models.Transformer(args.model_name, max_seq_length=350)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# ------------------- LOSS FUNCTION -------------------
if args.loss_fn == "listwise_softmax":
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

elif args.loss_fn == "pairwise_hinge":
    class PairwiseHingeLoss(nn.Module):
        def __init__(self, model, margin=1.0):
            super().__init__()
            self.model = model
            self.margin = margin

        def forward(self, sentence_features, labels):
            reps = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
            embeddings_a = reps[0]
            embeddings_b = torch.cat(reps[1:])
            scores = torch.matmul(embeddings_a, embeddings_b.T)
            pos_scores = scores.diag()
            losses = F.relu(self.margin - (pos_scores.unsqueeze(1) - scores))
            losses.fill_diagonal_(0)
            return losses.mean()

    train_loss = PairwiseHingeLoss(model=model)

elif args.loss_fn == "hybrid_listwise_hinge":
    class FastHybridLoss(nn.Module):
        def __init__(self, model, w_listwise=0.9, w_hinge=0.1, margin=0.5):
            super().__init__()
            self.model = model
            self.listwise_loss = losses.MultipleNegativesRankingLoss(model=model)
            self.w_listwise = w_listwise
            self.w_hinge = w_hinge
            self.margin = margin

        def forward(self, sentence_features, labels):
            # Listwise loss (efficient, batched)
            loss_listwise = self.listwise_loss(sentence_features, labels)

            # Pairwise hinge loss (fast, reuses embeddings)
            with torch.no_grad():
                reps = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
            query_emb = reps[0]
            doc_emb = torch.cat(reps[1:])
            scores = torch.matmul(query_emb, doc_emb.T)
            pos_scores = scores.diag()
            margin_diff = self.margin - (pos_scores.unsqueeze(1) - scores)
            mask = 1 - torch.eye(scores.size(0), device=scores.device)
            loss_hinge = F.relu(margin_diff) * mask
            loss_hinge = loss_hinge.mean()

            return self.w_listwise * loss_listwise + self.w_hinge * loss_hinge

    train_loss = FastHybridLoss(model=model)

else:
    raise ValueError(f"Unsupported loss function: {args.loss_fn}")

# ------------------- EVALUATOR -------------------
dev_evaluator = evaluation.InformationRetrievalEvaluator(
    dev_queries, dev_corpus, dev_qrels,
    name=args.dataset_name, main_score_function=args.product
)

# ------------------- DATALOADER -------------------
train_dataloader = DataLoader(train_samples, batch_size=32, shuffle=True)

# ------------------- TRAINING -------------------
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    logging.info(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

logging.info("Starting training...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=args.num_epochs,
    output_path=model_save_path,
    warmup_steps=100,
    use_amp=False,
    checkpoint_path=model_save_path,
    checkpoint_save_steps=len(train_dataloader),
    evaluation_steps=1000,
    save_best_model=True
)

# ------------------- SAVE FINAL MODEL -------------------
logging.info(f"Training complete. Saving final model to {model_save_path}")
model.save(model_save_path)

# Ensure config.json exists
config_file = os.path.join(model_save_path, "config.json")
if not os.path.exists(config_file):
    try:
        if hasattr(model._modules['0'], 'auto_model') and hasattr(model._modules['0'].auto_model, 'config'):
            model._modules['0'].auto_model.config.save_pretrained(model_save_path)
            logging.info("Created config.json from model configuration.")
    except Exception as e:
        logging.warning(f"Could not create config.json: {e}")

logging.info("Final model saved successfully.")
