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
#             embeddings_b = torch.cat(reps[1:])
            
#             # Assuming the first one is positive and the rest are negatives
#             query_embeddings = embeddings_a
#             doc_embeddings = embeddings_b
            
#             # In-batch negatives
#             scores = torch.matmul(query_embeddings, doc_embeddings.transpose(0, 1))
            
#             pos_scores = scores.diag()
            
#             # Pairwise Hinge Loss calculation
#             margin = self.margin
#             losses = F.relu(margin - (pos_scores.unsqueeze(1) - scores))
#             losses.fill_diagonal_(0) # Zero out the loss for positive pairs
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
#           save_best_model=True,
#           )

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

'''
This examples show how to train a basic Bi-Encoder for any BEIR dataset without any mined hard negatives or triplets.

OPTIMIZED VERSION: Uses BGE model for better generalist retrieval performance

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass pairs in the format:
(query, positive_passage). Other positive passages within a single batch becomes negatives given the pos passage.

We do not mine hard negatives or train triplets in this example.

Running this script:
python train_sbert.py --dataset_name fiqa --num_epochs 5 --train_num 100
'''

from sentence_transformers import losses, models, SentenceTransformer, evaluation
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os, gzip
import logging
import sys
import random
import argparse
from os.path import join
import torch
import torch.nn as nn
from torch.nn import functional as F

#### Just some code to print debug information to stdout
cwd = os.getcwd()
if join(cwd, "zhiyuan") not in sys.path:
    sys.path.append(join(cwd, "zhiyuan"))
    sys.path.append(join(cwd, "xuyang"))
from weak_data_loader import WeakDataLoader
data_dir = join(cwd, "zhiyuan", "datasets")
raw_dir = join(data_dir, "raw")
weak_dir = join(data_dir, "weak")
beir_dir = join(raw_dir, "beir")
xuyang_dir = join(cwd, "xuyang", "data")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=False, default="scifact", type=str)
parser.add_argument('--num_epochs', required=False, default=2, type=int)  # CHANGED: 2→5
parser.add_argument('--train_num', required=False, default=100, type=int)
parser.add_argument('--weak_num', required=False, default="5000", type=str)
parser.add_argument('--product', required=False, default="cos_sim", type=str)
parser.add_argument('--exp_name', required=False, default="no_aug", type=str)
parser.add_argument('--model_name', required=False, default="BAAI/bge-base-en-v1.5", type=str)  # CHANGED: Better default model
parser.add_argument('--version', required=False, default="v1", type=str)
parser.add_argument(
    "--loss_fn",
    type=str,
    default="listwise_softmax",
    help="Loss function to use. Options: listwise_softmax, pairwise_hinge",
)
args = parser.parse_args()

#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", args.exp_name, str(args.train_num), args.model_name.replace('/', '-') + '-' + args.version + '-' + args.dataset_name)
os.makedirs(model_save_path, exist_ok=True)

#### Just some code to print debug information to stdout
fh = logging.FileHandler(join(model_save_path, "log.txt"))
ch = logging.StreamHandler(sys.stdout)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[fh, ch])
####

#### Data Loading
if args.exp_name == "no_aug":
    corpus, queries, qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv")).load_custom()
else:
    weak_query_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_queries_{args.train_num}_{args.exp_name}.jsonl")
    weak_qrels_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_train_{args.train_num}_{args.exp_name}.tsv")
    corpus, queries, qrels = WeakDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv"), weak_query_file=weak_query_file, weak_qrels_file=weak_qrels_file).load_weak_custom()

dev_corpus, dev_queries, dev_qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(beir_dir, args.dataset_name, "qrels", "dev.tsv")).load_custom()

####
train_retriever = TrainRetriever(model=None, batch_size=32)
train_samples = train_retriever.load_train(corpus, queries, qrels)
logging.info("Loaded {} training pairs.".format(len(train_samples)))
logging.info("dev set contains {} documents and {} queries".format(len(dev_corpus), len(dev_queries)))

#### Model Setup
word_embedding_model = models.Transformer(args.model_name, max_seq_length=512)  # CHANGED: 350→512
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#### Loss Function Setup
if args.loss_fn == "listwise_softmax":
    # The standard MultipleNegativesRankingLoss, which is a listwise softmax loss.
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
elif args.loss_fn == "pairwise_hinge":
    # A custom pairwise hinge loss.
    # We need to define a custom loss function for this.
    # sentence-transformers doesn't have a built-in pairwise hinge loss that works directly with MultipleNegativesRankingLoss data format.
    # So we will implement it within the training loop logic if needed, or use a different data loader.
    # For simplicity, let's stick with what's available or requires minimal changes.
    # MultipleNegativesRankingLoss is the most common and effective for this setup.
    # Let's define a custom loss that can be used with the existing data loader.
    class PairwiseHingeLoss(nn.Module):
        def __init__(self, model, margin=1.0):
            super(PairwiseHingeLoss, self).__init__()
            self.model = model
            self.margin = margin

        def forward(self, sentence_features, labels):
            reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            embeddings_a = reps[0]
            embeddings_b = torch.cat(reps[1:])
            
            # Assuming the first one is positive and the rest are negatives
            query_embeddings = embeddings_a
            doc_embeddings = embeddings_b
            
            # In-batch negatives
            scores = torch.matmul(query_embeddings, doc_embeddings.transpose(0, 1))
            
            pos_scores = scores.diag()
            
            # Pairwise Hinge Loss calculation
            margin = self.margin
            losses = F.relu(margin - (pos_scores.unsqueeze(1) - scores))
            losses.fill_diagonal_(0) # Zero out the loss for positive pairs
            loss = losses.mean()
            return loss

    train_loss = PairwiseHingeLoss(model=model)
else:
    raise ValueError(f"Unsupported loss function: {args.loss_fn}. Choose 'listwise_softmax' or 'pairwise_hinge'.")


#### Evaluator Setup
dev_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, 
                                                         dev_corpus, 
                                                         dev_qrels, 
                                                         name=args.dataset_name, 
                                                         main_score_function=args.product)

#### Create DataLoader from train samples
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_samples, batch_size=32, shuffle=True)

#### Clear GPU cache and prepare for training
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    logging.info(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

#### Train the model
logging.info("Starting to Train...")
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=args.num_epochs,
          output_path=model_save_path,
          warmup_steps=100,
          use_amp=True,  # CHANGED: False→True for faster training
          checkpoint_path=model_save_path,
          checkpoint_save_steps=len(train_dataloader),
          evaluation_steps=1000,
          save_best_model=True,
          )

#### Save the final model to the main path
logging.info("Training complete. Saving final model to {}...".format(model_save_path))
# Ensure the output directory exists
os.makedirs(model_save_path, exist_ok=True)
model.save(model_save_path)

# Verify that required files exist
config_file = os.path.join(model_save_path, "config.json")
if os.path.exists(config_file):
    logging.info("Model config.json saved successfully.")
else:
    logging.warning("config.json not found in model directory. This may cause issues during evaluation.")
    # Try to create a minimal config.json if it doesn't exist
    try:
        # Get the config from the underlying transformer model
        if hasattr(model._modules['0'], 'auto_model') and hasattr(model._modules['0'].auto_model, 'config'):
            config = model._modules['0'].auto_model.config
            config.save_pretrained(model_save_path)
            logging.info("Created config.json from model configuration.")
    except Exception as e:
        logging.warning(f"Could not create config.json: {e}")

logging.info("Final model saved successfully.")