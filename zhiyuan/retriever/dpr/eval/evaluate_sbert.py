# import os
# import sys
# import json
# import argparse
# import logging
# import pathlib
# from os.path import join
# import random

# from beir.datasets.data_loader import GenericDataLoader
# from beir.retrieval import models
# from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
# from beir.retrieval.evaluation import EvaluateRetrieval

# # Set up local import paths
# cwd = os.getcwd()
# zhiyuan_path = join(cwd, "zhiyuan")
# xuyang_path = join(cwd, "xuyang")

# if zhiyuan_path not in sys.path:
#     sys.path.append(zhiyuan_path)
# if xuyang_path not in sys.path:
#     sys.path.append(xuyang_path)

# # Local imports
# from data_process import load_dl, merge_queries, extract_results

# # Argument parser
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_name', type=str, required=False, default="msmarco")
# parser.add_argument('--train_num', type=int, required=False, default=50)
# parser.add_argument('--dpr_v', type=str, choices=["v1", "v2"], default="v1")
# parser.add_argument('--exp_name', type=str, default="no_aug")
# args = parser.parse_args()

# # Paths
# model_name = "bert-base-uncased"
# model_save_path = os.path.join(
#     pathlib.Path(__file__).parent.parent.absolute(),
#     "train", "output", args.exp_name, str(args.train_num),
#     f"{model_name}-{args.dpr_v}-{args.dataset_name}"
# )
# os.makedirs(model_save_path, exist_ok=True)

# # Setup logging
# log_file = join(model_save_path, "test_log.txt")
# handler = logging.FileHandler(log_file)
# logging.basicConfig(
#     format='%(asctime)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     level=logging.INFO,
#     handlers=[handler]
# )

# # BEIR paths
# data_dir = join(zhiyuan_path, "datasets")
# raw_dir = join(data_dir, "raw")
# beir_dir = join(raw_dir, "beir")
# model_save_path=r"/home/aiml_cse/vishnu/sptar_loss/sptar_v1/zhiyuan/retriever/dpr/train/output/llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70/50/bert-base-uncased-v1-fiqa-combined"
# print(model_save_path)
# # Load model
# model = DRES(models.SentenceBERT(model_save_path), batch_size=256, corpus_chunk_size=100000)
# retriever = EvaluateRetrieval(model, k_values=[1, 3, 5, 10, 100, 300, 500, 1000], score_function="cos_sim")

# # Load corpus and queries
# if args.dataset_name == "msmarco":
#     corpus, queries, qrels = GenericDataLoader(join(beir_dir, args.dataset_name)).load(split="dev")
#     queries_19, qrels_19, qrels_binary_19 = load_dl(join(beir_dir, "TREC_DL_2019"))
#     queries_20, qrels_20, qrels_binary_20 = load_dl(join(beir_dir, "TREC_DL_2020"))
# else:
#     corpus, queries, qrels = GenericDataLoader(join(beir_dir, args.dataset_name)).load(split="test")

# # Sample for debug (optional)
# # corpus = dict(random.sample(corpus.items(), 100000))

# # Prepare evaluation sets
# tobe_eval = {}

# if args.dataset_name == "msmarco":
#     ms_queries = merge_queries(queries, queries_19, queries_20)
#     ms_results = retriever.retrieve(corpus, ms_queries)
#     results, results_19, results_20 = extract_results(ms_results)
#     tobe_eval["dl2019"] = (qrels_19, results_19, qrels_binary_19)
#     tobe_eval["dl2020"] = (qrels_20, results_20, qrels_binary_20)
# else:
#     results = retriever.retrieve(corpus, queries)

# tobe_eval[args.dataset_name] = (qrels, results, "pad")

# # Run evaluation
# for dataset_name in tobe_eval:
#     qrels, results, qrels_binary = tobe_eval[dataset_name]
#     logging.info(f"Retriever evaluation for dataset {dataset_name}")
#     logging.info(f"--- Retriever evaluation for dataset: {dataset_name} ---")

#     # Evaluate returns 4 values
#     ndcg, map, recall, score_per_query = retriever.evaluate(qrels, results, retriever.k_values)
#     # Evaluate returns (ndcg, map, recall, precision)
#     # The fourth value is precision, which we will ignore.
#     # The evaluate method does not return per-query scores. We get them from evaluate_custom.
#     ndcg, _map, recall, _ = retriever.evaluate(qrels, results, retriever.k_values)

#     # score_per_query is a dict of dicts, let's flatten it for saving
#     all_scores = {**ndcg, **map, **recall}

#     # Special handling for TREC-DL binary relevance
#     if dataset_name in ["dl2019", "dl2020"]:
#         _, map_bin, recall_bin, score_per_query_override = retriever.evaluate(qrels_binary, results, retriever.k_values)
#         all_scores.update(map_bin)
#         all_scores.update(recall_bin)
#         # For TREC-DL, MAP and Recall are calculated on binary relevance judgements
#         _, _map, recall, _, score_per_query = retriever.evaluate(qrels_binary, results, retriever.k_values)
#     else:
#         # For other datasets like msmarco (dev), we also want MRR
#         print(retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr"))
#         # mrr, _, _, score_per_query = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
#         mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
#         # Merge MRR into the main metrics
#         ndcg.update(mrr)

#     # Log evaluation results
#     logging.info(f"Results for {dataset_name}:")

#     # Log evaluation results
#     for metric_dict in [ndcg, _map, recall]:
#         logging.info("\n")
#         for k in metric_dict:
#             logging.info(f"{k}: {metric_dict[k]:.4f}")
#         # The dictionaries can be empty if a metric (like MRR for TREC) isn't calculated
#         if metric_dict:
#             for k, v in metric_dict.items():
#                 logging.info(f"{k}: {v:.4f}")

#     # Save to file
#     # Save per-query scores to file for significance testing
#     output_file = join(model_save_path, f"{dataset_name}.json")
#     logging.info(f"Saving per-query scores to {output_file}")
#     with open(output_file, "w") as f:
#         json.dump(all_scores, f, indent=4)
#         json.dump(score_per_query, f, indent=4)

import os
import sys
import json
import argparse
import logging
import pathlib
from os.path import join

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval

# Add custom modules paths
cwd = os.getcwd()
zhiyuan_path = join(cwd, "zhiyuan")
xuyang_path = join(cwd, "xuyang")

if zhiyuan_path not in sys.path:
    sys.path.append(zhiyuan_path)
if xuyang_path not in sys.path:
    sys.path.append(xuyang_path)

from data_process import load_dl, merge_queries, extract_results

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="msmarco")
parser.add_argument('--train_num', type=int, default=50)
parser.add_argument('--dpr_v', type=str, choices=["v1", "v2"], default="v1")
parser.add_argument('--exp_name', type=str, default="no_aug")
args = parser.parse_args()

# --- Hardcoded model save path ---
model_save_path = r"/home/aiml_cse/vishnu/sptar_loss/sptar_v1/zhiyuan/retriever/dpr/train/output/llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70/50/bert-base-uncased-v1-fiqa"

# ---------------------------------------------------

# Make sure directory exists (logging needs it
os.makedirs(model_save_path, exist_ok=True)

# Setup logging to file test_log.txt in model_save_path
log_file = join(model_save_path, "test_log.txt")
handler = logging.FileHandler(log_file)
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[handler]
)

logging.info(f"Using model path: {model_save_path}")

# Dataset directories
data_dir = join(zhiyuan_path, "datasets")
raw_dir = join(data_dir, "raw")
beir_dir = join(raw_dir, "beir")

# Load model and retriever
model = DRES(models.SentenceBERT(model_save_path), batch_size=256, corpus_chunk_size=100000)
retriever = EvaluateRetrieval(model, k_values=[1, 3, 5, 10, 100, 300, 500, 1000], score_function="cos_sim")

# Load data
if args.dataset_name == "msmarco":
    corpus, queries, qrels = GenericDataLoader(join(beir_dir, args.dataset_name)).load(split="dev")
    queries_19, qrels_19, qrels_binary_19 = load_dl(join(beir_dir, "TREC_DL_2019"))
    queries_20, qrels_20, qrels_binary_20 = load_dl(join(beir_dir, "TREC_DL_2020"))
else:
    corpus, queries, qrels = GenericDataLoader(join(beir_dir, args.dataset_name)).load(split="test")

# Retrieve and prepare evaluation sets
tobe_eval = {}

if args.dataset_name == "msmarco":
    merged_queries = merge_queries(queries, queries_19, queries_20)
    full_results = retriever.retrieve(corpus, merged_queries)
    results, results_19, results_20 = extract_results(full_results)
    tobe_eval["dl2019"] = (qrels_19, results_19, qrels_binary_19)
    tobe_eval["dl2020"] = (qrels_20, results_20, qrels_binary_20)
else:
    results = retriever.retrieve(corpus, queries)

tobe_eval[args.dataset_name] = (qrels, results, "pad")

# Evaluate and log results
for dataset_name in tobe_eval.keys():
    qrels, results, qrels_binary = tobe_eval[dataset_name]
    logging.info(f"--- Evaluation for dataset: {dataset_name} ---")

    # Standard metrics
    ndcg, map_score, recall, _ = retriever.evaluate(qrels, results, retriever.k_values)

    # Handle binary relevance for TREC DL
    if dataset_name in ["dl2019", "dl2020"]:
        _, map_bin, recall_bin, _ = retriever.evaluate(qrels_binary, results, retriever.k_values)
        map_score.update(map_bin)
        recall.update(recall_bin)

    # Get MRR scores (evaluate_custom returns dict)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    ndcg.update(mrr)  # merge MRR into main metric dict

    # Combine all scores for saving
    all_scores = {}
    all_scores.update(ndcg)
    all_scores.update(map_score)
    all_scores.update(recall)

    # Logging
    for metric_group in [ndcg, map_score, recall]:
        logging.info("\n")
        for k, v in metric_group.items():
            logging.info(f"{k}: {v:.4f}")

    # Save to JSON file
    output_file = join(model_save_path, f"{dataset_name}.json")
    logging.info(f"Saving evaluation results to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(all_scores, f, indent=4)

logging.info("Evaluation complete!")
