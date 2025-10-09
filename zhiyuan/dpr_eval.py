import subprocess
import argparse
import os
import pathlib


def multirun(args):
    # Convert args namespace to dictionary
    arg_dict = vars(args)

    for exp in arg_dict["exp_names"]:
        print(f"GPU {arg_dict['gpu_id']} Training: {arg_dict['dataset_name']} on {exp}")

        # Set CUDA_VISIBLE_DEVICES for the subprocesses
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(arg_dict["gpu_id"])

        # Define model save path to check for README.md
        model_save_path = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            "retriever", "dpr", "train", "output", exp, str(arg_dict["train_num"]),
            f"bert-base-uncased-{arg_dict['version']}-{arg_dict['dataset_name']}"
        )
        
        # To allow resuming training, remove the README.md if it exists.
        # The sentence-transformers `fit` method skips training if it finds this file.
        readme_path = os.path.join(model_save_path, "README.md")
        if os.path.exists(readme_path):
            print(f"Removing existing README.md to allow training to resume: {readme_path}")
            os.remove(readme_path)

        # Uncomment below if training is needed
        if arg_dict["version"] == "v1":
            train_command = [
                "python", "zhiyuan/retriever/dpr/train/train_sbert.py",
                "--dataset_name", f"{arg_dict['dataset_name']}",
                "--train_num", f"{arg_dict['train_num']}",
                "--weak_num", f"{arg_dict['weak_num']}",
                "--exp_name", exp
            ] + ["--loss_fn", arg_dict["loss_fn"],
                 "--num_epochs", str(arg_dict["num_epochs"])]
        elif arg_dict["version"] == "v2":
            train_command = [
                "python", "zhiyuan/retriever/dpr/train/train_sbert_BM25_hardnegs.py",
                "--dataset_name", f"{arg_dict['dataset_name']}",
                "--train_num", f"{arg_dict['train_num']}",
                "--weak_num", f"{arg_dict['weak_num']}",
                "--exp_name", exp
            ] + ["--loss_fn", arg_dict["loss_fn"],
                 "--num_epochs", str(arg_dict["num_epochs"])]
        print("Running training command:", " ".join(train_command))
        subprocess.call(train_command, env=env)

        # Always run evaluation
        eval_command = [
            "python", "zhiyuan/retriever/dpr/eval/evaluate_sbert.py",
            "--dataset_name", f"{arg_dict['dataset_name']}",
            "--train_num", f"{arg_dict['train_num']}",
            "--exp_name", exp,
            "--dpr_v", arg_dict["version"]
        ]
        print("Running evaluation command:", " ".join(eval_command), "\n")
        subprocess.call(eval_command, env=env)


def main():
    parser = argparse.ArgumentParser(description='Training Starts ...')
    parser.add_argument("--dataset_name", type=str, help="")
    parser.add_argument("--version", type=str, help="")
    parser.add_argument("--gpu_id", type=int, help="")
    parser.add_argument("--train_num", type=int, help="")
    parser.add_argument("--weak_num", type=str, help="")
    parser.add_argument('-exps','--exp_names', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--loss_fn", type=str, default="listwise_softmax", help="Loss function to use. Options: listwise_softmax, pairwise_hinge")
    args = parser.parse_args()
    multirun(args)


if __name__ == "__main__":
    main()
