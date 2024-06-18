"""
Modified from https://github.com/ys-zong/VL-ICL/blob/main/I2T_inference.py
"""

import torch
import os
import json
import argparse
import gc
from eval_utils import model_inference, dataset_processing, load_models

# Define consts_dataset.py for file paths
from consts_dataset import *
from eval_utils.dataset_processing import *
from utils import *
from tqdm import tqdm
import time


# pip 22.2.2 from /home/tul02009/anaconda3/envs/py39/lib/python3.9/site-packages/pip (python 3.9)
def parse_args():
    parser = argparse.ArgumentParser(description="Flowchart Inference")
    parser.add_argument(
        "--dataset",
        default="operator_induction",
        type=str,
        choices=["SciFlowchart", "SimFlowchart-word", "SimFlowchart-char"],
    )
    parser.add_argument(
        "--task",
        default="ocr",
        type=str,
        choices=[
            "ocr",
            "Flowchart-isTrue",
            "Flowchart-isFalse",
            "Flowchart-isTrue-AtoB",
            "Flowchart-isTrue-betweenAB",
            "Flowchart-isFalse-AtoB",
            "Flowchart-isFalse-betweenAB",
            "Num_Nodes",
            "Num_Arrows",
            "Flowchart-to-Mermaid",
            "Flowchart-to-Description",
            "Flowchart-to-Caption",
        ],
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=[
            "openflamingo",
            "otter-llama",
            "llava16-7b",
            "llava16-34b",
            "qwen-vl",
            "qwen-vl-max",
            "qwen-vl-chat",
            "internlm-x2",
            "emu2-chat",
            "idefics-9b-instruct",
            "idefics-80b-instruct",
            "gpt4v",
            "gemini-pro-vision",
            "claude",
            "deepseek-vl-7b-chat",
            "deepseek-vl-7b-chat-v2",  # modified prompting template
            "step-1v",
        ],
        default=["gemini-pro-vision"],
        nargs="+",
    )
    parser.add_argument(
        "--n_shot", default=[0, 1, 2, 4, 8], nargs="+", help="Number of support images."
    )
    parser.add_argument(
        "--max-new-tokens", default=15, type=int, help="Max new tokens for generation."
    )
    parser.add_argument(
        "--task_description",
        default="detailed",
        type=str,
        choices=["nothing", "concise", "detailed"],
        help="Detailed level of task description.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--save_middle", action="store_true", help="Save during inference."
    )
    parser.add_argument(
        "--is_cot", action="store_true", help="Enable chain of thoughts."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    return parser.parse_args()


def eval_questions(
    args,
    engine,
    n_shot,
    model,
    tokenizer,
    processor,
    out_path,
):
    if args.save_middle and os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
    else:
        results = {}
    max_new_tokens = args.max_new_tokens
    if args.task == "ocr":
        image_subset = "ocr"
    else:
        image_subset = "default"
    data_path = DATA_PATH[args.dataset]["images"][image_subset]

    query_file_path = load_json_file(DATA_PATH[args.dataset]["query"])

    query_questions, query_answers = get_questions_answers(
        query_file_path, args.dataset, args.task, args.is_cot
    )

    support_file = load_json_file(DATA_PATH[args.dataset]["support"])
    support_questions, support_answers = get_questions_answers(
        support_file, args.dataset, args.task, args.is_cot
    )
    if "isTrue" in args.task or "isFalse" in args.task:
        if "isTrue" in args.task:
            flip_task = args.task.replace("isTrue", "isFalse")
        elif "isFalse" in args.task:
            flip_task = args.task.replace("isFalse", "isTrue")
        support_questions_flip, support_answers_flip = get_questions_answers(
            support_file, args.dataset, flip_task
        )
        support_questions_og = copy.deepcopy(support_questions)
        support_answers_og = copy.deepcopy(support_answers)
        support_questions = {}
        support_answers = {}
        for i, image_id in enumerate(support_questions_og.keys()):
            if i % 2:
                support_questions[image_id] = support_questions_og[image_id]
                support_answers[image_id] = support_answers_og[image_id]
            else:
                support_questions[image_id] = support_questions_flip[image_id]
                support_answers[image_id] = support_answers_flip[image_id]

    short_instruction = model_inference.get_task_instruction(
        args, description="concise"
    )
    support = model_inference.get_n_shot_support(
        engine, data_path, support_questions, support_answers, n_shot, short_instruction
    )

    image_files = list(query_answers.keys())
    if args.debug:
        image_files = image_files[:5]
    if engine in ["gpt4v", "claude", "step-1v"]:
        image_files = image_files[:100]
    start_time = time.time()
    print(args.engine)
    print(f"Start inference for {len(image_files)} images at {time.ctime()}")

    # for image_file in tqdm(image_files, disable=args.debug == False):
    for image_file in tqdm(image_files):
        if args.save_middle and image_file in results:
            continue
        predicted_answer = model_inference.I2T_inference(
            args,
            data_path,
            image_file,
            query_questions[image_file] if image_file in query_questions else None,
            support,
            engine,
            model,
            tokenizer,
            processor,
            max_new_tokens,
        )
        results[image_file] = {
            "prediction": predicted_answer,
            "reference": query_answers[image_file],
        }

        if args.save_middle and (
            len(results) % 10 == 0
            or engine
            in ["gpt4v", "claude", "step-1v", "deepseek-vl-7b-chat-v2", "llava16-34b"]
        ):
            with open(out_path, "w") as f:
                json.dump(results, f, indent=4)
            # print("saved to", out_path)
    print(
        f"Finished inference for at {time.ctime()}, with total time: {(time.time() - start_time)//60} minutes."
    )
    return results


if __name__ == "__main__":
    args = parse_args()

    if args.task == "ocr":
        image_subset = "ocr"
    else:
        image_subset = "default"

    for engine in args.engine:
        model, tokenizer, processor = load_models.load_i2t_model(engine, args)
        print("Loaded model: {}\n".format(engine))
        set_random_seed(args.seed)
        for shot in args.n_shot:
            out_path = f"results/{args.dataset}/{engine}_{args.task}_{shot}-shot.json"
            if args.is_cot:
                out_path = out_path.replace(".json", "-cot.json")
            print(f"Start evaluating. Output is to be saved to: {out_path}")
            results_dict = eval_questions(
                args,
                engine,
                int(shot),
                model,
                tokenizer,
                processor,
                out_path,
            )
            os.makedirs(f"results/{args.dataset}", exist_ok=True)
            # if args.debug == False:
            with open(out_path, "w") as f:
                json.dump(results_dict, f, indent=4)
            print(f"Finished evaluating. Output saved to: {out_path}")
        del model, tokenizer, processor
        torch.cuda.empty_cache()
        gc.collect()
