"""
Modified from VL-ICL/ICL_utils.py
"""

import random
import copy
import os
import base64
import json
from consts_dataset import *


def get_task_instruction(args, description=None):
    dataset = args.dataset
    task = args.task
    instr = None
    if description is None:
        description = args.task_description
    if description == "nothing":
        instr = ""
        return instr

    # if task == "Flowchart-to-Description":
    #     if description == "concise":

    #     elif description == "detailed":

    if task == "ocr":
        if description == "concise":
            instr = "Answer with the text inside the red box."
        elif description == "detailed":
            instr = "A flowchart will be provided where a red box is drawn around the text node of interest. Answer with the text inside the red box. Ensure that the transcription is precise, reflecting the exact letters."

    elif task == "Flowchart-to-Code":
        if description == "concise":
            instr = "Write a Mermaid Code to represent the flowchart."
        elif description == "detailed":
            instr = "The image contains a flowchart. Generate the Mermaid code to represent the flowchart, reflecting the text nodes and arrows as depicted."

    elif task in ["Flowchart-to-Description", "Flowchart-to-Caption"]:
        if description == "concise":
            instr = "Generate a description of the flowchart."
        elif description == "detailed":
            instr = "The image contains a flowchart. Generate the description of the flowchart, reflecting the text nodes and arrows as depicted."

    elif task.startswith("Flowchart-isTrue") or task.startswith("Flowchart-isFalse"):
        if description == "concise":
            instr = (
                "Determine if the statement about the given flowchart is true or false."
            )
        elif description == "detailed":
            if dataset == "SciFlowchart":
                instr = 'The given image is a flowchart extracted from a scientific literature. Based on the process outlined in the flowchart, determine the correctness of the given statement. Answer with either "true" or "false".'
            elif dataset.startswith("SimFlowchart-"):
                instr = 'The given image is a simulated flowchart. Based on the process outlined in the flowchart, determine the correctness of the given statement. Answer with either "true" or "false".'

    elif dataset.startswith("SimFlowchart-"):
        if task in ["Num_Nodes", "Num_Arrows"]:
            if task == "Num_Nodes":
                target = "text nodes"
            else:
                target = "arrows"

            if description == "concise":
                instr = f"Determine the number of {target} in the flowchart."
            elif description == "detailed":
                instr = f"The given image contains a simulated flowchart. You should find all {target} and determine the total number of {target} in the flowchart. Answer the question with a number."

        elif task == "Flowchart-to-Mermaid":
            if description == "concise":
                instr = "Write a Mermaid Code to represent the flowchart."
            elif description == "detailed":
                instr = "The image contains a simulated flowchart. Generate the Mermaid code to represent the flowchart, reflecting the text nodes and arrows as depicted."
    if instr is None:
        exit(
            f"Task '{task}' is not defined in dataset_preprocessing.py/get_task_instruction"
        )
    return instr


def get_questions_answers(file, dataset, task, is_cot=False):

    if is_cot and task != "Flowchart-to-Mermaid":
        exit("Chain of Thoughts is only available for Flowchart-to-Mermaid task.")
    elif is_cot:
        cot_ans = "First, the flowchart has following nodes: {}.\nThen, the flowchart has following edges: {}\nFinally, the Mermaid code for the flowchart is: {}"
        meta_path = DATA_PATH[dataset]["others"]["all_meta"]
        with open(meta_path, "r") as f:
            all_files = json.load(f)
        all_files = [
            f for f in all_files["train"] + all_files["test"] if f["file"] in file
        ]
        support_meta = {f["file"]: f for f in all_files}

    # True/False
    if task.startswith("Flowchart-is"):
        ans = "true" if task.startswith("Flowchart-isTrue") else "false"
        answers = {image_id: ans for image_id in file.keys()}
        if dataset == "SciFlowchart":
            questions = {}
            for image_id in file.keys():
                questions[image_id] = file[image_id][f"{ans.capitalize()}_Statements"][
                    0
                ]

        elif dataset.startswith("SimFlowchart-"):
            subtask = task.split("-")[-1]
            questions = {}
            for image_id in file.keys():
                # print(image_id)
                tmp = file[image_id][f"Arrow_{subtask}"][ans]
                a = tmp["a"]
                b = tmp["b"]
                if subtask == "AtoB":
                    questions[image_id] = f"Arrow points from node '{a}' to node '{b}'."
                elif subtask == "betweenAB":
                    questions[image_id] = (
                        f"Arrow exists between node '{a}' and node '{b}'."
                    )

    elif task == "ocr" and dataset == "SciFlowchart":
        questions = {}
        answers = {}
        for image_id, item in file.items():
            if "ocr" not in item:
                print(f'OCR is not defined for Dataset "{dataset}" - {image_id}.')
            else:
                answers[image_id] = item["ocr"][1][0]
    else:
        questions = {}
        answers = {}
        for image_id, item in file.items():
            if is_cot:
                answers[image_id] = cot_ans.format(
                    support_meta[image_id]["ocr"],
                    support_meta[image_id]["caption"],
                    support_meta[image_id]["mermaid"],
                )
            elif task in item:
                answers[image_id] = str(item[task])
            else:
                print(f'Task "{task}" is not defined for Dataset "{dataset}".')
                return {}, {}
    return questions, answers


def encode_image(image_path):
    _, file_extension = os.path.splitext(image_path)
    file_extension = file_extension.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    mime_type = mime_types.get(file_extension)
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image, mime_type
