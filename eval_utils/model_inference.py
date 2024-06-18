import torch

try:
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import tokenizer_image_token
except:
    pass

import os
import time
from PIL import Image
from consts_dataset import *
from .dataset_processing import *
import PIL.Image


def get_n_shot_support(
    engine,
    data_path,
    support_questions,
    support_answers,
    n_shot,
    short_instruction,
):
    support = []
    support_qn = ""
    input_text = ""
    images = []
    if engine == "gemini-pro-vision":
        for image_file in list(support_answers.keys())[:n_shot]:
            support.append(PIL.Image.open(f"{data_path}/{image_file}"))
            if image_file in support_questions:
                support.append(support_questions[image_file])
            support.append(support_answers[image_file])
    elif "qwen-vl" in engine:
        for image_file in list(support_answers.keys())[:n_shot]:
            support.append({"image": f"{data_path}/{image_file}"})
            text = "User: "
            support_qn = ""
            if image_file in support_questions:
                support_qn = support_questions[image_file]
                # text = f"User: {support_questions[image_file]}\nAssistant: {support_answers[image_file]}\n"
            # else:
            #   support_qn = short_instruction
            #   text = f"Assistant: {support_answers[image_file]}\n"
            text = f"User: {support_qn}\nAssistant: {support_answers[image_file]}\n"
            support.append({"text": text})
    elif "llava" in engine:
        for image_file in list(support_answers.keys())[:n_shot]:
            images.append(Image.open(f"{data_path}/{image_file}").convert("RGB"))
            input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
            support_qn = ""
            if image_file in support_questions:
                support_qn = support_questions[image_file]
            input_text += f"{support_qn}\nAnswer: {support_answers[image_file]}\n"
        support = input_text, images
    elif engine.startswith("otter-"):
        for image_file in list(support_answers.keys())[:n_shot]:
            images.append(Image.open(f"{data_path}/{image_file}").convert("RGB"))
            input_text += "<image>"
            support_qn = ""
            if image_file in support_questions:
                support_qn = support_questions[image_file]

            input_text += f"User: {support_qn}\nGPT:<answer> {support_answers[image_file]}<|endofchunk|>"
        support = input_text, images
    elif "internlm-x" in engine:
        for image_file in list(support_answers.keys())[:n_shot]:
            image = Image.open(f"{data_path}/{image_file}").convert("RGB")
            # image = model.vis_processor(image)
            images.append(image)
            input_text += "<ImageHere>"
            support_qn = ""
            if image_file in support_questions:
                support_qn = support_questions[image_file]
            input_text += f"{support_qn}\nAnswer: {support_answers[image_file]}\n"
        support = input_text, images
    elif "flamingo" in engine:
        for image_file in list(support_answers.keys())[:n_shot]:
            images.append(Image.open(f"{data_path}/{image_file}").convert("RGB"))
            input_text += "<image>"
            support_qn = ""
            if image_file in support_questions:
                support_qn = support_questions[image_file]
            input_text += (
                f"{support_qn}\nAnswer: {support_answers[image_file]}<|endofchunk|>"
            )
        support = input_text, images
    elif "emu2-chat" in engine:
        for image_file in list(support_answers.keys())[:n_shot]:
            images.append(Image.open(f"{data_path}/{image_file}").convert("RGB"))
            input_text += "[<IMG_PLH>]"
            support_qn = ""
            if image_file in support_questions:
                support_qn = support_questions[image_file]
            input_text += f"[{support_qn}\nAnswer: {support_answers[image_file]}]."
        support = input_text, images
    elif "idefics" in engine:
        for image_file in list(support_answers.keys())[:n_shot]:
            support.append(Image.open(f"{data_path}/{image_file}").convert("RGB"))
            support_qn = ""
            if image_file in support_questions:
                support_qn = support_questions[image_file]
            support.append(f"\nUser: {support_qn}")
            # support.append("<end_of_utterance>")
            support.append(f"\nAssistant: {support_answers[image_file]}\n")

    elif "gpt4v" in engine or "step-1v" in engine:
        for image_file in list(support_answers.keys())[:n_shot]:
            support_qn = ""
            if image_file in support_questions:
                support_qn = support_questions[image_file]

            base64_image, mime_type = encode_image(f"{data_path}/{image_file}")
            support.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "low",
                    },
                }
            )
            support.append({"type": "text", "text": support_qn})
            support.append(
                {
                    "type": "text",
                    "text": "The answer is " + str(support_answers[image_file]),
                }
            )
    elif "claude" in engine:
        for image_file in list(support_answers.keys())[:n_shot]:
            support_qn = ""
            if image_file in support_questions:
                support_qn = support_questions[image_file]

            base64_image, mime_type = encode_image(f"{data_path}/{image_file}")
            support.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_image,
                    },
                }
            )

            support.append(
                {
                    "type": "text",
                    "text": f"{support_qn} The answer is "
                    + str(support_answers[image_file]),
                }
            )
    elif "deepseek" in engine:
        from deepseek_vl.utils.io import load_pil_images

        for image_file in list(support_answers.keys())[:n_shot]:
            support_qn = ""
            if image_file in support_questions:
                support_qn = support_questions[image_file]

            if "-v2" in engine:
                if len(support) == 0:
                    support = [
                        {
                            "role": "User",
                            "content": f"<image_placeholder>{support_qn} The answer is {support_answers[image_file]}",
                            "images": [f"{data_path}/{image_file}"],
                        }
                    ]
                else:
                    support[0][
                        "content"
                    ] += f"<image_placeholder>{support_qn} The answer is {support_answers[image_file]}"
                    support[0]["images"].append(f"{data_path}/{image_file}")
            else:
                support += [
                    {
                        "role": "User",
                        "content": f"<image_placeholder>{support_qn}",
                        "images": [f"{data_path}/{image_file}"],
                    },
                    {
                        "role": "Assistant",
                        "content": f"The answer is {support_answers[image_file]}",
                    },
                ]
    return support


def I2T_inference(
    args,
    data_path,
    image_file,
    query_question,
    support,
    engine,
    model,
    tokenizer,
    processor,
    max_new_tokens,
):
    task_instruction = get_task_instruction(args)
    if args.is_cot:
        task_instruction += " Let's think step by step."
    short_instruction = get_task_instruction(args, description="concise")
    if not query_question:
        query_question = ""
        # query_question = short_instruction

    if (
        engine == "gemini-pro-vision"
        or "gpt4v" in engine
        or "claude" in engine
        or "step-1v" in engine
    ):
        # all need api
        if engine == "gemini-pro-vision":
            from api_keys import SAFETY_SETTINGS

            image = PIL.Image.open(f"{data_path}/{image_file}")
            model_input = [image]
            if query_question != "":
                model_input.append(query_question)

        elif engine == "gpt4v" or engine == "step-1v":
            from openai import OpenAI

            if engine == "gpt4v":
                from api_keys import OPENAI_API_KEY as api_key
                from api_keys import OPENAI_ORGANIZATION

                client = OpenAI(api_key=api_key, organization=OPENAI_ORGANIZATION)
                client_model = "gpt-4-vision-preview"

            else:
                from api_keys import STEP_API_KEY

                client = OpenAI(
                    api_key=STEP_API_KEY, base_url="https://api.stepfun.com/v1"
                )
                client_model = "step-1v-32k"

            content = [
                {
                    "type": "text",
                    "text": f"{task_instruction}\nEnsure the generated answers only contain the answer to the question and no other information.",
                }
            ] + support

            base64_image, mime_type = encode_image(f"{data_path}/{image_file}")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "low",
                    },
                }
            )
            content.append({"type": "text", "text": query_question + " The answer is"})
            messages = [{"role": "user", "content": content}]
        elif "claude" in engine:
            import anthropic
            from api_keys import CLAUDE_API_KEYs

            client = anthropic.Anthropic(api_key=CLAUDE_API_KEYs[0])

            content = [
                {
                    "type": "text",
                    "text": f"{task_instruction}\nEnsure the generated answers only contain the answer to the question and no other information.",
                }
            ] + support
            base64_image, mime_type = encode_image(f"{data_path}/{image_file}")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_image,
                    },
                }
            )

            content.append({"type": "text", "text": f"{query_question} The answer is"})

            messages = [{"role": "user", "content": content}]

        wait = 1
        while True:
            try:
                if engine == "gemini-pro-vision":
                    response = model.generate_content(
                        [task_instruction] + support + model_input,
                        safety_settings=SAFETY_SETTINGS,
                    )
                    response.resolve()
                    predicted_answer = response.text
                elif engine in ["gpt4v", "step-1v"]:
                    response = client.chat.completions.create(
                        model=client_model,
                        messages=messages,
                        max_tokens=max_new_tokens,
                    )
                    predicted_answer = response.choices[0].message.content
                    print(image_file, "\t", predicted_answer)
                elif "claude" in engine:
                    response = client.messages.create(
                        model="claude-3-opus-20240229",
                        messages=messages,
                        max_tokens=max_new_tokens,
                    )
                    predicted_answer = response.content[0].text
                    print(image_file, "\t", predicted_answer)
                break
            except Exception as e:
                print(e)
                print(image_file)
                if "content that is not allowed" in str(
                    e
                ) or "content you provided or machine outputted is blocked." in str(e):
                    print(f"content that is not allowed")
                    predicted_answer = ""
                    break
                elif (
                    "error" in e
                    and "message" in e["error"]
                    and "content you provided or machine outputted is blocked."
                    in e["error"]["message"]
                ):
                    print(f"error: {e['error']}")
                    predicted_answer = ""
                    break
                elif (
                    "quick accessor only works when the response contains a valid"
                    in str(e)
                ):
                    predicted_answer = ""
                    break
                else:
                    # client = anthropic.Anthropic(api_key=CLAUDE_API_KEYs[wait % 2])
                    if wait == 3:
                        wait = 12
                    elif wait >= 12:
                        wait = 12 * 23  # wait for another day
                    print(
                        f"{engine}, Waiting for {5*wait} min. Starting from {time.ctime()}"
                    )
                    time.sleep(60 * 5 * wait)  # Wait for 1 hour (3600 seconds)
                    wait += 1
                    continue

    elif "qwen-vl" in engine:
        inputs = [
            {"text": f"You are a helpful assistant. {task_instruction}"}
        ] + support

        inputs.append({"image": f"{data_path}/{image_file}"})
        inputs.append({"text": "User: " + query_question + "\nAssistant: "})
        # if not query_question:
        #     text = "Assistant: "
        # else:
        #     text = "User: " + query_question + "\nAssistant: "
        # inputs.append({"text": text})

        if args.debug:
            print("Mode Input:", inputs)
        total_inputs = tokenizer.from_list_format(inputs)
        inputs = tokenizer(total_inputs, return_tensors="pt")
        inputs = inputs.to(model.device)
        with torch.no_grad():
            pred = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
            )
        input_token_len = inputs["input_ids"].shape[1]
        predicted_answer = tokenizer.decode(
            pred[:, input_token_len:].cpu()[0], skip_special_tokens=True
        )
    elif "otter" in engine:
        support_text, images = support
        input_text = f"{task_instruction}\n" + support_text
        images.append(Image.open(f"{data_path}/{image_file}").convert("RGB"))
        input_text += "<image>"
        input_text += f"User: {query_question}\nGPT:<answer>"

        vision_x = (
            processor.preprocess(images, return_tensors="pt")["pixel_values"]
            .unsqueeze(1)
            .unsqueeze(0)
        )
        lang_x = model.text_tokenizer(
            [
                input_text,
            ],
            return_tensors="pt",
        )
        bad_words_id = tokenizer(
            ["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False
        ).input_ids
        with torch.no_grad():
            predicted_answers = model.generate(
                vision_x=vision_x.to(model.device),
                lang_x=lang_x["input_ids"].to(model.device),
                attention_mask=lang_x["attention_mask"].to(model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                bad_words_ids=bad_words_id,
            )
        input_token_len = lang_x["input_ids"].shape[1]
        predicted_answer = tokenizer.decode(
            predicted_answers[:, input_token_len:].cpu()[0], skip_special_tokens=True
        )
    elif "llava" in engine:
        support_text, images = support
        input_text = f"{task_instruction}\n" + support_text
        images.append(Image.open(f"{data_path}/{image_file}").convert("RGB"))
        input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
        input_text += f"{query_question}\nAnswer:"

        image_tensor = torch.stack(
            [
                processor.preprocess(image_file, return_tensors="pt")["pixel_values"][0]
                for image_file in images
            ]
        )
        image_tensor = image_tensor.half().cuda()
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
            )
        input_token_len = input_ids.shape[1]
        predicted_answer = tokenizer.batch_decode(
            generated_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
    elif "internlm-x" in engine:
        support_text, images_ = support
        images = []
        for image in images_:
            images.append(model.vis_processor(image))

        input_text = f"{task_instruction}\n" + support_text
        query_image = Image.open(f"{data_path}/{image_file}").convert("RGB")
        images.append(model.vis_processor(query_image))
        input_text += "<ImageHere>"
        input_text += f"{query_question}\nAnswer:"
        image = torch.stack(images).to(torch.bfloat16).cuda()
        predicted_answer, history = model.chat(
            tokenizer,
            query=input_text,
            image=image,
            history=[],
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    elif "flamingo" in engine:
        support_text, images = support
        input_text = f"{task_instruction}\n" + support_text
        images.append(Image.open(f"{data_path}/{image_file}").convert("RGB"))
        input_text += "<image>"
        input_text += f"{query_question}\nAnswer:"

        vision_x = [processor(image).unsqueeze(0) for image in images]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        lang_x = tokenizer(
            [input_text],
            return_tensors="pt",
        )
        with torch.no_grad():
            predicted_answers = model.generate(
                vision_x=vision_x.to(torch.bfloat16).cuda(),
                lang_x=lang_x["input_ids"].cuda(),
                attention_mask=lang_x["attention_mask"].cuda(),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        input_token_len = lang_x["input_ids"].shape[1]
        predicted_answer = tokenizer.decode(
            predicted_answers[:, input_token_len:].cpu()[0], skip_special_tokens=True
        )
    elif "emu2-chat" in engine:
        support_text, images = support
        input_text = f"{task_instruction}\n" + support_text
        images.append(Image.open(f"{data_path}/{image_file}").convert("RGB"))
        input_text += "[<IMG_PLH>]"
        input_text += f"{query_question}\nAnswer:"

        inputs = model.build_input_ids(
            text=[input_text], tokenizer=tokenizer, image=images
        )

        with torch.no_grad():
            predicted_answers = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16),
                max_new_tokens=max_new_tokens,
            )
        predicted_answer = tokenizer.decode(
            predicted_answers[:, :].cpu()[0], skip_special_tokens=True
        )
    elif "idefics" in engine:
        prompts = [f"You are a helpful assistant.\n{task_instruction}\n"] + support
        prompts.append(Image.open(f"{data_path}/{image_file}").convert("RGB"))
        prompts.append(query_question)
        # prompts.append("<end_of_utterance>")
        prompts.append("\nAssistant:")

        inputs = processor(
            prompts, add_end_of_utterance_token=False, return_tensors="pt"
        ).to("cuda")
        exit_condition = processor.tokenizer(
            "<end_of_utterance>", add_special_tokens=False
        ).input_ids
        bad_words_ids = processor.tokenizer(
            ["<image>", "<fake_token_around_image>"], add_special_tokens=False
        ).input_ids

        generated_ids = model.generate(
            **inputs,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        input_token_len = inputs["input_ids"].shape[1]
        predicted_answer = tokenizer.decode(
            generated_ids[:, input_token_len:].cpu()[0], skip_special_tokens=True
        )

    elif "deepseek" in engine:
        from deepseek_vl.utils.io import load_pil_images

        input_text = f"{task_instruction}"
        if "-v2" in engine:
            conversation = support + [{"role": "Assistant", "content": ""}]
            conversation += support
            conversation[0]["content"] += f"<image_placeholder>{query_question}"
            conversation[0]["images"].append(f"{data_path}/{image_file}")
        else:
            conversation = [
                {
                    "role": "User",
                    "content": task_instruction,
                },
            ]
            conversation += support
            conversation += [
                {
                    "role": "User",
                    "content": f"<image_placeholder>{query_question}",
                    "images": [f"{data_path}/{image_file}"],
                },
                {"role": "Assistant", "content": "The answer is "},
            ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(model.device)

        with torch.no_grad():
            # run image encoder to get the image embeddings
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )

            predicted_answer = tokenizer.decode(
                outputs[0].cpu().tolist(), skip_special_tokens=True
            )
            # print(f"{prepare_inputs['sft_format'][0]}", answer)
    return predicted_answer
