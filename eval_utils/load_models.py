import transformers
import os
import torch


def load_i2t_model(engine, args=None):
    if engine == "gemini-pro-vision":
        import google.generativeai as genai
        from api_keys import GOOGLE_API_KEY

        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-pro-vision")
        tokenizer = None
        image_processor = None
        processor = None
    elif engine.startswith("otter-"):
        if engine == "otter-mpt":
            # Dec 19, 2023
            ckpt = "luodian/OTTER-Image-MPT7B"
        elif engine == "otter-llama":
            ## Latest: Feb 14, 2024
            ckpt = "luodian/OTTER-Image-LLaMA7B-LA-InContext"
        else:
            raise NotImplementedError
        from otter_ai import OtterForConditionalGeneration

        model = OtterForConditionalGeneration.from_pretrained(
            ckpt, device_map="cuda", torch_dtype=torch.bfloat16
        )
        tokenizer = model.text_tokenizer
        image_processor = transformers.CLIPImageProcessor()
        processor = image_processor
    elif engine.startswith("llava16-"):
        from llava.model.builder import load_pretrained_model as load_llava_model

        if engine == "llava16-7b":
            ckpt = "liuhaotian/llava-v1.6-vicuna-7b"
        elif engine == "llava16-34b":
            ckpt = "liuhaotian/llava-v1.6-34b"
        tokenizer, model, image_processor, context_len = load_llava_model(
            model_path=ckpt,
            model_base=None,
            model_name="llava",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

        processor = image_processor
    elif "deepseek" in engine:
        from transformers import AutoModelForCausalLM
        from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

        ckpt = f"deepseek-ai/deepseek-vl-7b-chat"
        processor: VLChatProcessor = VLChatProcessor.from_pretrained(ckpt)
        tokenizer = processor.tokenizer

        model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            ckpt, trust_remote_code=True
        )
        model = model.to(torch.bfloat16).cuda().eval()

    elif engine == "qwen-vl-chat":
        from transformers.generation import GenerationConfig

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL-Chat", trust_remote_code=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL-Chat",
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
        model.generation_config = GenerationConfig.from_pretrained(
            "Qwen/Qwen-VL-Chat", trust_remote_code=True
        )
        processor = None
    elif engine in "qwen-vl":
        from transformers.generation import GenerationConfig

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL", trust_remote_code=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True
        ).eval()
        model.generation_config = GenerationConfig.from_pretrained(
            "Qwen/Qwen-VL", trust_remote_code=True
        )
        processor = None
    elif engine == "internlm-x2":
        # ckpt = "internlm/internlm-xcomposer2-7b"
        ckpt = "internlm/internlm-xcomposer2-vl-7b"
        model = transformers.AutoModel.from_pretrained(
            ckpt,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cuda",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            ckpt, trust_remote_code=True
        )
        model.tokenizer = tokenizer
        processor = None
    elif engine == "openflamingo":
        from open_flamingo import create_model_and_transforms

        model, processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4,
        )
        model = model.to(torch.bfloat16).cuda()

    elif engine == "emu2-chat":
        from accelerate import (
            init_empty_weights,
            infer_auto_device_map,
            load_checkpoint_and_dispatch,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained("BAAI/Emu2-Chat")
        with init_empty_weights():
            model = transformers.AutoModelForCausalLM.from_pretrained(
                "BAAI/Emu2-Chat",
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).eval()
        # adjust according to your device
        device_map = infer_auto_device_map(
            model,
            max_memory={0: "80GiB", 1: "80GiB"},
            no_split_module_classes=["Block", "LlamaDecoderLayer"],
        )
        device_map["model.decoder.lm.lm_head"] = 0

        model = load_checkpoint_and_dispatch(
            model,
            "/home/tul02009/.cache/huggingface/hub/models--BAAI--Emu2-Chat/snapshots/20ea30b04f8fee599cf97535e655c200df728501",
            device_map=device_map,
        ).eval()
        processor = None
    elif engine == "idefics-9b-instruct":
        from transformers import IdeficsForVisionText2Text, AutoProcessor

        checkpoint = "HuggingFaceM4/idefics-9b-instruct"
        model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == "idefics-9b":
        from transformers import IdeficsForVisionText2Text, AutoProcessor

        checkpoint = "HuggingFaceM4/idefics-9b"
        model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == "idefics-80b-instruct":
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        from accelerate import (
            init_empty_weights,
            infer_auto_device_map,
            load_checkpoint_and_dispatch,
        )

        checkpoint = "HuggingFaceM4/idefics-80b-instruct"
        model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == "gpt4v" or "claude" in engine or "step-1v" in engine:
        model, tokenizer, processor = None, None, None
    else:
        raise NotImplementedError
    return model, tokenizer, processor
