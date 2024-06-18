# FlowLearn
FlowLearn: Evaluating Large Vision-Language Models on Flowchart Understanding

## Data Preparation [To-be-updated]
We host the our dataset at HuggingFace [here](https://huggingface.co/datasets/ys-zong/VL-ICL).
```bash
git lfs install
git clone https://huggingface.co/datasets/ys-zong/VL-ICL
cd VL-ICL
bash unzip.sh
cd ..
```

## Environment
Different conda environments may be needed for different models.

```bash
conda create -n {env_name} python==3.10 -y
pip install -r requirements/{model.txt}
conda activate {env_name}
```
Replace `{model.txt}` with corresponding file.

Most of the models can be automatically downloaded from Huggingface. For Text-to-image models (Emu1, Emu2, GILL, SEED-LLaMA), please see here for detailed instructions.

## Inference
I2T_inference.sh
