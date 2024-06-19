# FlowLearn

[[Webpage]](https://github.com/Jo-Pan/FlowLearn)[[Paper]](.) [[Data]](https://huggingface.co/datasets/jopan/FlowLearn)

FlowLearn: Evaluating Large Vision-Language Models on Flowchart Understanding

## Data Preparation
We host the our dataset at HuggingFace [here](https://huggingface.co/datasets/jopan/FlowLearn).
```bash
git lfs install
git clone https://huggingface.co/datasets/jopan/FlowLearn
```
The paths for each data subset can be seen in consts_dataset.py
To generate custom Mermaid Flowcharts, you can first use get_mermaid.py and then parse_mermaid.py

## Environment
Different conda environments may be needed for different models.

```bash
conda create -n {env_name} python==3.10 -y
pip install -r requirements/{model.txt}
conda activate {env_name}
```
Replace `{model.txt}` with corresponding file.

Most of the models can be automatically downloaded from Huggingface. 

## Inference
I2T_inference.sh
