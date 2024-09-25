# FlowLearn

[[Webpage]](https://github.com/Jo-Pan/FlowLearn)[[Paper]](.) [[Data]](https://huggingface.co/datasets/jopan/FlowLearn)

FlowLearn: Evaluating Large Vision-Language Models on Flowchart Understanding

Flowcharts are graphical tools for representing complex concepts in concise visual representations. This paper introduces the FlowLearn dataset, a resource tailored to enhance the understanding of flowcharts. FlowLearn contains complex scientific flowcharts and simulated flowcharts. The scientific subset contains 3,858 flowcharts sourced from scientific literature and the simulated subset contains 10,000 flowcharts created using a customizable script. The dataset is enriched with annotations for visual components, OCR, Mermaid code representation, and VQA question-answer pairs. Despite the proven capabilities of Large Vision-Language Models (LVLMs) in various visual understanding tasks, their effectiveness in decoding flowcharts - a crucial element of scientific communication - has yet to be thoroughly investigated. The FlowLearn test set is crafted to assess the performance of LVLMs in flowchart comprehension. Our study thoroughly evaluates state-of-the-art LVLMs, identifying existing limitations and establishing a foundation for future enhancements in this relatively underexplored domain. For instance, in tasks involving simulated flowcharts, GPT-4V achieved the highest accuracy (58%) in counting the number of nodes, while Claude recorded the highest accuracy (83%) in OCR tasks. Notably, no single model excels in all tasks within the FlowLearn framework, highlighting significant opportunities for further development.

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
