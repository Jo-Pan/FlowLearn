#### NOTES ####
# for otter: xformers needs to be 0.0.22 instead of 0.0.25.post1
# llava needs transformers == 4.31
# idefics needs  v == 4.32
# deepseek need python 3.11

#### PARAMETERS ####
tasks=("ocr" "Num_Nodes" "Num_Arrows" "Flowchart-isTrue-AtoB" "Flowchart-isTrue-betweenAB" "Flowchart-isFalse-AtoB" "Flowchart-isFalse-betweenAB")
tasks500=("Flowchart-to-Description" "Flowchart-to-Mermaid")
tasksSci=("ocr" "Flowchart-isTrue" "Flowchart-isFalse")
engine="qwen-vl-chat"

#### SimFlowchart ####
for dataset in "SimFlowchart-word" "SimFlowchart-char"
do
    for task in "${tasks[@]}"
    do
    
        CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine "$engine" \
        --n_shot 2 --dataset "$dataset" --task "$task" --max-new-tokens 100 --save_middle
    done

    for task in "${tasks500[@]}"
    do
        CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine "$engine" \
        --n_shot 2 --dataset "$dataset" --task "$task" --max-new-tokens 500 --save_middle
    done
done

#### SciFlowchart ####
for task in "${tasksSci[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine "$engine" \
    --n_shot 2 --dataset SciFlowchart --task "$task" --max-new-tokens 100 --save_middle
done

CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine "$engine" \
--n_shot 2 --dataset SciFlowchart --task Flowchart-to-Caption --max-new-tokens 500 --save_middle 

#### Chain of Thought ####
# for dataset in "SimFlowchart-char"
# do
#     CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine "$engine" \
#         --n_shot 2 --dataset "$dataset" --task "$task" --max-new-tokens 100 --save_middle --is_cot
# done