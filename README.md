# coexp-iros24
This repository contains code & data for the paper Multimodal Coherent Explanation Generation of Robot Failures, IROS 2024.

![](teaser.png)


# Setup
Use the requirements.txt file to install the dependencies.

# Experiments
Please follow these steps to make the train-validation-test splits and fine-tune the NLI models for coherence classification.

1. Run `make_splits.py` to reproduce the splits from combining the RoboFail and the CounterFactual datasets, as explained in the paper.
2. To run evaluation on only-NLI baselines, run `eval_RoBERTa-large-MNLI.py` and `eval_DeBERTa-v3-base-NLI.py` inside eval_scripts.
3. Use the script under 'training_scripts' to run fine-tuning experiments. Load the checkpoint with the highest macro F1 on the validation set and run evaluation.


# [Counterfactual-generation Code]

If you need the code for generating counterfactual examples, please contact me by email or create an issue on GitHub, and I'll try to resolve this as soon as possible.
