# A dataset for coherence classification of multimodal explanations

This dataset contains data used in the experiments described in the paper - Multimodal Coherent Explanation Generation of Robot Failures, Pradip Pramanick & Silvia Rossi, IROS 2024.
There are two json files:  
  

 1. RoboFail_generated_w_SRL (RF) - contains annotated explanations from the RoboFail dataset (https://github.com/real-stanford/reflect) that are generated using GPT-3.5-turbo LLM, based on observations from Ai2Thor simulator.   
 2. CounterFactual_generated (CF) - contains annotated explanations from counterfactual generation method in (Pramanick & Rossi 2024).

# Fields in json

| Field              | Description                                                                                                                                                                               |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| task_id					       | Unique random id for indexing, includes task type                                                                                                                                         |
| plan_until_failure | A sequence of actions until failure observation, i.e., $\mathcal{E}^\pi$                                                                                                                  |
| action             | Action executed during failure observation                                                                                                                                                |
| observation        | Robot observations after action, i.e., a natural language representation of $\mathcal{E}^{O_i}$                                                                                           |
| explanation        | Expert/LLM generated explanation for RF, template-based explanations for CF                                                                                                               |
| label              | Meta-reasoned coherence label for the $\mathcal{E}^\pi$, $\mathcal{E}^{O_i}$, $\mathcal{E}^{t}$ triplet                                                                                   |
| L_plan_exp         | Coherence label for $\mathcal{E}^\pi, \mathcal{E}^{t}$ pair                                                                                                                               |
| L_obs_exp          | Coherence label for $\mathcal{E}^{O_i}, \mathcal{E}^{t}$ pair                                                                                                                             |
| comment            | Annotator's comment on the labeling decision                                                                                                                                              |
| SRL                | Predicate-argument structure extracted using [(Shi 2019)](https://arxiv.org/abs/1904.05255). Each SRL is a list of predicate and arguments, where each argument has a tag and the tokens. |

> Please note the SRL model is imperfect and the predicate-argument predictions are only given as a reference. For this dataset, the correct labels are given, even for incorrect SRL predictions.  
Also, no SRL for CF, as it is auto-labeled.

We will soon release the code for the paper which will contains scripts to process this data.

