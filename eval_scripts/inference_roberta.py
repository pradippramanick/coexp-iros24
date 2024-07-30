import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class RobertaNLI:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.label2id = dict(E=2, N=1, C=0)
        self.id2label = {val: key for key, val in self.label2id.items()}

    def forward(self, premise, hypothesis, return_argmax=False):
        input = self.tokenizer(premise, hypothesis, truncation=False, return_tensors="pt")
        output = self.model(input["input_ids"].to(device))
        prediction = torch.softmax(output[0],1)[0]
        if return_argmax:
            argmax_idx = prediction.argmax().item()
            return {'label': self.id2label[argmax_idx], 'score': prediction[argmax_idx].item()}
        else:
            prediction = prediction.tolist()
            return {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, self.label2id.keys())}

    def forward_batch(self, premises, hypothesises):
        pairs = [(p, h) for p, h in zip(premises, hypothesises)]
        tokenized_tensor = self.tokenizer(pairs, return_tensors='pt', truncation=True, padding=True)
        print(tokenized_tensor.input_ids.shape)
        logits_tensor = \
            self.model(input_ids=tokenized_tensor.input_ids, attention_mask=tokenized_tensor.attention_mask)[
                'logits']
        return [self.id2label[logits.argmax().item()] for logits in logits_tensor]

# model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
# model_name = "/home/pradip/PycharmProjects/multimodal_explanation/coherence_assessment/checkpoints/deberta_fine_tuned_e5_lr_1e-5"
# model_name = "/home/pradip/PycharmProjects/multimodal_explanation/coherence_assessment/checkpoints/deberta_fine_tuned_e10_lr_3e-5_merged"
