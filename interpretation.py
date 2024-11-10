
import torch
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

from data.list_dataset import ListDataset



def interpret():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    index = 100
    model = AutoModelForSequenceClassification.from_pretrained('./models/checkpoints/llm_cls/distilbert_qwen/model',device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained('./models/checkpoints/llm_cls/distilbert_qwen/model', device_map='cuda')    
    eval_dataset = ListDataset(torch.load('./data/checkpoints/yelp/eval_dataset.pt', weights_only=True))
    text = eval_dataset[index]['text']
    cls_explainer = SequenceClassificationExplainer(
        model,
        tokenizer,
        custom_labels=["Human-Authentic", "LLM-Generated"])
    word_attributions = cls_explainer(text)
    print(word_attributions)
    cls_explainer.visualize(f'./interpreter_visualizations/vis-{index}.html')

# vis_data_records_ig = []

# lig = LayerIntegratedGradients(model, model.embedding)

# def interpret_sentence(sentence, label = 0):
#     token_reference = TokenReferenceBase(model.pad_token_id)
#     inputs = tokenizer(sentence, return_tensors="pt").to('cuda')

#     # predict
#     model.eval()
#     pred = model(**inputs).item()
#     pred_ind = round(pred)

#     # generate reference indices for each sample
#     reference_indices = token_reference.generate_reference(7, device=get_args().device).unsqueeze(0)

#     # compute attributions and approximation delta using layer integrated gradients
#     attributions_ig, delta = lig.attribute(inputs['input_ids'], reference_indices, \
#                                            n_steps=500, return_convergence_delta=True)

#     print('pred: ', Label.vocab.itos[pred_ind], '(', '%.2f'%pred, ')', ', delta: ', abs(delta))

#     add_attributions_to_visualizer(attributions_ig, sentence, pred, pred_ind, label, delta, vis_data_records_ig)
    
# def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta, vis_data_records):
#     attributions = attributions.sum(dim=2).squeeze(0)
#     attributions = attributions / torch.norm(attributions)
#     attributions = attributions.cpu().detach().numpy()

#     # storing couple samples in an array for visualization purposes
#     vis_data_records.append(visualization.VisualizationDataRecord(
#                             attributions,
#                             pred,
#                             Label.vocab.itos[pred_ind],
#                             Label.vocab.itos[label],
#                             Label.vocab.itos[1],
#                             attributions.sum(),
#                             text,
#                             delta))