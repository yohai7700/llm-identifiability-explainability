
from data.text_datasets import load_text_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer


# lig = LayerIntegratedGradients(model, model.embedding)

def interpret():
    model = AutoModelForSequenceClassification.from_pretrained('./models/checkpoints/llm_cls/distilbert_qwen/model',device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained('./models/checkpoints/llm_cls/distilbert_qwen/model', device_map='cuda')    
    train_dataset, _ = load_text_datasets()
    text = train_dataset[0]['text']
    cls_explainer = SequenceClassificationExplainer(
        model,
        tokenizer)
    word_attributions = cls_explainer(text)
    print(word_attributions)
    cls_explainer.visualize('vis.html')

# vis_data_records_ig = []

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