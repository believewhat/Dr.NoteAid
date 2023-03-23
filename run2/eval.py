
import numpy as np
import pandas as pd
from collections import defaultdict
from datasets import load_dataset, load_metric, concatenate_datasets
import os
from sklearn.metrics import f1_score
# from templates import PATTERNS

import medspacy
import nltk
from medspacy.util import DEFAULT_PIPENAMES
def entity_overlap_scores(predictions, references):
    """
    Evaluation metric using medspacy to detect entities over predicted texts and ground truths.  
    Filtered with UMLS Semantic Type to include "meaningful" entities. 
    See here for more detail https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/documentation/SemanticTypesAndGroups.html
    Args:
    predictions: list of str predicted from generator model
    references: list of str  gold text/ground truth  
    same as metric rouge
    """
    # init 
    tret_sem_ids = ['T059', 'T060', 'T061', 'T058', 'T056'] #test and treatment
    symp_sem_ids = ['T184', 'T034', 'T037', 'T033'] # symptom
    dise_sem_ids = ['T020', 'T019','T046', 'T047', 'T048', 'T191', 'T049', 'T050'] #disease
    drug_sem_ids = ['T073', 'T074', 'T203', 'T075', 'T200', 'T192'] #drug
    sust_sem_ids = ['T120', 'T121', 'T195', 'T122', 'T123', 'T125', 'T126', 'T127', 'T129', 'T130', 'T131', 'T104', 'T109', 'T114', 'T116', 'T197', 'T196', 'T168'] # substance
    cuitypes_toinclude = tret_sem_ids + symp_sem_ids + dise_sem_ids + drug_sem_ids + sust_sem_ids
    # df = pd.read_csv('/home/zhichaoyang/medspacy_test/SemanticTypes_2018AB.txt', sep="|", names=['abrev','TUI','text'])
    # tui2abrev = dict(zip(df.TUI, df.abrev))
    medspacy_pipes = DEFAULT_PIPENAMES.copy()
    if 'medspacy_quickumls' not in medspacy_pipes: 
        medspacy_pipes.add('medspacy_quickumls')
    print(medspacy_pipes)
    nlp = medspacy.load(enable = medspacy_pipes, quickumls_path='/home/zhichaoyang/medspacy_test/')

    results = {"concept_recall": [], "concept_precision": [], "concept_f1": []}
    assert len(predictions) == len(references)
    # compare
    for pred, reff in zip(predictions, references):
        pred_cuis = []
        doc = nlp(pred)
        for ent in doc.ents:
            for a in ent._.semtypes:
                if a in cuitypes_toinclude:
                    pred_cuis.append(ent.label_)
                    break
            # print('Entity text : {}'.format(ent.text))
            # print('Label (UMLS CUI) : {}'.format(ent.label_))
            # print('Similarity : {}'.format(ent._.similarity))
            # print('Semtypes : {}'.format(ent._.semtypes))
        reff_cuis = []
        doc = nlp(reff)
        for ent in doc.ents:
            for a in ent._.semtypes:
                if a in cuitypes_toinclude:
                    reff_cuis.append(ent.label_)
                    break
        num_intersect = len(set(pred_cuis).intersection(set(reff_cuis)))
        num_rec = len(set(reff_cuis))
        num_pre = len(set(pred_cuis))
        rec = float(num_intersect)/(num_rec+ 0.000001) 
        pre = float(num_intersect)/(num_pre+ 0.000001) 
        f1 = (2 * pre * rec) / (pre + rec + 0.000001) 
        results["concept_recall"].append(rec)
        results["concept_precision"].append(pre)
        results["concept_f1"].append(f1)
    results = {"concept_recall": np.array(results["concept_recall"]).mean()*100, 
        "concept_precision": np.array(results["concept_precision"]).mean()*100, 
        "concept_f1": np.array(results["concept_f1"]).mean()*100
    }
    return results

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def getheadername(header_full):
    short2full = {
        "fam/sochx": "FAMILY HISTORY/SOCIAL HISTORY",
        "genhx": "HISTORY of PRESENT ILLNESS",
        "pastmedicalhx": "PAST MEDICAL HISTORY",
        "cc": "CHIEF COMPLAINT",
        "pastsurgical": "PAST SURGICAL HISTORY",
        "allergy": "allergy",
        "ros": "REVIEW OF SYSTEMS",
        "medications": "medications", 
        "assessment": "assessment",
        "exam": "exam",
        "diagnosis": "diagnosis",
        "disposition": "disposition",
        "plan": "plan",
        "edcourse": "EMERGENCY DEPARTMENT COURSE",
        "immunizations": "immunizations",
        "imaging": "imaging",
        "gynhx": "GYNECOLOGIC HISTORY",
        "procedures": "procedures",
        "other_history": "other_history",
        "labs": "labs"
    }
    full2short = {}
    for k,v in short2full.items():
        full2short[v.lower()] = k.upper()
    return full2short[header_full]




# read prediction
gen_path = "/home/zhichaoyang/pubmedgpt_ct_data/mediqachat_codetosubmit/tmp/mediqachat_medinsgpt_hc_result_to_submit/generated_predictions_dev/generated_predictions_dev.csv"
gen_df = pd.read_csv(gen_path)

# gen_path = "/home/zhichaoyang/pubmedgpt_ct_data/mediqachat_codetosubmit/tmp/mediqachat_medinsgpt_hc_result_to_submit/generated_predictions_dev/chatgpt_api_result.csv"
# gen_df = pd.read_csv(gen_path)



# read data
data_files = {}
data_files["train"] = f"/home/zhichaoyang/pubmedgpt_ct_data/data_processed/mediqa/train.json"
data_files["test"] = f"/home/zhichaoyang/pubmedgpt_ct_data/data_processed/mediqa/valid.json"
datasets_bytask = load_dataset("json", data_files=data_files, cache_dir="/home/zhichaoyang/pubmedgpt_ct_data/mediqa_eval/hf_data_cache")
len_test = len(datasets_bytask["test"])
print(len_test)
secheader2evalidx = defaultdict(list)
idx = 0
for row in datasets_bytask["test"]:
    secheader2evalidx[row["section_header"]].append(idx)
    idx += 1
print(secheader2evalidx.keys())

# eval header accuracy
preds = gen_df["SystemOutput1"].tolist()
# preds = gen_df["SystemOutput2_header"].tolist()
golds = datasets_bytask["test"]["section_header"]
golds = [getheadername(a) for a in golds]
assert len(preds) == len(golds)
acc = 0
for a,b in zip(preds, golds):
    if a == b:
        acc += 1
acc = acc/len(preds)
print("accuracy")
print(acc)
print("f1")
print(f1_score(golds, preds, average="macro"))


# load eval metrics
metric_rouge = load_metric('rouge')
metric_bertscore = load_metric('bertscore')
predictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]
results = metric_bertscore.compute(predictions=predictions, references=references, model_type='microsoft/deberta-xlarge-mnli')


def compute_metrics(decoded_preds, decoded_labels):
    result = entity_overlap_scores(decoded_preds, decoded_labels)
    tmp = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    for key, value in tmp.items():
        result[key+"_hig"] = value.high.fmeasure * 100 
        result[key+"_mid"] = value.mid.fmeasure * 100 
        result[key+"_low"] = value.low.fmeasure * 100 
    tmp = metric_bertscore.compute(predictions=decoded_preds, references=decoded_labels, model_type='microsoft/deberta-xlarge-mnli')
    result["bertscore_precision"] = np.mean(results['precision']) * 100 
    result["bertscore_recall"] = np.mean(tmp["recall"]) * 100 
    result["bertscore_f1"] = np.mean(tmp["f1"]) * 100 
    return result


# start eval conv2note, note2conv
decoded_labels = datasets_bytask["test"]["section_text"]
decoded_preds = gen_df["SystemOutput2"].tolist()
# decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
rouge_conv2note = compute_metrics(decoded_preds, decoded_labels)

print(f"rouge2 for conv2note task : {rouge_conv2note['rouge2_mid']}")  # 15
print(f"conceptf1 for conv2note task : {rouge_conv2note['concept_f1']}")
print(f"concept_prec for conv2note task : {rouge_conv2note['concept_precision']}")
print(f"concept_recall for conv2note task : {rouge_conv2note['concept_recall']}")
print(f"bertscore_f1 for conv2note task : {rouge_conv2note['bertscore_f1']}")
print(f"bertscore_precision for conv2note task : {rouge_conv2note['bertscore_precision']}")
print(f"bertscore_recall for conv2note task : {rouge_conv2note['bertscore_recall']}")


print("Done")
