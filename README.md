# Can LLMs generate high-quality synthetic note-oriented doctor-patient conversations?

![Image text](https://github.com/believewhat/MEDIQA-Chat-2023-UMASS_BioNLP/blob/main/figure/sample.png)

# Introduction
This project presents UMASS\_BioNLP team participation in the MEDIQA-Chat 2023 shared task for Task-A (automated clinical note generation from doctor-patient conversations) and Task-C (automated conversation from clinical notes). We focus especially on Task-C and propose a novel LLMs cooperation system named a doctor-patient loop to generate high-quality conversation data sets. The experiment results demonstrate that our approaches yield excellent performance as evaluated by automatic metrics such as ROUGE, medical concept recall, BLEU, and Self-BLEU. We won second place in the competition, but after further prompt engineering, our method can finally achieve better results than the first place in the competition. Furthermore, we conducted a comparative analysis between our proposed method and ChatGPT and GPT-4, thus providing further clarification on why our approach yields superior results. This analysis also investigates the potential of utilizing cooperation LLMs to generate high-quality datasets. Please see our paper for more details: xxxxxxxx.

# Datsets Links (10k version):
1. ChatGPT Dataset
2. GPT-4 Dataset
3. NoteChat Dataset

# Plan
- [ ] Expand the ChatGPT dataset(about 167k).
- [ ] Expand the GPT-4 dataset.
- [ ] Expand our NoteChat-ChatGPT dataset.
- [ ] Expand our NoteChat-GPT4 dataset.
- [ ] Finetune chatbot models.

# LLM Model
1. Demo
2. Model Link

# Experiment

# Citation
```
@inproceedings{mediqa-chat-2023-umass-bionlp,
  author={Junda Wang* and Zonghai Yao* and Avijit Mitra and Samuel Osebe and Zhichao Yang and Hong Yu},
  title     = {Can LLMs generate high-quality synthetic note-oriented doctor-patient conversations?},
  booktitle = {ACL-ClinicalNLP 2023},
  year      = {2023}
}
```



