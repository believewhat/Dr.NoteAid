# Can LLMs generate high-quality synthetic note-oriented doctor-patient conversations?

![Image text](https://github.com/believewhat/MEDIQA-Chat-2023-UMASS_BioNLP/blob/main/figure/sample.png)

# Introduction
In the MEDIQA-Chat 2023, we focus especially on Task-C and propose a novel LLMs cooperation system named a doctor-patient loop to generate high-quality conversation data sets. The experiment results demonstrate that our approaches yield excellent performance as evaluated by automatic metrics such as ROUGE, medical concept recall, BLEU, and Self-BLEU. We won second place in the competition, but after further prompt engineering, our method can finally achieve better results than the first place in the competition. Furthermore, we conducted a comparative analysis between our proposed method and ChatGPT and GPT-4, thus providing further clarification on why our approach yields superior results. This analysis also investigates the potential of utilizing cooperation LLMs to generate high-quality datasets. Please see our paper for more details: xxxxxxxx.

# Datsets Links (10k version):
1. [ChatGPT Dataset](https://drive.google.com/file/d/1wwXYF9ictgZQ0DyxRsbkP5M6tXHxExsC/view?usp=sharing)
2. [GPT-4 Dataset](https://drive.google.com/file/d/17r34QBMq45Ykmc-fkcMEva4zN3hT6Tft/view?usp=sharing)
3. [NoteChat Dataset](https://drive.google.com/file/d/1ZJ3hTCZ6TyJ5sUhkKy80rah0KthwwpX5/view?usp=drive_link)

# Plan
- [ ] Expand the ChatGPT dataset(about 167k).
- [ ] Expand the GPT-4 dataset.
- [ ] Expand our NoteChat-ChatGPT dataset.
- [ ] Expand our NoteChat-GPT4 dataset.
- [ ] Finetune chatbot models.

# LLM Model
1. Demo
2. Model Link

We have now uploaded all versions of the datasets to Hugging Face: [link](https://huggingface.co/datasets/akemiH/NoteChat)


```
@article{wang2023notechat,
  title={NoteChat: A Dataset of Synthetic Doctor-Patient Conversations Conditioned on Clinical Notes},
  author={Wang, Junda and Yao, Zonghai and Yang, Zhichao and Zhou, Huixue and Li, Rumeng and Wang, Xun and Xu, Yucheng and Yu, Hong},
  journal={arXiv preprint arXiv:2310.15959},
  year={2023}
}

@article{wang2023umass_bionlp,
  title={UMASS\_BioNLP at MEDIQA-Chat 2023: Can LLMs generate high-quality synthetic note-oriented doctor-patient conversations?},
  author={Wang, Junda and Yao, Zonghai and Mitra, Avijit and Osebe, Samuel and Yang, Zhichao and Yu, Hong},
  journal={arXiv preprint arXiv:2306.16931},
  year={2023}
}
```


