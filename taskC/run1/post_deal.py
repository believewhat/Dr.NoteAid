import pandas as pd
import ipdb
import openai
from chatgpt_conv import chat
from multiprocessing.pool import ThreadPool as Pool
openai.api_key = "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"
from datasets import load_dataset
import ipdb
import os
import pandas as pd
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import cpu_count
import medspacy
from medspacy.util import DEFAULT_PIPENAMES
import openai
import re
medspacy_pipes = DEFAULT_PIPENAMES.copy()
if 'medspacy_quickumls' not in medspacy_pipes: 
    medspacy_pipes.add('medspacy_quickumls')
nlp = medspacy.load(enable = medspacy_pipes, quickumls_path='/home/zhichaoyang/medspacy_test/')

tret_sem_ids = ['T059', 'T060', 'T061', 'T058', 'T056','T033'] #test and treatment
symp_sem_ids = ['T184', 'T034', 'T037', 'T033'] # symptom
dise_sem_ids = ['T020', 'T019','T046', 'T047', 'T048', 'T191', 'T049', 'T050'] #disease
drug_sem_ids = ['T073', 'T074', 'T203', 'T075', 'T200', 'T192'] #drug
sust_sem_ids = ['T120', 'T121', 'T195', 'T122', 'T123', 'T125', 'T126', 'T127', 'T129', 'T130', 'T131', 'T104', 'T109', 'T114', 'T116', 'T197', 'T196', 'T168'] # substance
cuitypes_toinclude = tret_sem_ids + symp_sem_ids + dise_sem_ids + drug_sem_ids + sust_sem_ids


def cui_code(text):
  doc = nlp(text)
  dict_vis = dict()
  for entity in doc.ents:
    flag = 0
    cui = ''
    for ent in entity._.semtypes:
      if ent in cuitypes_toinclude:
        flag = 1
        cui = ent
        break
    if flag and str(entity) not in dict_vis:
      dict_vis[str(entity)] = 1
  return dict_vis


def apply_chatgpt(messages, temperature=0.7, max_tokens=32, presence_penalty=1.5): 
    cnt = 0
    while cnt < 2:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=max_tokens, 
                temperature=temperature,
                presence_penalty=presence_penalty
            )
            content = completion.choices[0].message.content
            return content
        except:
            cnt += 1
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=max_tokens, 
        temperature=temperature,
        presence_penalty=presence_penalty
    )
    content = completion.choices[0].message.content
    return content

def main():
    data = pd.read_csv('taskC_testset4participants_inputNotes.csv')
    #prompt_check_base = "Check whether the following conversation include the key words:"
    prompt_check_base = "Check whether the following conversation include the importan information from clinical note:\n\nConversation:\n\n"
    topics = ['family history/social history',
        'history of present illness',
        'past medical history', 
        'chief complaint', 
        'past surgical history', 
        'allergy', 
        'review of systems', 
        'medications', 
        'assessment', 
        'exam', 
        'disposition', 
        'plan', 
        'diagnosis', 
        'emergency department course', 
        'immunizations', 
        'labs',
        'imaging', 
        'procedures', 
        'gynecologic history', 
        'other_history']
    name_topics = ','.join(topics).upper()
    for i in range(data.shape[0]):
        ipdb.set_trace()
        text = data['note'].loc[i]
        prompt = f"Divide the notes into different parts based on the following topics and all the letters of topic name you generate should be in caps\
            and your topic name should be the same with the following list: \
            {name_topics}.\n If the clinical notes I give didn't mention the topic, your answer should not include them and also don't say none mention just ignore them.\
            Your answer should include all the details of each topic and don't combine the different topics such as: combine and assessment\
            and don't answer which topics are in the clinical note. Here is your clinical your answer should not include the topics that is not mentioned in the note:\n"
        messages = [{"role": "user", "content": prompt + text}]
        result = apply_chatgpt(messages, temperature=0.7, max_tokens=2000, presence_penalty=0)
        results_raw = re.split(r'\n[A-Z:_/\x20]+\n',result)
        topic_names_raw = re.findall(r'\n([A-Z:_\x20]+)\n',result)
        with open(f"./chat_conv/{i}_unfluence.txt", 'r') as file:
            conversation = file.read()
        for sentence in results_raw:
            if len(sentence.split()) > 3:
                temp_prompt = prompt_check_base + conversation + "\n\nClinical Note:\n\n" + sentence + "\n\nIf you find some information that is very important in clinical note\
                but it is not mentioned in conversation, please expand and revised the conversation to include the important information.\
                Don't revise too many parts, just revise the parts you think are missing, and don't revise the complete parts and do not shorten text length.\
                 If you want to expand the conversation, please just increase the number of turn of QA.\
                You only need to return the revised version please not return what is original conversation and don't return the clinical note'\n\n"
                messages_check = [{"role": "user", "content": temp_prompt}]
                conversation_revised = apply_chatgpt(messages_check, temperature=0.2, max_tokens=2000, presence_penalty=0)
                conversation = conversation_revised
        
        conv_cui = cui_code(conversation)
        note_cui = cui_code(data['note'].loc[i])
        conv_word_list = set(list({key: value for key, value in conv_cui.items() if value != 0}.keys()))
        note_word_list = set(list({key: value for key, value in conv_cui.items() if value != 0}.keys()))
        need_word = note_word_list - conv_word_list
        need_word = [word for word in need_word]
        delete_word = conv_word_list - note_word_list
        delete_word = [word for word in delete_word]
        ipdb.set_trace()
        if len(need_word) >= 1:
            for word in need_word:
                sent = [results_raw[0]]
                for sentence in results_raw:
                    if word in sentence:
                        sent.append(sentence)
                sent = '\n'.join(sent)
                prompt_check = prompt_check_base + ','.join(word) + '\n\nYour Conversation:\n\n' + conversation
                prompt_middle = "\n\nIf there are some key words that conversation doesn't mention, please expand the conversation based on the key words and the following information:\n\n"\
                + sent + "\n\nYour answer should only return the revised conversation, if you think the conversation have mentioned all the words, just return the \
                original conversation:\n\n"
                
#main()
"""
data = pd.read_csv('taskc_our.csv')
for i in range(data.shape[0]):
    with open(f'./chat_conv/prompt{i}.txt', 'r') as f:
        dialogue = f.read()
    data['our'].loc[i] = dialogue
data.to_csv('taskc_unfluence.csv',index=False)
"""
"""
data = pd.read_csv('taskc_our.csv')
for i in range(data.shape[0]):
    with open(f'./chat_conv/prompt{i}.txt', 'w') as f:
        f.write('Expand the following conversation based on the clinical note. Check whether there are some important information that clinical note include but conversation not include. \
        Expand the conversation to include all the important information. Do not increase the length of QA but increase the number of the turn of the QA:\n\n Conversation:\n\n')
        f.write(data['our'].loc[i])
        f.write("\n\nNote:\n\n")
        f.write(data['note'].loc[i])
"""