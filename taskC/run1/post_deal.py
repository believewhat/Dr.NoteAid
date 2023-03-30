import pandas as pd
import ipdb
import openai
from multiprocessing.pool import ThreadPool as Pool
openai.api_key = ""
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

tret_sem_ids = ['T059', 'T060', 'T061', 'T058', 'T056'] #test and treatment
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

prompt_check_base = "Check whether the following conversation include all the information from clinical note:\n\nConversation:\n\n"
    
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


def generate(content):
    id, text = content
    prompt_divide = open('./prompts/divide_prompt.txt', 'r').read()
    with open(f"./chat_conv/{id}_unfluence.txt", 'r') as file:
        conversation = file.read()
    conversation = re.sub('Doctor:', '\nDoctor:', conversation)
    convs = re.findall(r'Doctor:[\s\S]+?[Patient:.*]+?\n', conversation)
    divided_conv = []
    for conv in convs:
        messages = [{"role": "user", "content": prompt_divide + conv}]
        result = apply_chatgpt(messages, temperature=0.7, max_tokens=500, presence_penalty=0)
        if 'Doctor:' in result:
            divided_conv.append(result)
        else:
            divided_conv.append(conv)
    QAs = []
    history = divided_conv[0]
    history = re.sub('\n\n', '\n', history)
    with open('connect.txt', 'r') as f:
        prompt_fluence_base = f.read()
    QAs.append(history)
    lastindex = 0
    for i in range(1,len(divided_conv)):
        conv = divided_conv[i]
        conv = re.sub('\n\n', '\n', conv)
        prompt_fluence = prompt_fluence_base + f"\\n\nConversation:\n\n{conv}\n\n\
        Your Answer:\n\n"
        messages = [{"role": "user", "content": prompt_fluence}]
        result = apply_chatgpt(messages, temperature=0, max_tokens=200, presence_penalty=0)
        history = result
        history = re.sub('\n\n', '\n', history)
        if 'Doctor:' not in history:
            history = conv
        QAs.append(history)
    conversation = '\n'.join(QAs)
    conversation = re.sub('\n\n','\n',conversation)
    """
    temp_prompt = prompt_check_base + conversation + "\n\nClinical Note:\n\n" + text + "\n\nIf you find some information that is mentioned in clinical note\
            but it is not mentioned in conversation, please expand and the conversation to include all the important information.\
            Don't shorten conversation length or delete QA. Your conversation should be longer than the conversation I gave you\
             If you want to expand the conversation, please just increase the number of turn of QA.\
            You only need to return your conversation please not return any other information such as clinical note and original conversation, only return the conversation'\n\n"
    messages_check = [{"role": "user", "content": temp_prompt}]
    try:
        conversation_revised = apply_chatgpt(messages_check, temperature=0.7, max_tokens=1800, presence_penalty=0)
    except:
        try:
            conversation_revised = apply_chatgpt(messages_check, temperature=0.2, max_tokens=1500, presence_penalty=0)
        except:
            conversation_revised = apply_chatgpt(messages_check, temperature=0.6, max_tokens=1200, presence_penalty=0)
    """
    """
    prompt_key_word = f"Check whether the following conversation include all the key words from clinical note. \n\nKey Words:\n {cui_code(text)}\n\nConversation:\n\n"
    temp_prompt = prompt_key_word + conversation + "\n\nClinical Note:\n\n" + "\n\nIf you find some information that is mentioned in clinical note\
            but it is not mentioned in conversation, please expand and the conversation to include all the important information.\
            Don't shorten text length or delete QA and add these key words into the conversation.\
             If you want to expand the conversation, please just increase the number of turn of QA.\
            You only need to return the revised conversation please not return any other information such as clinical note and original conversation, only return the conversation'\n\n"
    
    messages_check = [{"role": "user", "content": temp_prompt}]
    try:
        conversation_revised_checked = apply_chatgpt(messages_check, temperature=0.6, max_tokens=2000, presence_penalty=0)
    except:
        try:
            conversation_revised_checked = apply_chatgpt(messages_check, temperature=0.6, max_tokens=1500, presence_penalty=0)
        except:
            conversation_revised_checked = apply_chatgpt(messages_check, temperature=0.6, max_tokens=1200, presence_penalty=0)
    """
    with open(f'./chat_conv/divided_{id}.txt', 'w') as f:
        f.write(conversation)


def main():
    data = pd.read_csv('taskC_testset4participants_inputNotes.csv')
    #prompt_check_base = "Check whether the following conversation include the key words:"
    

    content = []
    for i in range(data.shape[0]):
        content.append((i, data['note'].loc[i]))
    tf = Pool()
    tf.map(generate, content)
    """
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
        """


def main2():
    import numpy as np
    data = pd.read_csv('taskC_testset4participants_inputNotes.csv')
    data['SystemOutput'] = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        with open(f'./chat_conv/{i}_unfluence.txt', 'r') as f:
            dialogue = f.read()
        data['SystemOutput'].loc[i] = dialogue
    data['SystemOutput'].to_csv('taskC_UMASS_BioNLP_run1.csv',index=False)
    for i in range(data.shape[0]):
        with open(f'./chat_conv/divided_{i}.txt', 'r') as f:
            dialogue = f.read()
        data['taskC_UMASS_BioNLP_run2.csv'].loc[i] = dialogue
    data['SystemOutput'].to_csv('taskC_UMASS_BioNLP_run2.csv',index=False)

def convert(content):
    i, conv = content
    prompt = f"Rewrite the following conversations to be one fluence conversation and keep all the information as possible. You should generate a longer conversation\
    ran I gave you\n\nConversation:\n\n{conv}"
    messages = [{"role": "user", "content": prompt}]
    try:
        result = apply_chatgpt(messages, temperature=0, max_tokens=2500, presence_penalty=0)
    except:
        result = apply_chatgpt(messages, temperature=0, max_tokens=1800, presence_penalty=0)
    file = open(f'./chat_conv/{i}_unfluence.txt', 'w')
    file.write(result)
    file.close()
def main3():
    data = pd.read_csv('TaskC-ValidationSet.csv')
    #prompt_check_base = "Check whether the following conversation include the key words:"
    content = []
    for i in range(data.shape[0]):
        with open(f'./chat_conv/divided_{i}.txt', 'r') as f:
            text = f.read()
        content.append((i, text))
    tf = Pool()
    tf.map(convert, content)

main()
main2()
#main3()
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