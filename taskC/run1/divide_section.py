import pandas as pd
import ipdb
import openai
from chatgpt_conv import chat
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
import numpy as np
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
  while cnt < 4:
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
def conversation(doctor_prompt, patient_prompt, conv):
  messages_doctor = [{"role": "user", "content": doctor_prompt}]
  try:
    doctor = apply_chatgpt(messages_doctor)
    conv += doctor
    patient_prompt = patient_prompt + conv
    messages_patient = [{"role": "user", "content": patient_prompt}]
    patient = apply_chatgpt(messages_patient)
    flag = 0
    conv += patient
  except:
    ipdb.set_trace()
  return conv, flag
  
def chat(text, max_epochs=15):
  id, header, text = text
  cui_word = cui_code(text)
  conv = ""
  with open(f'./prompts/{header}/prompt_conversation.txt','r',encoding='utf-8') as f:
      prompt = f.read()
  doctor_prompt = prompt + text + "\n" + "Please act as a doctor and ask me one question:\n"
  
  with open(f'./prompts/{header}/prompt_conversation_patient.txt','r',encoding='utf-8') as f:
    prompt_patient = f.read()
  patient_prompt = prompt_patient + text + "\n" "Please act as a patient and answer my question or follow up the conversation:\n"
  conv,_ = conversation(doctor_prompt, patient_prompt, conv)

  for round in range(max_epochs):
    temp_cui = cui_code(conv)
    for cui in temp_cui.keys():
      if cui in cui_word:
        cui_word[cui] = 0
    word_list = list({key: value for key, value in cui_word.items() if value != 0}.keys())
    """
    doctor_prompt = prompt + "\n" + "Please act as a doctor and ask me one question following up the chat history:" + conv + "\n" + \
    "Your question should be around at most four key points which you think is important or if you think you have included \
      all the important points you should check whether the whole conversation include the following list if not please talk about it:" +  ','.join(word_list)
    """
    doctor_prompt = prompt + "\n" + "Please act as a doctor and ask me one question following up the chat history:" + conv + "\n" + \
    "Your question should be around at most four key points"
    doctor_prompt = prompt + doctor_prompt
    patient_prompt = prompt_patient + "\n" + "Please act as a patient whose reading and writing skills is about fifth-grade student \
     and try to answer my question as colloquially as possible or follow up the conversation:\n"
    print(round)
    conv, Flag = conversation(doctor_prompt, patient_prompt, conv)
    print(conv)
    check_finish = "Do you think the following conversation have talked about all the key words from the list?\n \
    The conversation:\n" + conv + "\n" + "The list:\n" + ','.join(word_list) + "\n" + "Just say yes or no"
    messages_check = [{"role": "user", "content": check_finish}]
    content = apply_chatgpt(messages_check)
    if "Yes." in content or "yes" in content:
      break
    if Flag:
      break
  with open('fluence_conversation.txt','r',encoding='utf-8') as f:
    prompt_fluence = f.read()
  fluence_prompt = "Please rewrite all the conversations based on the notes to become fluence and more colloquial, \
    like a normal conversation between the doctor and patient based on the clinical notes, just like this:\n" + \
    prompt_fluence + "\n" + "Now you should rewrite the following conversations and your conversation should include the following key words:\n" + ','.join(list(cui_word.keys())) + "\n \
  you should only return the conversation:\n" + conv + "\n" + "The note:\n" + text + "\n"
  messages_fluence = [{"role": "user", "content": fluence_prompt}]
  fluence_conv = apply_chatgpt(messages_fluence, max_tokens=2000)
  fluence_conv = re.sub('\n\n', '\n', fluence_conv)
  #file = open(f"./chat_conv/{id}.txt", "w")
  #file.write(fluence_conv)
  #file.close()
  return fluence_conv


short2full = {
    "fam/sochx": "FAMILY HISTORY/SOCIAL HISTORY",
    "genhx": "HISTORY OF PRESENT ILLNESS",
    "pastmedicalhx": "PAST MEDICAL HISTORY",
    "cc": "CHIEF COMPLAINT",
    "pastsurgical": "PAST SURGICAL HISTORY",
    "allergy": "ALLERGY",
    "ros": "REVIEW OF SYSTEMS",
    "medications": "medications", 
    "assessment": "ASSESSMENT",
    "exam": "PHYSICAL EXAMINATION",
    "diagnosis": "diagnosis",
    "disposition": "disposition",
    "plan": "PLAN",
    "edcourse": "EMERGENCY DEPARTMENT COURSE",
    "immunizations": "immunizations",
    "imaging": "imaging",
    "gynhx": "GYNECOLOGIC HISTORY",
    "procedures": "procedures",
    "other_history": "other history",
    "labs": "labs"
}

topic_map = dict()
for key, value in short2full.items(): 
    topic_map[value.upper()] = key.upper()




topic_map['ASSESSMENT AND PLAN'] = topic_map['ASSESSMENT']
def divide_chat(text):
  id, text = text
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
  prompt = f"Divide the notes into different parts based on the following topics and all the letters of topic name you generate should be in caps\
            and your topic name should be the same with the following list: \
            {name_topics}.\n If the clinical notes I give didn't mention the topic, your answer should not include them and also don't say none mention just ignore them.\
            Your answer should include all the details of each topic and don't combine the different topics such as: combine and assessment\
            and don't answer which topics are in the clinical note. Here is your clinical your answer should not include the topics that is not mentioned in the note:\n"
  messages = [{"role": "user", "content": prompt + text}]
  result = apply_chatgpt(messages, temperature=0.7, max_tokens=2000, presence_penalty=0)

  results_raw = re.split(r'\n[A-Z:_/\x20]+\n',result)
  topic_names_raw = re.findall(r'\n([A-Z:_/\x20]+)\n',result)
  index = 0
  flag_index = 0
  while len(results_raw[index]) <= 10 and index < len(results_raw): 
    index += 1
  results = []
  topic_names = []
  for i in range(index, len(results_raw)):
    if 'None mentioned' in results_raw[i]:
      continue
    results.append(results_raw[i])
    topic_names.append(topic_names_raw[flag_index])
    flag_index += 1
  headers = []
  for topic in topic_names:
    if topic in topic_map:
        headers.append(topic_map[topic])
    else:
        headers.append('LABS')
  index = 0
  flag_index = 0
  print(index, len(headers), results)
  header = headers[flag_index]
  header = re.sub("/", "_", header)
  content = (flag_index, header, results[index])
  chat_combine = chat(content)
  index += 1
  with open('combine.txt','r',encoding='utf-8') as f:
    prompt_combine_base = f.read()
  print(results)
  
  for i in range(index, len(results)):
    flag_index += 1
    if len(results[i]) < 1:
        continue
    if "None mentioned" in results[i]:
        continue
    header = headers[flag_index]
    header = re.sub("/", "_", header)
    content = (flag_index, header, results[i])
    chat_topic_temp = chat(content)
    prompt_combine = prompt_combine_base + "\n\nFirst one:\n\n" + chat_combine + "\n\nSecond one:\n\n" + chat_topic_temp + "\n\nYour answer should only be one conversation:\n\n"
    messages_combine = [{"role": "user", "content": prompt_combine}]
    try:
      chat_combine = apply_chatgpt(messages_combine, max_tokens=1500)
    except:
      chat_combine = apply_chatgpt(messages_combine, max_tokens=1200)
    print(chat_combine)
  cui_word = cui_code(text)
  file = open(f'./chat_conv/{id}_unfluence.txt', 'w')
  file.write(chat_combine)
  file.close()
  fluence_prompt = "Please rewrite and expand the conversations based on the notes to become fluence and more colloquial, \
    like a normal conversation between the doctor and patient based on the clinical notes,\
    your conversation should include the following key words:\n" + ','.join(list(cui_word.keys())) + "\n \
    For the doctor part, the length of the question should not be long, you should divide the long QA into serveral short QA \
    and you should only return the conversation without any other information:\n" + chat_combine + "\n" + "The note:\n" + result + "\n"
  messages_fluence = [{"role": "user", "content": fluence_prompt}]
  try:
    fluence_conv = apply_chatgpt(messages_fluence, max_tokens=2000)
  except:
    fluence_conv = apply_chatgpt(messages_fluence, max_tokens=1200)
  fluence_conv = re.sub('\n\n', '\n', fluence_conv)
  file = open(f'./chat_conv/{id}.txt', 'w')
  file.write(fluence_conv)
  file.close()
  return fluence_conv

def main():
  try:
    data = pd.read_csv('taskC_testset4participants_inputNotes.csv')
    tf = Pool()
    remain = []
    for i in range(data.shape[0]):
        if os.path.exists(f"./chat_conv/{i}.txt"):
          continue
        remain.append((i, data['note'].loc[i]))
    test = (1, data['note'].loc[1])
    #chat = divide_chat(test)
    #ipdb.set_trace()
    for i in range(len(remain)):
      for k in range(2):
        try:
          chat = divide_chat(remain[i])
          break
        except:
          continue
    chat = tf.map(divide_chat, remain)
    ipdb.set_trace()
  except:
    data['our'] = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
      #with open(f'./chat_conv/{i}_fluence.txt', 'r') as f:
      #with open(f'./chat_conv/{i}_unfluence.txt', 'r') as f:
      with open(f'./chat_conv/prompt{i}.txt', 'r') as f:
        file = f.read()
        data['our'].loc[i] = file
  data.to_csv('taskC_UMASS_BioNLP_run1.csv',index=False)
main()