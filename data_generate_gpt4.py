import ipdb
import os
import pandas as pd
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import cpu_count
import medspacy
from medspacy.util import DEFAULT_PIPENAMES
import openai
import re
import time
import argparse
import numpy as np
import re
import random

medspacy_pipes = DEFAULT_PIPENAMES.copy()
if 'medspacy_quickumls' not in medspacy_pipes: 
    medspacy_pipes.add('medspacy_quickumls')
nlp = medspacy.load(enable = medspacy_pipes, quickumls_path='../medspacy_test/')

tret_sem_ids = ['T059', 'T060', 'T061', 'T058', 'T056'] #test and treatment
symp_sem_ids = ['T184', 'T034', 'T037', 'T033'] # symptom
dise_sem_ids = ['T020', 'T019','T046', 'T047', 'T048', 'T191', 'T049', 'T050'] #disease
drug_sem_ids = ['T073', 'T074', 'T203', 'T075', 'T200', 'T192'] #drug
sust_sem_ids = ['T120', 'T121', 'T195', 'T122', 'T123', 'T125', 'T126', 'T127', 'T129', 'T130', 'T131', 'T104', 'T109', 'T114', 'T116', 'T197', 'T196', 'T168'] # substance
cuitypes_toinclude = tret_sem_ids + symp_sem_ids + dise_sem_ids + drug_sem_ids + sust_sem_ids

def lower_check(text):
    # The BAGEL dataset uses X to replace named entities.
    if text.startswith("X ") == False:
        text1 = text[0].lower() + text[1:]
    else:
        text1 = text
    return text1

def cui_code(text):
  doc = nlp(text)
  dict_vis = dict()
  cui_code = dict()
  cui_entity = {}
  for entity in doc.ents:
    flag = 0
    cui = ''
    for ent in entity._.semtypes:
      if ent in cuitypes_toinclude:
        flag = 1
        cui = ent
        break
    if flag and str(entity) not in dict_vis:
      dict_vis[str(entity).lower()] = 1
      cui_code[entity.label_] = str(entity).lower()
      cui_entity[str(entity).lower()] = entity.label_
  return dict_vis, cui_code, cui_entity

def apply_chatgpt(messages, temperature=0.5, max_tokens=-1, presence_penalty=0, frequency_penalty=0, method="gpt-4"):
  cnt = 0
  while cnt < 5:
    cnt += 1
    if max_tokens != -1:
      try:
        completion = openai.ChatCompletion.create(
          model=method,
          messages=messages,
          max_tokens=max_tokens, 
          temperature=temperature,
          presence_penalty=presence_penalty,
          frequency_penalty=frequency_penalty
        )
        content = completion.choices[0].message.content
        break
      except:
        time.sleep(1)
        if max_tokens>200:
          max_tokens -= 200
    else:
      try:
        completion = openai.ChatCompletion.create(
          model=method,
          messages=messages,
          temperature=temperature
        )
        content = completion.choices[0].message.content
        break
      except:
        continue
  if cnt == 5:
    ipdb.set_trace()
  return content


def diff(key1, key2, entity1, code2, content):
  aw = []
  for key in key1:
    if entity1[key] not in code2 and ' ' + key + ' ' not in content:
      aw.append(key)
  return aw



def conversation(doctor_prompt, patient_prompt, history, conv, conv_m, text_patient, text_doctor='', keywords='', reference=''):
  doctor_message = [{"role": "system", "content": doctor_prompt}]
  if len(text_doctor) > 0:
    doctor_message.append({"role": "user", "content":  'Clinical Note:\n' + text_doctor})
  if len(keywords) > 0:
    doctor_message.append({"role": "user", "content":  'Key Words:\n' + keywords})
  if len(reference) > 0:
    doctor_message.append({"role": "user", "content":  'Serveral examples you could use to generate:\n' + reference})
  doctor_message = doctor_message + history
  doctor_message.append({"role": "user", "content":  "You should only generate one utterance based on history conversation. Remenber you are doctor not patient. Please only return conversation. Add 'Doctor:' before this utterance. Don't mention the information that has been mentioned in history conversation. If you feel that the patient's information is incomplete, you can supplement it based on the clinical note and include relevant keywords. However, please refrain from saying, 'based on medical record or clinical note.' Instead, you should say, 'I guess...'"})
  if len(keywords) > 0:
    doctor_message.append({"role": "user", "content": "You should include all the keywords I provided to you and corresponding information of the clinical note. If it's not possible to include them all, you can use the original words in the notes to construct the sentences. Your generation must follow the logical sequence of a doctor's inquiry. Your generated responses should be as concise as possible. You shoudn't use the abbreviation if you know the full name(you should use full name not abbreviation, such as D9 must be day 9, D7 must be day 7. Add the Doctor: before your generation and you must follow up the role play if you cannot you should ouput Doctor:"})
  doctor = apply_chatgpt(doctor_message, temperature=0.7, max_tokens=200)
  if 'Doctor:' not in doctor:
    doctor = 'Doctor:' + doctor
  print(doctor)
  conv_m.append({"role": "user", "content": doctor})
  patient_message = [{"role": "system", "content": patient_prompt}]
  patient_message.append({"role": "user", "content": 'Clinical Note:\n' + text_patient})
  patient_message = patient_message + history
  patient_message.append({"role": "user", "content": doctor})
  patient_message.append({"role": "user", "content": patient_prompt + "Your reply should be succinct and accurate in a colloquial lay language style and must be aligned with clinical note. Don't generate the part which should be said by doctor. Do not say all the information unless the doctor asks about it. You cannbot say any information of your test result or vital signs. Your medical history, vaccination history and medication history are all belong to medical history. Your reply must be completely aligned with clinical note. But you cannot say any examination or test results because you are not doctor. You must not be able to use highly specialized terms or medical terminology. You can only describe limited common symptoms. You shoudn't use the abbreviation if you know the full name(you should use full name not abbreviation, such as D9 must be day 9, D7 must be day 7. You must generate something which is on the clinical note or you could say I don't know."})
  #ipdb.set_trace()
  patient = apply_chatgpt(patient_message, temperature=0.5, max_tokens=100)
  print(patient)
  conv_m.append({"role": "user", "content": patient})
  conv = conv + '\n' + doctor +'\n'+ patient
  return conv, conv_m


def judge_exist(key, text):
  cui1, cui2, cui3 = cui_code(key)
  cui11, cui22, cui33 = cui_code(key)
  for key in cui2.keys():
    if key in list(cui22.keys()) or ' ' + key in text:
      return True
    else:
      prompt = f"Check whether the conversation include this key words:{key}(It must be an exact match)"
      messages = [{"role": "user", "content": prompt}]
      messages.append({"role": "user", "content": f"Conversation:\n{text}"})
      messages.append({"role": "user", "content": f"you should only return yes or no"})
      answer = apply_chatgpt(messages, temperature=0, max_tokens=10, method='gpt-3.5-turbo')
      if 'yes' in answer or 'Yes' in answer:
        return True
      return False


def update_key(select_key_words, conv, vis_dict):
  new_list = []
  now_num = len(select_key_words.split(','))
  for keyword in select_key_words.split(','):
    if len(keyword) == 0:
      continue
    if keyword not in vis_dict and not judge_exist(keyword, conv):
      new_list.append(keyword)
      vis_dict[keyword] = 1
  select_key_words = ','.join(new_list)
  return select_key_words

def chat(text, history_conv='', flag=0, max_epochs=60):
  print(max_epochs)
  id, header, text = text
  cui_word,cui_word_code, cui_word_entity = cui_code(text)
  conv = ""
  word_list = list({key: value for key, value in cui_word.items() if value != 0}.keys())
  conv_m = []
  prompt1 = "Doctor: Good Morning, how are you feeling today"
  history = []
  patient_prompt = f"Act as a patient to reply the doctor and tell the doctor why you come here(you must only talk about your symptoms and you shouldn't mention any other information). Add '\nPatient:' before in each round. Your answer should align with the clinical notes. You are just an ordinary person, your response should be made as colloquial as possible. Don't mention any specialized diagnostic experimental results, vital signs and some conclusions because you're just an ordinary person and may not understand the meaning of these results. Your response should revolve around the doctor's words and avoid adding information that was not mentioned."
  messages_question = [{"role": "user", "content": patient_prompt}]
  messages_question.append({"role": "user", "content": 'Clinical Note:' + text})
  messages_question.append({"role": "user", "content": f"History Conversation\n{prompt1}"})
  messages_question.append({"role": "user", "content": "Your reply should be succinct and accurate in a colloquial lay language style and must be aligned with clinical note. Don't generate the part which should be said by doctor. Do not say all the information unless the doctor asks about it. You cannot say any information of your test result or vital signs. Your reply must be completely aligned with clinical note. But you cannot say any examination or test results because you are not doctor. "})
  questions = apply_chatgpt(messages_question, temperature=0.7, max_tokens=150)
  conv_m.append({"role": "user", "content": questions})
  conv_m.append({"role": "user", "content": "Doctor: Can you tell me about your medical history or give me your medical history record?"})
  prompt2 = prompt1 + '\n' + questions + '\n' + "Doctor: Can you tell me about your medical history or give me your medical history record?"
  messages_question = [{"role": "user", "content": patient_prompt}]
  messages_question.append({"role": "user", "content": 'Clinical Note:' + text})
  messages_question.append({"role": "user", "content": f"History Conversation\n{prompt2}"})
  messages_question.append({"role": "user", "content": "Your reply should be succinct and accurate in a colloquial lay language style and must be aligned with clinical note. Don't generate the part which should be said by doctor. Do not say all the information unless the doctor asks about it. You cannbot say any information of your test result or vital signs. Your reply must be completely aligned with clinical note. But you cannot say any examination or test results because you are not doctor"})
  questions = apply_chatgpt(messages_question, temperature=0.7, max_tokens=150)
  conv = prompt2 + '\n' + questions + "\n(After doctor updating the medical history records)\n"
  conv_m.append({"role": "user", "content": questions})
  conv_m.append({"role": "user", "content": "\n(After doctor updating the medical history records)\n"})
  prompt = f"Continue to generate 80 to {max_epochs} utterances conversations between doctor and patient to ask or tell the patient regarding the case(you must follow up the history conversation). The conversations you generate must cover all the keywords I gave you.  You cannot revise or eliminate any key words and you cannot use synonyms of the keywords. Your conversation should also include all information. If it's difficult to include all the information and key words, you can use the original sentences in the clinical note."
  messages_question = [{"role": "user", "content": prompt}]
  messages_question.append({"role": "user", "content": 'Clinical Note:' + text})
  messages_question.append({"role": "user", "content": 'Key Words:' + ','.join(list(cui_word.keys()))})
  messages_question = messages_question + conv_m
  messages_question.append({"role": "user", "content": "Your conversations must include all the keywords I provided to you, and if it's not possible to include them all, you can make slight modifications based on the original wording in the notes.  You cannot revise or eliminate any key words and you cannot use synonyms of the keywords. Your conversation should also include all information. If it's difficult to include all the information and key words, you can use the original sentences in the clinical note. Your generation must follow the logical sequence of a doctor's inquiry. Your conversations must follow the logical sequence of a doctor's inquiry. For example, the general logical order of the conversation is: first discussing symptoms, then discussing the medical history, followed by discussing testing and results, and finally discussing the conlusion and treatment options, etc. The doctor didn't know any information of medical history or symptoms. These information should be told by patient"})
  questions = apply_chatgpt(messages_question, temperature=0.7, method='gpt-4-1106-preview')
  cui_note_word, cui_note_code, cui_note_entity = cui_code(text)
  cui_conv_word, cui_conv_code, cui_conv_entity = cui_code(questions)
  delete_key = diff(cui_conv_word, cui_note_word, cui_conv_entity, cui_note_code, text)
  add_key = diff(cui_note_word, cui_conv_word, cui_note_entity, cui_conv_code, questions)
  question = questions.split('\n')
  question_list = []
  for q in question:
    if len(q) > 2:
      question_list.append(q)
  key_words = []
  
  
  for i in range(0, len(question_list), 2):
    qs = question_list[i:i+2]
    temp_cui, _, _ = cui_code(qs[0])
    temp_key = ','.join(list(temp_cui.keys()))
    try:
      temp_cui, _, _ = cui_code(qs[1])
      temp_key = temp_key + ',' + ','.join(list(temp_cui.keys()))
      key_words.append(temp_key)
    except:
      continue
  revise_key_words = []
  for keys in key_words:
    temp = []
    for key in keys.split(','):
      temp.append(key)
    revise_key_words.append(temp)
  flattened = [item for sublist in revise_key_words for item in sublist]
  # Convert every two elements into a sublist
  key_words_revised = [flattened[i:i+2] for i in range(0, len(flattened), 2)]
  key_words = []
  for keys in key_words_revised:
    key_words.append(','.join(keys))
  select_key_words = ','.join(add_key)
  question_list = []
  for q in question:
    if len(q) > 2:
      question_list.append(q)
  round = 0
  while round < len(key_words):
    select_key_words = key_words[round] + select_key_words
    now_num = len(select_key_words.split(','))
    vis_dict = {}
    while round < len(key_words) - 1:
      select_key_words = update_key(select_key_words, conv, vis_dict)
      if len(select_key_words.split(',')) == now_num:
        round += 1
        select_key_words = key_words[round] + select_key_words
      else:
        break
    doctor_prompt = f"Please role-play as a doctor and further generate questions, or conclusion, or the test result(such as medication test result or vital signs) based on the above dialogue and clinical note(after mentioned examination you have know test results and vital signs so you shouldn't ask patient about test result or vital signs). Add '\nDoctor:' before in each round. Your question, answer or conclusion(tell patient the test result) should only be around the key words(I gave you) corresponding to clinical note(finally the whole conversation should include all the key words but each time you should not include so much information. For example, you  should ask symptons one by one). And the answer of your questions can be found on the clinical note. You cannot modify these key words or use synonyms. You need to ensure the treatment plan, medication and dosage you give to the patient must also be totally consistent with the clinical note. Do not ask questions of which answer cannot be found in the clinical note. You may describe and explain professional judgment to the patient and instruct the patient on follow-up requirements, but not ask questions that require professional medical knowledge to answer. The order of the questions you ask must match the order of the keywords I provided. If it's not possible to include them all, you can make slight modifications based on the original wording in the notes. If the history conversation has included the key words there is no need to include them again. The treatment plan and conclusions you provide must align completely with the clinical notes. Do not add treatment plans that are not present in the clinical notes. You don't know the patient's medical history and symptoms. You should ask or lead patient to tell you the symptoms and his medical history and you don't have any information about his medical history and symptoms. All the information of medical history, symptoms, medication history and vaccination history should be told by patient. You can tell the patient the test results, vital signs and some conclusions. You shouldn't ask or mention the same information or question. You also shouldn't generate many words(under 30 words). you should follow up the  history conversation\n"   
    patient_prompt = f"Act as a patient to reply the doctor. Add '\nPatient:' before in each round. Your answer should align with the clinical notes. You are just an ordinary person, your response should be made as colloquial as possible. Don't mention any experimental results, conclusions or medical dosage. because you're just an ordinary person and may not understand the meaning of these results. But you could tell doctor your medical history, medication history or vaccination history(amedical history, medication history or vaccination history are all belong to medical history). Your response should revolve around the doctor's words and avoid adding information that was not mentioned."
    conv, conv_m = conversation(doctor_prompt, patient_prompt, history+conv_m, conv, conv_m, text, text, select_key_words, "")
    round += 1
    #ipdb.set_trace()
  word_list = list(cui_word.keys())

  fluence_prompt = """Expand the conversation. The conversation for patient parts can be more colloquial. When the doctor is speaking, the patient can have many modal particles (e.g. hmm, yes, okay) to increase interaction.
  All the numbers and medical concepts that appear in the note should be mentioned by the doctor.
  Professional medical terms and numbers should always occur in the doctor's utterances but not in the patient's answer. 
  The doctor may describe and explain professional judgment to the patient and instruct the patient on follow-up requirements, but not ask questions that require professional medical knowledge to answer.
  All the information of medical history, symptoms and medication history should be told by patient
  The patient's answer should be succinct and accurate in a colloquial lay language style. The answer should align with the clinical notes and as colloquial as possible.
  You can add some transitional phrases to make the conversation more logical. For example:
  Example 1:
  Patient: I understand, please go ahead.
  (After examination)
  Doctor: The result shows......
  Example 2:
  Patient: Thank you for the diagnosis, doctor.
  (After two years)
  Doctor: Hi....
  Example 3:
  Patient: Okay, I understand. 
  (Few days latter)
  Doctor: Hi....
  Your conversations must follow the logical sequence of a doctor's inquiry. For example, the general logical order of the conversation is: first discussing symptoms, then discussing the medical history, followed by discussing testing and results, and finally discussing treatment options, conclusioin etc."
  If you find this conversation to be incoherent, you can try dividing it into two separate coherent conversations.
  Patients should not say too much information at once.
  """
  messages_fluence = [{"role": "user", "content": fluence_prompt}]
  messages_fluence.append({"role": "user", "content": f'Clinical Note:\n{text}'})
  messages_fluence.append({"role": "user", "content": f'Conversation:\n{conv}'})
  messages_fluence.append({"role": "user", "content": f"Key Words:\n{','.join(word_list)}"})

  sample = pd.read_csv('TaskC-TrainingSet.csv')['dialogue'].loc[0]
  sample = sample.replace('[doctor]', 'Doctor:')
  sample = sample.replace('[patient]', 'Patient:')
  prompt = f"""
  There are only one patient and one doctor and just return the conversation. You conversation must include all the key words I gave you. 
  Your conversation should also include all information. if it's difficult to include them all, you can use the original sentences in the notes. 
  The common symptoms and common medical history should be told by patient. 
  Some specific symptoms and medical history should be added by the doctor after the patient has finished describing his symptoms and medical history.
  For example:
  Doctor: Can you give me your medical history record?
  Patient: Here you are.
  Doctor: Based on your medical history record...
  Because after patient has finished describing common symptoms or medical history, he will give doctor his medical history records. 
  After patient give the doctor his medical history record, the doctor could could know medical history record. Otherwise he didn't know any information of the medical history.
  Some result should not come from history clinical note they should come from examination.
  All the examination result, history examination result, vital sigh and medical number must be told by doctor.
  You could expand the parts of doctor to include more key words. If it is difficult to include you could just use the sentence of clinical note.
  The revised conversation should be at least around 80 to 150 utterances(doctor or patient should say too much information at once).
  The conversation must include all the information of the clinical note.
  You must include all the key words I gave you. If it is difficult to include all the key words you could use original the sentences of clinical note. 
  You cannot revise or eliminate any key words and you cannot use synonyms of the key words. 
  You shoudn't use the abbreviation if you know the full name(you should use full name not abbreviation, such as D9 must be day 9, D7 must be day 7. If both the full name and the abbreviation appear, it's better to use the full name rather than the abbreviation.
  Patients must not say any highly specialized terms, medical terminology or medical dosage. They can only describe limited common symptoms. The doctor should supplement the remaining information based on test results.
  Don't repeat the same information in long paragraphs. The utterance of the dialogue needs to be expanded as much as possible.
  Here is a good real dialogue example:
  {sample}
  the number of utterance should be at least 80 and sometimes patient didn't clearly hear and he could say parden to let the doctor say again.
  """
  messages_fluence.append({"role": "user", "content": prompt})
  min_len = 999
  final_conversation = ""
  fluence_conv = apply_chatgpt(messages_fluence, temperature=0.9)
  fluence_conv = re.sub('\n\n', '\n', fluence_conv)
  cui_note_word, cui_note_code, cui_note_entity = cui_code(text)
  cui_conv_word, cui_conv_code, cui_conv_entity = cui_code(fluence_conv)
  delete_key = diff(cui_conv_word, cui_note_word, cui_conv_entity, cui_note_code, text)
  add_key = diff(cui_note_word, cui_conv_word, cui_note_entity, cui_conv_code, fluence_conv)
  if len(add_key) < min_len:
    final_conversation = fluence_conv
    min_len = len(add_key)
  return final_conversation



def main():
  parser = argparse.ArgumentParser(description='index')
  parser.add_argument('--index', type=int, default = None)
  args = parser.parse_args()
  data = pd.read_csv('datasets/pmc-patient/data.csv')
  #data = pd.read_csv('TaskC-ValidationSet.csv')
  text = (str(args.index), 'MEDICATIONS', data['data'].loc[args.index])
  cui, _, _ = cui_code(data['data'].loc[args.index])
  conv = chat(text, max_epochs=min(50, max(len(data['data'].loc[args.index].split('.')), len(list(cui.keys())))))
  len_conv = len(conv.split('\n'))
  #file = open(f'./chat_conv/{args.index}.txt', 'w')
  
  file = open(f'./datasets/our_gpt4/{args.index}.txt', 'w')
  file.write(conv)
  file.close()


main()

