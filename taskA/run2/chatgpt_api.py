import os
import sys
import re
import pandas as pd
import numpy as np
import csv
import ujson, json
from collections import defaultdict
import pandas as pd
import ipdb
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import string
import time
import random
import openai

openai.api_key = ""

def apply_chatgpt(messages, temperature=0.7, max_tokens=2000, presence_penalty=1.5):
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=max_tokens, 
    temperature=temperature,
    presence_penalty=presence_penalty
  )
  content = completion.choices[0].message.content
  return content

"""
sectionhead2indexs = {'GENHX': 0, 
    'ROS': 1, 'PASTMEDICALHX': 2, 
    'MEDICATIONS': 3, 'CC': 4, 'PASTSURGICAL': 6, 
    'FAM_SOCHX': 38, 'DISPOSITION': 12, 'DIAGNOSIS': 20,
    'EDCOURSE': 22, 'PLAN': 33, 'LABS': 36, 'ASSESSMENT': 41, 
    'ALLERGY': 61, 'GYNHX': 74, 'EXAM': 91, 'OTHER_HISTORY': 94, 'PROCEDURES': 95, 
    'IMAGING': 97, 'IMMUNIZATIONS': 98}
df = pd.read_csv('TaskA-ValidationSet.csv')


sectionhead2indexs = {'GENHX': 0, 
    'ROS': 1, 'PASTMEDICALHX': 2, 
    'MEDICATIONS': 3, 'CC': 4, 'PASTSURGICAL': 6, 
    'FAM_SOCHX': 38, 'DISPOSITION': 12, 'DIAGNOSIS': 20,
    'EDCOURSE': 22, 'PLAN': 33, 'LABS': 36, 'ASSESSMENT': 41, 
    'ALLERGY': 61, 'GYNHX': 74, 'EXAM': 91, 'OTHER_HISTORY': 94, 'PROCEDURES': 95, 
    'IMAGING': 97, 'IMMUNIZATIONS': 98}
prompt_base = "Determine which of the following categories the content of the following conversation is about:\n\nCategory:\
ALLERGY,ASSESSMENT,CC,DIAGNOSIS,DISPOSITION,EDCOURSE,EXAMFAM/SOCHX,GENHX,GYNHX,IMAGING,IMMUNIZATIONS,LABSMEDICATIONS,OTHER_HISTORY,PASTMEDICALHX,PASTSURGICAL,PLANPROCEDURES,ROS\
\n\nFor Example:"
for key, value in sectionhead2indexs.items():
  prompt_base += "\n\nConversation:\n\n" + df['dialogue'].loc[value] + "\n\nYour Answer:\n\n" + df['section_header'].loc[value]
file = open('prompt_headers.txt', 'w')
file.write(prompt_base)
file.close()

"""
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
    "exam": "exam",
    "diagnosis": "diagnosis",
    "disposition": "disposition",
    "plan": "PLAN",
    "edcourse": "EMERGENCY DEPARTMENT COURSE",
    "immunizations": "immunizations",
    "imaging": "imaging",
    "gynhx": "GYNECOLOGIC HISTORY",
    "procedures": "procedures",
    "other_history": "other_history",
    "labs": "labs"
}

sectionhead2indexs = {'history of present illness': [437, 444, 825, 566, 1117], 'medications': [643, 903, 173, 238, 1084], 'chief complaint': [743, 789, 994, 518, 26], 'past medical history': [836, 815, 130, 39, 368], 'allergy': [7, 596, 154, 23, 134], 'family history/social history': [592, 230, 298, 271, 349], 'past surgical history': [398, 63, 621, 415, 1040], 'other_history': [21, 950], 'assessment': [688, 889, 372, 767, 25], 'review of systems': [453, 1029, 451, 604, 585], 'disposition': [288, 1183, 260, 151, 603], 'exam': [1076, 48, 333, 911, 430], 'plan': [819, 1129, 248, 266, 805], 'diagnosis': [528, 275, 224, 1147, 169], 'emergency department course': [234, 88, 274, 706, 520], 'immunizations': [1178, 448, 795, 833, 752], 'labs': [108, 1160], 'imaging': [650, 1088, 1143, 576, 774], 'procedures': [229, 920, 1073], 'gynecologic history': [310, 404, 495, 584, 754]}
header_index = dict()
for key, value in short2full.items():
  header = re.sub("/", "_", key)
  header_index[header.upper()] = sectionhead2indexs[value.lower()]

with open('prompt_headers.txt','r',encoding='utf-8') as f:
    header_prompt = f.read()


def main():
    df = pd.read_csv('taskA_testset4participants_inputConversations.csv')
    header_predict = pd.read_csv('taskA_UMASS_BioNLP_run1.csv')
    df['section_header_chat'] = np.zeros(df.shape[0])
    df['SystemOutput2'] = np.zeros(df.shape[0])
    df['SystemOutput1'] = header_predict['SystemOutput1']
    for i in range(df.shape[0]):
      try:
        header = df['SystemOutput1'].loc[i]
        header = re.sub("/","_", header)
        with open(f'./prompts/{header}/prompt_note.txt','r',encoding='utf-8') as f:
          prompt = f.read()
        messages = [{"role": "user", "content": prompt + df['dialogue'].loc[i] + '\n\n Now please generate brief and short clinical notes:\n'}]
        messages_header = [{"role": "user", "content": header_prompt + "\n\nNow choose one category that you think is correct, here is the conversation: \n\nConversation:\n\n" + df['dialogue'].loc[i]}]
        chat_header = apply_chatgpt(messages_header, temperature=0,presence_penalty=0,max_tokens=20)
        chat_header = '\n' + chat_header + '\n'
        if len(re.findall(r'\n([A-Z_\x20]+)\n',chat_header)) > 0:
          df['section_header_chat'].loc[i] = re.findall(r'\n([A-Z_\x20]+)\n',chat_header)[-1]
        else:
          df['section_header_chat'].loc[i] = 'GENHX'
        df['SystemOutput2'].loc[i] = apply_chatgpt(messages, temperature=0.5,presence_penalty=1,max_tokens=1000)
        df.to_csv('taskA_UMASS_BioNLP_run2.csv', index=False)
      except:
        ipdb.set_trace()
main()



