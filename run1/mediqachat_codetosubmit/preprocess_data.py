
#env to run: /home/zhichaoyang/miniconda3/envs/t5long/bin/python
import sys
import csv, json
from collections import defaultdict

def getheadername(header_short):
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
    return short2full[header_short.lower()].lower()


# get train
csvfile = open('./tmp/raw_data/TaskA-TrainingSet.csv', 'r')
jsonfile = open('./tmp/preprocessed_data/train.json', 'w')

header2count = defaultdict(int)
fieldnames = ("id","section_header","section_text","dialogue")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    row["section_header"] = getheadername(row["section_header"])
    header2count[row["section_header"]] += 1
    json.dump(row, jsonfile)
    jsonfile.write('\n')
csvfile.close()
jsonfile.close()

# get valid
csvfile = open('./tmp/raw_data/TaskA-ValidationSet.csv', 'r')
jsonfile = open('./tmp/preprocessed_data/valid.json', 'w')

fieldnames = ("id","section_header","section_text","dialogue")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    row["section_header"] = getheadername(row["section_header"])
    json.dump(row, jsonfile)
    jsonfile.write('\n')
csvfile.close()
jsonfile.close()


# get test
# csvfile = open('./tmp/raw_data/TaskA-TestSet.csv', 'r')
csvfile = open(sys.argv[1], 'r')
jsonfile = open('./tmp/preprocessed_data/test.json', 'w')
next(csvfile)
fieldnames = ("id","dialogue")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')
csvfile.close()
jsonfile.close()

