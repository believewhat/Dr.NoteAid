
#env to run: /home/zhichaoyang/miniconda3/envs/t5long/bin/python
import sys
import csv, json
from collections import defaultdict

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




# get test
# csvfile = open('./tmp/raw_data/TaskA-TestSet.csv', 'r')
csvfile = open(sys.argv[1], 'r')
jsonfile = open(sys.argv[2], 'w')
jsonfile.write('TestID,SystemOutput1,SystemOutput2\n')
next(csvfile)
fieldnames = ("TestID","SystemOutput1","SystemOutput2")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    row["SystemOutput1"] = getheadername(row["SystemOutput1"])
    jsonfile.write(row["TestID"]+","+row["SystemOutput1"]+",\""+row["SystemOutput2"]+"\"")
    jsonfile.write('\n')
csvfile.close()
jsonfile.close()

