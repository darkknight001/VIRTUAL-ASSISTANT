import pandas as pd
import csv
import os
from collections import OrderedDict

DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR,'../Face Dataset')
class FaceData:
    def __init__(self,file_name="face_db.csv"):
        self.file_path = os.path.join(DB_PATH,file_name)

    def get_face_count(self):
        try:
            with open(self.file_path,'r',newline='') as file:
                reader = csv.reader(file)
                row_count=len(list(reader))
                return row_count
        except FileNotFoundError as e:
            print("Unable to read file,error: "+str(e))
            return None

    def get_face_data(self):
        face_data = OrderedDict()
        if self.get_face_count()==None:
            return None
        with open(self.file_path,'r',newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                face_data[row[0]] = row[1]
            return face_data
            #{id : [name]}      
    
    def save_new_face(self,new_name):
        last_id = self.get_face_count()
        if last_id==None:
            last_id=-1
            os.system(r'touch /home/darkknight/Desktop/Project/VIRTUAL ASSISTANT/Face Dataset/face_db.csv')
        with open(self.file_path,'a',newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow([last_id+1,new_name])
