import pandas as pd
import csv
import os
from collections import OrderedDict
import face_recognition as fr

DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR,'../Face Dataset')
class FaceData:
    def __init__(self,file_name="face_db.csv"):
        self.file_path = os.path.sep.join((DB_PATH,file_name))

    def check_database(self):
        try:
            with open(self.file_path,'r',newline='') as file:
                reader = csv.reader(file)
        except FileNotFoundError:
            print("Problems with your Dataset, creating a new one!")
            os.system('touch "/home/darkknight/Desktop/Project/VIRTUAL ASSISTANT/Face Dataset/face_db.csv"')

            
    def get_face_count(self):
        self.check_database()
        with open(self.file_path,'r',newline='') as file:
            reader = csv.reader(file)
            row_count=len(list(reader))
            return row_count

    def get_face_data(self):
        face_data = OrderedDict()
        self.check_database()   
        with open(self.file_path,'r',newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                # print(row)
                face_data[row[0]] = row[1]
            return face_data
            #{id : [name]}      
    
    def get_face_id(self):
        id = list(self.get_face_data().keys())
        return id

    def get_face_names(self):
        known_names = [name for name in self.get_face_data().values()]
        return known_names

    def get_face_enc(self):
        # print(get_face_names())
        db_img = [fr.load_image_file(DB_PATH+'/'+name+'.jpg') for name in self.get_face_names()]
        print(len(db_img))  
        # print(type(fr.face_encodings(db_img[0])[0]))
        known_encodings = [fr.face_encodings(img)[0] for img in db_img]
        return known_encodings

    def save_new_face(self,new_name):
        last_id = self.get_face_count()
        with open(self.file_path,'a',newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow([last_id,new_name])
