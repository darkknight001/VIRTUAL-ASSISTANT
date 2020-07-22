import tkinter as tki
from PIL import Image, ImageTk
import threading
import datetime
import imutils
import cv2
import os
import numpy as np
from visualization_util import VisualizeFrame
from fr_utils import FaceRecognizer
import face_recognition as fr
from data_util import FaceData


DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR,'../Face Dataset')


class AssistantUI:
    def __init__(self,vs,database_path=DB_PATH):
        self.vs = vs
        self.stream = None
        self.frame = None
        self.thread = None
        self.newName = None 

        self.root = tki.Tk()
        self.panel = None
        self.processFrame = True

        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.ids = []

        self.frame_util = VisualizeFrame()
        self.recognizer = FaceRecognizer(model='hog')
        self.face_data = FaceData()

        self.known_face_encodings = self.face_data.get_face_enc()
        # btnStart = tki.Button(self.root,
        #                       text="Start Prediction",
        #                       command=self.startPrediction)
        # btnStart.pack(side='bottom',fill='both',expand='yes',padx=10,pady=10)
        self.nameEntry = tki.Entry(self.root,borderwidth=5)
        self.nameEntry.pack(side='bottom',fill='both',expand='yes',padx=10,pady=10)

        btnNewFace = tki.Button(self.root,
                              text="Capture New Face",
                              command=self.captureNewFace)
        btnNewFace.pack(side='bottom',fill='both',expand='yes',padx=10,pady=10)

        # self.streamEvent = threading.Event()
        # self.streamEvent.set()
        
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        self.last_frame = None
        self.last_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def videoLoop(self):

        _,self.frame = self.vs.read()
        
        self.frame = cv2.flip(self.frame,1)
        self.last_frame = self.frame.copy()
        
        self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(self.frame, (0, 0), fx=0.25, fy=0.25)


        
        if self.processFrame:
            self.ids = []
            self.face_names = []
            self.face_names,self.ids,self.face_locations = self.recognizer.recognize(img=small_frame,
                                known_face_encodings=self.known_face_encodings,
                                known_face_names=self.face_data.get_face_names(),
                                known_ids=self.face_data.get_face_id())
        
        
        # if "Unknown" in face_names:
        #     new_name = input("New Face Detected, Input a name:  ")
        #     # (new_top, new_right, new_bottom, new_left) = (int(0.8 * face_locations[0][0]), int(1.2* face_locations[0][1]), int(1.2*face_locations[0][2]), int(0.8*face_locations[0][3]))
        #     img_name = DB_PATH + '/'+new_name+'.jpg'
        #     cv2.imwrite((img_name),frame)
        #     # frame[new_top:new_bottom, new_left:new_right]
        #     face_data.save_new_face(new_name)
        #     known_face_encodings = get_face_enc()
    
        self.processFrame = not self.processFrame

        self.frame = self.frame_util.display_bounding_boxes(frame=self.frame,
                                                face_locations=self.face_locations,
                                                face_names=self.face_names,
                                                face_ids=self.ids)

        img = Image.fromarray(self.frame)
        img = ImageTk.PhotoImage(img)
        
        # Display the resulting image
        if self.panel is None:
            print(type(img))
            self.panel = tki.Label(image = img)
            self.panel.image = img
            self.panel.pack(side='left',
                            padx = 10,
                            pady=10)
        else:
            self.panel.configure(image=img)
            self.panel.after(10,self.videoLoop)
            self.panel._image_cache = img

    def captureNewFace(self):

        self.newName = self.nameEntry.get()     
        fileName = self.newName+'.jpg'  
        p = os.path.sep.join((DB_PATH,fileName))
        cv2.imwrite(p,self.last_frame)
        print(f"[INFO] Image saved as {p}")
        self.face_data.save_new_face(self.newName)
        self.known_face_encodings = self.face_data.get_face_enc()
    
    # def startPrediction(self):
    
    

