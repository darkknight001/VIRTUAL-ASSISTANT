import numpy as np
import cv2
import face_recognition as fr
from data_util import FaceData
import time
import os
from visualization_util import VisualizeFrame


DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR,'../Face Dataset')
cap = cv2.VideoCapture(0)
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
face_data = FaceData()
frame_util = VisualizeFrame()

#Get data from Database
def get_face_id():
    id = list(face_data.get_face_data().keys())
    return id

def get_face_names():
    known_names = [name for name in face_data.get_face_data().values()]
    return known_names

def get_face_enc():
    db_img = [fr.load_image_file(os.path.join(DB_PATH,names+'.jpg')) for names in get_face_names()]
    known_encodings = [fr.face_encodings(img)[0] for img in db_img]
    return known_encodings


known_face_encodings = get_face_enc()

while True:
    # Grab a single frame of video
    ret, frame = cap.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = fr.face_locations(rgb_small_frame,model = 'cnn')
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
        ids = []
        face_names = []

        
        if face_data.get_face_count()==None:
            print("Here")
            no_db = True
        
        else:
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = fr.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_id = None
                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = fr.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = get_face_names()[best_match_index]
                    face_id = get_face_id()[best_match_index]
                face_names.append(name)
                ids.append(face_id)
            if "Unknown" in face_names:
                print(face_locations,type(face_locations))
                new_name = input("New Face Detected, Input a name:  ")
                # (new_top, new_right, new_bottom, new_left) = (int(0.8 * face_locations[0][0]), int(1.2* face_locations[0][1]), int(1.2*face_locations[0][2]), int(0.8*face_locations[0][3]))
                img_name = DB_PATH + '/'+new_name+'.jpg'
                cv2.imwrite((img_name),frame)
                # frame[new_top:new_bottom, new_left:new_right]
                face_data.save_new_face(new_name)
                known_face_encodings = get_face_enc()
    process_this_frame = not process_this_frame

    frame = frame_util.display_bounding_boxes(frame=frame,
                                            face_locations=face_locations,
                                            face_names=face_names,
                                            face_ids=ids)
    # Display the resulting image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break   

cap.release()
cv2.destroyAllWindows()