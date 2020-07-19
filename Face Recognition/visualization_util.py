import numpy as np
import cv2

class VisualizeFrame:
    def __init__(self):
        pass

    def display_bounding_boxes(self,frame,face_locations,face_names,face_ids):
        
        for (top, right, bottom, left), name,face_id in zip(face_locations, face_names,face_ids):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = str(face_id) + " : " + name
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        return frame