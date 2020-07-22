import face_recognition as fr
import numpy as np


class FaceRecognizer:
    def __init__(self,model='cnn'):
        self.model = model

    def recognize(self,img,known_face_encodings,known_face_names,known_ids):
        
        face_locations = fr.face_locations(img,model = self.model)
        face_encodings = fr.face_encodings(img, face_locations)
        ids = []
        face_names = []

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
                name = known_face_names[best_match_index]
                face_id = known_ids[best_match_index]
            face_names.append(name)
            ids.append(face_id)
        return face_names,ids,face_locations