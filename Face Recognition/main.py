import cv2
import time
import os
from app import AssistantUI

DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR,'../Face Dataset')
# Initialize some variables
print("[INFO] warming up camera...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

app = AssistantUI(cap,DB_PATH)
app.videoLoop()
app.root.mainloop()

app.root.wm_title("Virtual Assistant")
app.root.wm_protocol("WM_DELETE_WINDOW",onClose)

def onClose():
    print("[INFO] Closing..")
    app.root.quit()
    cap.release()
