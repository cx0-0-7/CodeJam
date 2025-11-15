import cv2
import mediapipe as mp
import time


class handTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode 
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,          # Expected Boolean/Integer
            max_num_hands=self.maxHands,          # Expected Integer
            min_detection_confidence=self.detectionCon, # Expected Float (0.0 to 1.0)
            min_tracking_confidence=self.trackCon       # Expected Float (0.0 to 1.0)
        )
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            #for each hand detected
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand =self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                #get the pixel values
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList
        


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    while True:
        sucess, img = cap.read()
        img = tracker.findHands(img)
        lmList = tracker.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 30, 0), 3)


        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        try:
            if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
                print("Exiting via window close button ('X').")
                break
        except cv2.error:
            print("Exiting: Window was closed externally.")
            break

if __name__ == "__main__":
    main()