import cv2
import numpy as np

cap = cv2.VideoCapture("resources/redColor.mp4")
#cap = cv2.VideoCapture(0)

lowRed = np.array([161, 155, 84])
upRed = np.array([179, 255, 255])

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (540, 960))
    hsv_zey = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv_zey, lowRed, upRed)
    red = cv2.bitwise_and(frame, frame, mask = red_mask)
    red = cv2.resize(red, (540, 960))
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = 0, 0, 0, 0
    if contours:
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y), ((x+w), (y+h)), (255, 0, 0), 2)
                break


        x_c = ((2 * x) + w) / 2
        y_c = ((2 * y) + h) / 2
        center = (x_c, y_c)
        cv2.circle(frame, (int(x_c), int(y_c)), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(red, (int(x_c), int(y_c)), 5, (255, 0, 0), cv2.FILLED)

        print("[INFO].. center is calculated", center)

    cv2.imshow("mask", red)
    cv2.imshow("red", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()