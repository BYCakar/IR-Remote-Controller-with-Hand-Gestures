import numpy as np
import cv2
import math
import time
import os


def getMaxYContour(contour: np.ndarray):
    maxX = contour[np.argmax(contour[:, :, 0])][0][0]
    minX = contour[np.argmin(contour[:, :, 0])][0][0]

    maxY = list()
    for i in range(maxX - minX + 1):
        maxY.append([[minX + i, 200]])

    appendContour: list = contour.tolist()

    for i in range(contour.shape[0] - 1, -1, -1):
        fill = contour[i][0][0] - contour[i - 1][0][0]
        j = 0
        while j < abs(fill) - 1:
            x0 = contour[i - 1][0][0]
            x1 = contour[i][0][0]
            y0 = contour[i - 1][0][1]
            y1 = contour[i][0][1]

            m = (y1 - y0) / (x1 - x0)
            n = (x1 * y0 - x0 * y1) / (x1 - x0)

            fillX = x1 - np.sign(fill) * (j + 1)
            fillY = int(m * fillX + n)

            appendContour.insert(i, [[fillX, fillY]])
            j += 1

    contour = np.array(appendContour)

    for i in range(contour.shape[0]):
        maxY[contour[i][0][0] - minX][0][1] = min(contour[i][0][1], maxY[contour[i][0][0] - minX][0][1])

    for i in range(maxX - minX, 0, -1):
        if maxY[i][0][1] == 200:
            maxY.pop(i)

    maxY.insert(0, [[minX, 200]])
    maxY.append([[maxX, 200]])
    maxY = np.array(maxY)
    maxY = np.array(maxY[np.argmin(maxY[:, :, 1]):maxY.shape[0]].tolist() + maxY[0:np.argmin(maxY[:, :, 1])].tolist())
    return maxY


def getExtremes(contour: np.ndarray):
    contour = getMaxYContour(contour)
    lastydir = np.sign(contour[0][0][1] - contour[-1][0][1])
    minimum = list()
    maximum = list()
    for i in range(1, contour.shape[0]):
        ydir = np.sign(contour[i][0][1] - contour[i - 1][0][1])

        if (ydir == 1 and lastydir == -1) or (ydir != 0 and lastydir == 0):
            maximum.append(contour[i - 1][0])
            lastydir = ydir
        elif (ydir == -1 and lastydir == 1) or (ydir != 0 and lastydir == 0):
            if contour[i - 1][0][1] < 200:
                minimum.append(contour[i - 1][0])
            lastydir = ydir

    return (np.array(minimum), np.array(maximum))


os.system("sudo systemctl restart lircd")
# Open Camera
capture = cv2.VideoCapture(0)

state = "IDLE"
doneCommandFlag = 0

# Initialize FPS calculating variables
lastSampleTime = time.time()
count = 0
fps = 0

while capture.isOpened():

    # Capture frames from the camera
    ret, frame = capture.read()
    frame = cv2.flip(frame, 2)

    # Get hand data from the rectangle sub window
    cv2.rectangle(frame, (300, 100), (600, 300), (255, 0, 0), 1)
    crop_image = frame[100:300, 300:600]

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([0, 28, 40]), np.array([40, 255, 255]))

    # Kernel for morphological transformation
    kernel = np.ones((5, 5))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    # Show threshold image
    cv2.imshow("Thresholded Hemal", thresh)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:

        # Find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Find convex hull
        hull = cv2.convexHull(contour)

        # Find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
        # tips) for all defects
        far = list()

        for i in range(defects.shape[0]):
            _, _, f, _ = defects[i, 0]
            far.append(tuple(contour[f][0]))

        farToContour = np.array(np.reshape(far, (len(far), 1, 2)))

        M = cv2.moments(farToContour)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        else:
            cX, cY = 0, 0

        # cv2.circle(crop_image, (cX, cY), 65, (255, 0, 0), 3)

        count_fingertips = 0
        indexFinger = 0
        thumb = 0
        minimum, maximum = getExtremes(contour)

        rad = min(int(
            max([math.sqrt((minimum[i][0] - cX) ** 2 + (minimum[i][1] - cY) ** 2) for i in range(len(minimum))])) + 15,
                  65) if (len(minimum) != 0 and state == "THUMB UP") else 65
        cv2.circle(crop_image, (cX, cY), rad, (0, 255, 255), 3)
        # cv2.circle(crop_image, (cX, cY), 65, (255, 0, 0), 3)
        for i in range(minimum.shape[0]):
            cv2.circle(crop_image, tuple(minimum[i]), 3, [0, 0, 255], -1)
        for i in range(maximum.shape[0]):
            cv2.circle(crop_image, tuple(maximum[i]), 3, [0, 255, 0], -1)
            dist = math.sqrt((maximum[i][0] - cX) ** 2 + (maximum[i][1] - cY) ** 2)
            if dist > rad:
                count_fingertips += 1
                arccos = math.acos((maximum[i][0] - cX) / dist)
                angle = arccos / math.pi * 180 if maximum[i][1] - cY < 0 else 360 - arccos / math.pi * 180
                if 135 < angle < 225:
                    cv2.putText(crop_image, "THUMB", tuple(maximum[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                                1)
                    thumb = 1
                elif 90 < angle < 135:
                    cv2.putText(crop_image, "INDEX", tuple(maximum[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                                1)
                    indexFinger = 1

        if state == "IDLE":
            if count_fingertips == 0:
                cv2.putText(frame, "IDLE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                lastIdleTime = time.time()
                doneCommandFlag = 0
            elif doneCommandFlag == 1:
                cv2.putText(frame, "RETURN IDLE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            elif count_fingertips == 1 and indexFinger == 1:
                state = "INDEX FINGER UP"
                cv2.putText(frame, "INDEX FINGER UP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                lastIndexFingerAngle = angle
                isSetVolume = 0
                keepSending = 0
            elif count_fingertips == 1 and thumb == 1:
                state = "THUMB UP"
                cv2.putText(frame, "THUMB UP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                keepSending = 0
                lastcX = cX
            elif count_fingertips == 1:
                cv2.putText(frame, "WAITING FOR ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                if time.time() - lastIdleTime > 1:
                    feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_1").read()
                    while feedback.find("repeating") == 0:
                        os.system("sudo systemctl restart lircd")
                        print("Error detected! Restarting lircd and retrying to send signal")
                        feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_1").read()
                    print("Select Ch. 1 signal sent")
                    doneCommandFlag = 1
            elif count_fingertips == 2 and indexFinger * thumb == 1:
                cv2.putText(frame, "WAITING FOR MUTE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                if time.time() - lastIdleTime > 1:
                    feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_MUTE").read()
                    while feedback.find("repeating") == 0:
                        os.system("sudo systemctl restart lircd")
                        print("Error detected! Restarting lircd and retrying to send signal")
                        feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_MUTE").read()
                    print("Mute signal sent")
                    doneCommandFlag = 1
            elif count_fingertips == 2:
                cv2.putText(frame, "WAITING FOR TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                if time.time() - lastIdleTime > 1:
                    feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_2").read()
                    while feedback.find("repeating") == 0:
                        os.system("sudo systemctl restart lircd")
                        print("Error detected! Restarting lircd and retrying to send signal")
                        feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_2").read()
                    print("Select Ch. 2 signal sent")
                    doneCommandFlag = 1
            elif count_fingertips == 3:
                cv2.putText(frame, "WAITING FOR THREE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                if time.time() - lastIdleTime > 1:
                    feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_3").read()
                    while feedback.find("repeating") == 0:
                        os.system("sudo systemctl restart lircd")
                        print("Error detected! Restarting lircd and retrying to send signal")
                        feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_3").read()
                    print("Select Ch. 3 signal sent")
                    doneCommandFlag = 1
            elif count_fingertips == 4:
                cv2.putText(frame, "WAITING FOR FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                if time.time() - lastIdleTime > 1:
                    feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_4").read()
                    while feedback.find("repeating") == 0:
                        os.system("sudo systemctl restart lircd")
                        print("Error detected! Restarting lircd and retrying to send signal")
                        feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_4").read()
                    print("Select Ch. 4 signal sent")
                    doneCommandFlag = 1
            elif count_fingertips == 5:
                state = "OPEN HAND"
                cv2.putText(frame, "WAITING FOR FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                blink = 0

        elif state == "INDEX FINGER UP":
            if count_fingertips != 1:
                state = "IDLE"
                if keepSending > 0:
                    keepSending = 0
                    os.system("irsend SEND_STOP LG_Remote_Controller KEY_VOLUMEUP")
                    os.system("irsend SEND_STOP LG_Remote_Controller KEY_VOLUMEDOWN")
                    print("Volume signal stopped sending")
            elif lastIndexFingerAngle - angle > 15:
                cv2.putText(frame, "VOLUME UP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                isSetVolume = 1
                doneCommandFlag = 1
                if keepSending == 0:
                    lastIdleTime = time.time()
                    keepSending = 1
                elif keepSending == 1 and time.time() - lastIdleTime > 1:
                    lastIdleTime = time.time()
                    keepSending += 1
                    feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_VOLUMEUP").read()
                    while feedback.find("repeating") == 0:
                        os.system("sudo systemctl restart lircd")
                        print("Error detected! Restarting lircd and retrying to send signal")
                        feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_VOLUMEUP").read()
                    print("Volume up signal sent")
                elif keepSending == 2 and time.time() - lastIdleTime > 1:
                    lastIdleTime = time.time()
                    keepSending += 1
                    os.system("irsend SEND_START LG_Remote_Controller KEY_VOLUMEUP")
                    print("Volume up signal started sending")

            elif angle - lastIndexFingerAngle > 15:
                cv2.putText(frame, "VOLUME DOWN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                isSetVolume = 1
                doneCommandFlag = 1
                if keepSending == 0:
                    lastIdleTime = time.time()
                    keepSending = 1
                elif keepSending == 1 and time.time() - lastIdleTime > 1:
                    lastIdleTime = time.time()
                    keepSending += 1
                    feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_VOLUMEDOWN").read()
                    while feedback.find("repeating") == 0:
                        os.system("sudo systemctl restart lircd")
                        print("Error detected! Restarting lircd and retrying to send signal")
                        feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_VOLUMEDOWN").read()
                    print("Volume down signal sent")
                elif keepSending == 2 and time.time() - lastIdleTime > 1:
                    lastIdleTime = time.time()
                    keepSending += 1
                    os.system("irsend SEND_START LG_Remote_Controller KEY_VOLUMEDOWN")
                    print("Volume down signal started sending")
            elif time.time() - lastIdleTime > 1 and isSetVolume != 1:
                state = "IDLE"
                feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_1").read()
                while feedback.find("repeating") == 0:
                    os.system("sudo systemctl restart lircd")
                    print("Error detected! Restarting lircd and retrying to send signal")
                    feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_1").read()
                print("Select Ch. 1 signal sent")
                doneCommandFlag = 1
            elif keepSending > 0:
                keepSending = 0
                os.system("irsend SEND_STOP LG_Remote_Controller KEY_VOLUMEUP")
                os.system("irsend SEND_STOP LG_Remote_Controller KEY_VOLUMEDOWN")
                print("Volume signal stopped sending")
            else:
                cv2.putText(frame, "INDEX FINGER UP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        elif state == "THUMB UP":
            if count_fingertips > 3 or thumb != 1 or indexFinger == 1:
                state = "IDLE"
            elif lastcX - cX > 25:
                cv2.putText(frame, "CHANNEL DOWN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                doneCommandFlag = 1
                if keepSending == 0:
                    lastIdleTime = time.time()
                    keepSending = 1
                elif time.time() - lastIdleTime > 1:
                    lastIdleTime = time.time()
                    feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_CHANNELDOWN").read()
                    while feedback.find("repeating") == 0:
                        os.system("sudo systemctl restart lircd")
                        print("Error detected! Restarting lircd and retrying to send signal")
                        feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_CHANNELDOWN").read()
                    print("Channel down signal sent")
            elif cX - lastcX > 25:
                cv2.putText(frame, "CHANNEL UP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                doneCommandFlag = 1
                if keepSending == 0:
                    lastIdleTime = time.time()
                    keepSending = 1
                elif time.time() - lastIdleTime > 1:
                    lastIdleTime = time.time()
                    feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_CHANNELUP").read()
                    while feedback.find("repeating") == 0:
                        os.system("sudo systemctl restart lircd")
                        print("Error detected! Restarting lircd and retrying to send signal")
                        feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_CHANNELUP").read()
                    print("Channel up signal sent")
            else:
                cv2.putText(frame, "THUMB UP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                keepSending = 0
        elif state == "OPEN HAND":
            if time.time() - lastIdleTime > 3:
                state = "IDLE"
                doneCommandFlag = 1
            elif time.time() - lastIdleTime > 1 and count_fingertips == 5 and blink == 0:
                state = "IDLE"
                print("Select Ch. 5 signal sent")
                feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_5").read()
                while feedback.find("repeating") == 0:
                    os.system("sudo systemctl restart lircd")
                    print("Error detected! Restarting lircd and retrying to send signal")
                    feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_5").read()
                doneCommandFlag = 1
            elif blink == 5:
                state = "IDLE"
                feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_POWER").read()
                while feedback.find("repeating") == 0:
                    os.system("sudo systemctl restart lircd")
                    print("Error detected! Restarting lircd and retrying to send signal")
                    feedback = os.popen("irsend SEND_ONCE LG_Remote_Controller KEY_POWER").read()
                print("Shutdown signal sent")
            elif blink % 2 == 0 and count_fingertips == 0:
                blink = blink + 1
                lastIdleTime = time.time()
            elif blink % 2 == 1 and count_fingertips == 5:
                blink = blink + 1
                lastIdleTime = time.time()
            else:
                cv2.putText(frame, "WAITING FOR FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        # Show and calculate FPS
        cv2.putText(frame, "FPS: " + str(fps), (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        count += 1

        if time.time() - lastSampleTime > 1:
            lastSampleTime = time.time()
            fps = count
            count = 0
    except:
        pass

    # Show required images
    cv2.imshow("Gesture", frame)

    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

