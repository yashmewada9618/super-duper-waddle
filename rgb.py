import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    ret, image = cap.read()
    # imag = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.line(image, (200, 150), (400, 150), (255, 0, 0), 2)
    cv2.line(image, (200, 150), (200, 350), (255, 0, 0), 2)
    cv2.line(image, (400, 350), (200, 350), (255, 0, 0), 2)
    cv2.line(image, (400, 350), (400, 150), (255, 0, 0), 2)
    boundaries = [
        # ([230, 230, 230], [250, 250, 250]),  # white
        # ([17, 15, 100], [50, 56, 200]),  # red
        ([85, 35, 9], [220, 88, 50]),  # blue
        # ([25, 146, 190], [62, 174, 250 ]),  # yellow
    ]

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        bilateral_filtered_image = cv2.bilateralFilter(output, 5, 350, 350)
        edge_detected_image = cv2.Canny(bilateral_filtered_image, 85, 200)
        contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours = imutils.grab_contours(contours)
        contour_list = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                # set values as what you need in the situation
                cX, cY = 0, 0
            area = cv2.contourArea(c)
            if area > 2400:
                cv2.drawContours(output, [c], -1, (0, 255, 0), 2)
                cv2.circle(output, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(output, "center", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            if (len(approx) > 8) & (len(approx) < 23) & (area > 30):
                contour_list.append(contour)
                if cX < 200:
                    print("right###########")
                    while 200 < cX and cX < 400 and 150 < cY and cY < 350:
                        print('*****aligned**************')

    print('000000')
    imageOut = np.hstack((image, output))
    cv2.imshow('RGB', imageOut)
    # cv2.imshow('bilateral',edge_detected_image)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
