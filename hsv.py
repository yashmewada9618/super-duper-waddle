import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, image = cap.read()
    cv2.line(image, (200, 150), (400, 150), (255, 0, 0), 2)
    cv2.line(image, (200, 150), (200, 350), (255, 0, 0), 2)
    cv2.line(image, (400, 350), (200, 350), (255, 0, 0), 2)
    cv2.line(image, (400, 350), (400, 150), (255, 0, 0), 2)

    boundaries = [
        ([220,95,33], [227,77,86]),  # white
       ]

    # loop over the boundaries
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)
        bilateral_filtered_image = cv2.bilateralFilter(output, 5, 350, 350)
        edge_detected_image = cv2.Canny(bilateral_filtered_image, 85, 200)
        contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_list = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            if ((len(approx) > 8) & (len(approx) < 23) & (area > 30)):
                contour_list.append(contour)
                img = cv2.drawContours(output, contour_list, -1, (0, 0, 255), 2)
                if (200 < (area) and (area) < 400 and 150 < (area) and (area) < 350):
                    print('execute this=====================')
    print('000000')

    imageOut = np.hstack([image, output])

    # Display the resulting frame
    cv2.imshow('hsv', imageOut)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
