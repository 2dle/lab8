import cv2
import numpy as np

ref_image_path = 'd:/VScode/lab7/ref-point.jpg'
ref_image = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)

big_bro = cv2.SIFT_create()

kp_ref, des_ref = big_bro.detectAndCompute(ref_image, None)

def find_marker(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = big_bro.detectAndCompute(gray_frame, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_ref, des_frame)
    matches = sorted(matches, key = lambda x: x.distance)
    
    # Если достаточно совпадений, определяем местоположение метки. 
    # Можно поэксперементирвоать со значением совпадений, но после 25 у меня почти ничего не находило.
    # Решил, что наиболее оптимальное значение +-21, если взять сильно меньше, то маркер будет колбасить по всему экрану.
    if len(matches) > 21:
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        h, w = ref_image.shape
        pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        
        return np.int32(dst)
    return None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    marker = find_marker(frame)
    
    if marker is not None:
        center = np.mean(marker, axis=0).astype(int)
        if center[0][0] < 50 and center[0][1] < 50:
            color = (255, 0, 0) # угол
        elif center[0][0] > frame.shape[1] - 50 and center[0][1] > frame.shape[0] - 50:
            color = (0, 0, 255) # другой угол
        else:
            color = (0, 255, 0) 

        cv2.polylines(frame, [marker], True, color, 3, cv2.LINE_AA)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()