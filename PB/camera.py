import cv2

color = (255,255,255)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_size = 1
y_offset = 100


cap =cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("error. now exiting.")
        break
    
    frame = cv2.resize(frame, (650, 400))
    frame = cv2.flip(frame, 1)
    
    cv2.imshow("Welcome to my Artbooth! Press 'S' to capture. Press 'Q' to quit the program!", frame)
    
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite("picture.png", frame)
        print("captured picture saved as 'picture.png'.")
    elif key == ord('q'):
        exit_booth = frame.copy()
        cv2.putText(exit_booth, "Thank you for testing Artbooth! Hope to see you again!", (50,y_offset), font, font_scale, color, font_size)
        cv2.imshow("Goodbye!", exit_booth)
        cv2.waitKey(3000)
        break
    
cap.release()
cv2.destroyAllWindows()