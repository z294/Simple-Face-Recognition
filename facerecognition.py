import cv2
import face_recognition
import os 
import sys


# check to see if it contains a input file 
if len(sys.argv) != 2:
    print("Usage: facerecognition.py <known faces folder path> ")
    sys.exit(1)
    

folder_path = sys.argv[1]


# Known faces
face_encodings = [] 
face_names = []

# Load known faces
for face_images in os.listdir(folder_path):
    #load only jpeg/lpg photo files into the list
    if face_images.endswith(".jpeg") or face_images.endswith(".jpg"):

        img_path = os.path.join(folder_path, face_images)
        # loads the image and encoding with it 
        face_rec_img = face_recognition.load_image_file(img_path)
        face_rec_encoding = face_recognition.face_encodings(face_rec_img)
        
        if face_rec_encoding:  # Check if any encoding was found to make sure only actual photos are used 
            face_encodings.append(face_rec_encoding[0])
            face_image_name = os.path.splitext(face_images)[0]
            face_names.append(face_image_name)

# Initialize webcam
video_cap = cv2.VideoCapture(0)

while True:
    # Capture frame by frame
    ret, frame = video_cap.read()

    # Find all faces in the current frame
    face_loc = face_recognition.face_locations(frame)
    live_face_encodings = face_recognition.face_encodings(frame, face_loc)

    if live_face_encodings:  # Check if any live face encodings were found
        for (top, right, bottom, left), face_encoding in zip(face_loc, live_face_encodings):
            matches = face_recognition.compare_faces(face_encodings, face_encoding)

            # Default to Unknown with red borders if the individual is unknown 
            name = "Unknown"
            box_color = (0, 0, 225)  # Red box for unknowns

            if True in matches:
                # Use a try-except block to handle potential index errors
                try:
                    first_match_idx = matches.index(True)
                    name = face_names[first_match_idx]
                    box_color = (0, 225, 0)  # Green box for recognized faces
                except IndexError:
                    continue  # Skip this iteration if there's an error for no name found for matched face

            # Draw a box around the face and label with the name
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 5)
            cv2.putText(frame, name, (left, top + 25), cv2.FONT_HERSHEY_PLAIN, 1.5, box_color, 4)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # End program using 'CTRL + q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

video_cap.release()
cv2.destroyAllWindows()
