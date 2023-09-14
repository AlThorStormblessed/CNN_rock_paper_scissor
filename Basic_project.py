#importing libraries
import cv2
import numpy as np
import tensorflow as tf
import random


def get_class_arg(array):
    return np.argmax(array)

def get_class(argument):
    if argument == 0:
        return "Rock"
    elif argument == 1:
        return "Paper"
    elif argument == 2:
        return "Scissors"
    else:
        return "Empty"

#loading the trained model
model = tf.keras.models.load_model('saved_model/model_t')

#video capture and model prediction
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
cv2.namedWindow("Rock paper scissors!")

played_move = 0
n = 5
pc_move = random.choice(["Rock", "Paper", "Scissors"])
scores = [0, 0]

frame_width = int(cap.get(3)) + 60
frame_height = int(cap.get(4)) + 60
   
size = (frame_width, frame_height)

result = cv2.VideoWriter('rock_paper_scissors.mp4', 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         10, size)

rock_img = cv2.imread("/Rock_img.jpg")
paper_img = cv2.imread("/paper_img.jpg")
scissor_img = cv2.imread("/Scissor_img.jpg")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (100, 130), (500, 530), (255, 255, 255), 2)
    cv2.rectangle(frame, (800, 130), (1200, 530), (0, 0, 0), 2)

    frame = cv2.copyMakeBorder(frame,30,30,30,30,cv2.BORDER_CONSTANT,value=[0,0,0])

    # extract the region of image within the user rectangle
    roi = frame[130:530, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img=img/255.0 #normalizing pixel values between 0 and 1

    # predict the move made
    pred = model.predict(np.array([img]))
    pred_num = get_class_arg(pred)
    move=get_class(pred_num)
    
    if(played_move < n and pred_num < 3):
        played_move += 1
        if played_move >= n:
            pc_move = random.choice(["Rock", "Paper", "Scissors"])
            if(move == "Rock" and pc_move == "Paper") or (move == "Paper" and pc_move == "Scissors") or (move == "Scissors" and pc_move == "Rock"):
                scores[0] += 1
                text = "Lose"
            elif(pc_move == "Rock" and move == "Paper") or (pc_move == "Paper" and move == "Scissors") or (pc_move == "Scissors" and move == "Rock"):
                scores[1] += 1
                text = "Win"
            else:
                text = "Tie"

    if played_move >= n:
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (930, 280)
        fontScale = 0.8
        color = (0, 100, 100)
        thickness = 2
        if pc_move == "Rock":
            target_img = rock_img
        elif pc_move == "Paper":
            target_img = paper_img
        else:
            target_img = scissor_img

        target_img = cv2.resize(target_img, (400, 400))
        img2gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        if ret:
            frame[160:560, 830:1230] = target_img
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (1180, 100)
        fontScale = 0.8
        color = (255, 0, 0)
        thickness = 2
        frame = cv2.putText(frame, pc_move, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (600, 330)
        fontScale = 3
        color = (0, 0, 200)
        thickness = 2
        frame = cv2.putText(frame, text, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        
    if(played_move and pred_num == 3):
        played_move = 0
        
    #adding text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (80, 100)
    # fontScale
    fontScale = 0.8
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    frame = cv2.putText(frame, move, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (550, 640)
    fontScale = 3
    color = (0, 0, 200)
    thickness = 5
    frame = cv2.putText(frame, f"{scores[1]} - {scores[0]}", org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
    
    cv2.imshow("Rock paper scissors!", frame)
    result.write(frame)
    
    k = cv2.waitKey(5) #waitKey(1) will display a frame for 1 ms, after which display will be automatically closed
    if k%256 == 27: #escape key for quitting 
        # ESC pressed
        break
    if k == ord('r'):
        scores[0] = 0
        scores[1] = 0
        result = cv2.VideoWriter('rock_paper_scissors.mp4', 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         10, size)

cap.release()
result.release()
cv2.destroyAllWindows()
