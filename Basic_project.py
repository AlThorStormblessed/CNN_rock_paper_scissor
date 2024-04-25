# importing libraries
import cv2
import numpy as np
import keras
import random
import time, pickle
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dropout, Dense, Flatten
from keras.models import Sequential
from keras.applications import MobileNet
from keras.utils import to_categorical
import json
import requests


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
    
def create_model():
    model = Sequential()

    base_model = MobileNet(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(224, 224, 3),
        classes=4,
        pooling="avg",
        include_top=False,
    )
    # freeezing the weights of the final layer
    for layer in base_model.layers:
        layer.trainable = False

    # model.add(base_model)
    # # model.add(Flatten())
    # model.add(Dense(512, activation="relu"))
    # model.add(Dropout(0.5))
    # model.add(Dense(4, activation="softmax"))  # final op layer

    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(4, activation="softmax")
    ])

    # model.build = True
    return model

model = create_model()
# model.compile(
#     optimizer=Adam(learning_rate=0.01),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"])
model.build(input_shape=(None, 224,224,3)) 
model.load_weights("saved_model/model_t2.weights.h5")


alpha = 0.5
discount = 1
total_reward = 0
moves = ["Rock", "Rock", "Rock"]
dic = {"Rock" : 0, "Paper" : 1, "Scissors" : 2}

# video capture and model prediction
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cv2.namedWindow("Rock paper scissors!")

played_move = 0
n = 5
scores = [0, 0]

frame_width = int(cap.get(3)) + 60
frame_height = int(cap.get(4)) + 60

size = (frame_width, frame_height)
rock_img = cv2.imread(
    "Rock_img.jpg"
)
paper_img = cv2.imread(
    "paper_img.jpg"
)
scissor_img = cv2.imread(
    "Scissor_img.jpg"
)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (100, 130), (500, 530), (255, 255, 255), 2)
    cv2.rectangle(frame, (900, 130), (1300, 530), (0, 0, 0), 2)

    frame = cv2.copyMakeBorder(
        frame, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # extract the region of image within the user rectangle
    roi = frame[160:560, 130:530]

    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # normalizing pixel values between 0 and 1

    # predict the move made
    pred = model.predict(np.array([img]), verbose = 0)
    pred_num = get_class_arg(pred)
    move = get_class(pred_num)

    if played_move < n and pred_num < 3:
        played_move += 1
        if played_move >= n:
            # print(moves)
            pc_move = random.choice(["Rock", "Paper", "Scissors"])

            if (
                (move == "Rock" and pc_move == "Paper")
                or (move == "Paper" and pc_move == "Scissors")
                or (move == "Scissors" and pc_move == "Rock")
            ):
                scores[0] += 1
                text = "Lose"
            elif (
                (pc_move == "Rock" and move == "Paper")
                or (pc_move == "Paper" and move == "Scissors")
                or (pc_move == "Scissors" and move == "Rock")
            ):
                scores[1] += 1
                text = "Win"
            else:
                text = "Tie"

    if played_move >= n:
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (630, 280)
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
        # if ret:
        frame[160:560, 930:1330] = target_img

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (1180, 100)
        fontScale = 0.8
        color = (255, 0, 0)
        thickness = 2
        frame = cv2.putText(
            frame, pc_move, org, font, fontScale, color, thickness, cv2.LINE_AA
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (600, 330)
        fontScale = 3
        color = (0, 0, 200)
        thickness = 2
        frame = cv2.putText(
            frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA
        )

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
        frame = cv2.putText(
            frame, move, org, font, fontScale, color, thickness, cv2.LINE_AA
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (550, 640)
        fontScale = 3
        color = (0, 0, 200)
        thickness = 5
        frame = cv2.putText(
            frame,
            f"{scores[1]} - {scores[0]}",
            org,
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        cv2.imshow("Rock paper scissors!", frame)
        # time.sleep(20)

    if played_move and pred_num == 3:
        played_move = 0

    # adding text information
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
    frame = cv2.putText(
        frame, move, org, font, fontScale, color, thickness, cv2.LINE_AA
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (550, 640)
    fontScale = 3
    color = (0, 0, 200)
    thickness = 5
    frame = cv2.putText(
        frame,
        f"{scores[1]} - {scores[0]}",
        org,
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    cv2.imshow("Rock paper scissors!", frame)

    # data = {'move':played_move}
    # data_json = json.dumps(data)
    # payload = {'json_payload': data_json}
    # r = requests.get('http://localhost:5000/results', data=payload)

    k = cv2.waitKey(
        5
    )  # waitKey(1) will display a frame for 1 ms, after which display will be automatically closed
    if k % 256 == 27:  # escape key for quitting
        # ESC pressed
        break
    if k == ord("r"):
        scores[0] = 0
        scores[1] = 0


# cap.release()
# cv2.destroyAllWindows()
