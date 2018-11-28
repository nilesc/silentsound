import csv
import sys
import os
from scipy.io import wavfile
import numpy as np
import cv2
import pickle

def string_to_int(x):
    return int(float(x))

def extract_info(filename, prefix):
    available_files = os.listdir('{}_videos'.format(prefix))

    seen_videos = set()
    f = open(f'{prefix}_condensed.pkl', 'wb')
    pickler = pickle.Pickler(f)

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            video_id = row[0]
            face_x = row[3]
            face_y = row[4]

            if '{}.mp4'.format(video_id) not in available_files:
                continue

            if video_id in seen_videos:
                continue

            seen_videos.add(video_id)

            rate, audio_data = wavfile.read('{}_audio/{}.wav'.format(prefix, video_id)) # sample rate is samples/sec

            # 1D array with one audio channel
            data = np.ravel(np.delete(audio_data, 1, 1))

            cap = cv2.VideoCapture('{}_videos/{}.mp4'.format(prefix, video_id))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))*2

            if cap.isOpened() == False:
                print("Error opening video file.")
                continue

            # Read until video is completed
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frames.append(frame)

            cap.release()

            frames_np = np.asarray(frames)

            pickler.dump((data, frames_np, face_x, face_y, rate, video_id))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Please provide the name of a file to download from, ' +
        'and a prefix to use when storing them')
        sys.exit()
    extract_info(sys.argv[1], sys.argv[2])
