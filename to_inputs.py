import csv
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import multiprocessing as mp
from math import ceil
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

    all_data = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            video_id = row[0]
            face_x = row[3]
            face_y = row[4]

            if '{}.mp4'.format(video_id) not in available_files:
                continue

            rate, data = wavfile.read('{}_audio/{}.wav'.format(prefix, video_id)) # sample rate is samples/sec

            # 1D array with one audio channel
            #data = np.ravel(np.delete(data, 1, 1))

            cap = cv2.VideoCapture('{}_videos/{}.mp4'.format(prefix, video_id))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))*2

            if cap.isOpened() == False:
                print("Error opening video file.")
                continue

            i = 0

            # Read until video is completed
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()

                frames.append(frame)

                cap.release()

            frames_np = np.asarray(frames)
            all_data.append((data, frames_np, face_x, face_y))

    with open('{}_condensed.pkl'.format(prefix), 'wb') as output_file:
        pickle.dump(all_data, output_file, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Please provide the name of a file to download from, ' +
        'and a prefix to use when storing them')
        sys.exit()
    extract_info(sys.argv[1], sys.argv[2])