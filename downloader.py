import csv
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import multiprocessing as mp
import subprocess
from math import ceil
import sys
import os
import cv2

import requests

CHUNK_SIZE = 3 * 2**20  # bytes

# download_video and download_chunk based on following code:
# https://github.com/nficano/pytube/issues/180#issuecomment-363529167
def download_video(video_url, itag, filename):
    stream = YouTube(video_url).streams.get_by_itag(itag)
    url = stream.url
    filesize = stream.filesize

    ranges = [[url, i * CHUNK_SIZE, (i+1) * CHUNK_SIZE - 1] for i in range(ceil(filesize / CHUNK_SIZE))]
    ranges[-1][2] = None  # Last range must be to the end of file, so it will be marked as None.

    pool = mp.Pool(min(len(ranges), 64))
    chunks = pool.map(download_chunk, ranges)

    with open(filename, 'wb') as outfile:
        for chunk in chunks:
            outfile.write(chunk)


def download_chunk(args):
    url, start, finish = args
    range_string = '{}-'.format(start)

    if finish is not None:
        range_string += str(finish)

    response = requests.get(url, headers={'Range': 'bytes=' + range_string})
    return response.content

def string_to_int(x):
    return int(float(x))

def extract_audio(prefix, filename):
    ## View video
    video_file = '{}_videos/{}.mp4'.format(prefix, filename)
    audio_file = '{}_audio/{}.wav'.format(prefix, filename)

    cap = cv2.VideoCapture(video_file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))*2
    subprocess.call("ffmpeg -i " + video_file + " -aframes " + str(num_frames) + " " + audio_file, shell=True)

def download_file(filename, prefix, num_videos=None):
    try:
        os.mkdir('{}_videos'.format(prefix))
    except Exception as e:
        print(e)

    try:
        os.mkdir('{}_audio'.format(prefix))
    except Exception as e:
        print(e)

    num_downloaded = 0

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            video_id = row[0]
            start_time = string_to_int(row[1])
            end_time = string_to_int(row[2])

            url = 'http://youtube.com/watch?v={}'.format(video_id)

            try:
                yt = YouTube(url)
            except:
                print('Error when trying to fetch video')
                continue

            if int(yt.length) > 500:
                continue

            filtered = yt.streams.filter(res='240p', mime_type='video/3gpp', fps=30, adaptive=False)

            if not filtered.all():
                continue

            num_downloaded += 1

            itag = filtered.first().itag
            filename = '{}.3gpp'.format(video_id)
            filename_mp4 = '{}.mp4'.format(video_id)
            download_video(url, itag, filename='videos/{}'.format(filename))

            ffmpeg_extract_subclip('videos/{}'.format(filename),
                    start_time,
                    end_time,
                    targetname='{}_videos/{}'.format(prefix, filename_mp4))

            extract_audio(prefix, video_id)

            if num_videos and num_downloaded == num_videos:
                break


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Please provide the name of a file to download from, ' +
        'a prefix to use when storing them,' +
        'and, a number of videos to download')
        sys.exit()
    download_file(sys.argv[1], sys.argv[2], num_videos=int(sys.argv[3]))
