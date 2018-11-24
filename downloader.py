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

    subprocess.call(f'ffmpeg -i {video_file} -ab 160k -ac 2 -ar 44100 -vn {audio_file}', shell=True)

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
    available_files = set(os.listdir('{}_videos'.format(prefix)))

    with open(filename) as csv_file:
        # Format of csv files is: YouTube ID, start segment, end segment, X coordinate, Y coordinate
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            video_id = row[0]
            filename_mp4 = f'{prefix}_videos/{video_id}.mp4'
            if filename_mp4 in available_files: # skip if video already downloaded
                continue
            available_files.add(filename_mp4)

            start_time = (string_to_int(row[1]) + string_to_int(row[2]))//2
            end_time = start_time + 1

            url = 'http://youtube.com/watch?v={}'.format(video_id)

            try:
                yt = YouTube(url)
            except:
                print('Error when trying to fetch video')
                continue

            if int(yt.length) > 500:
                continue

            filtered = yt.streams.filter(res='360p', mime_type='video/mp4', fps=30, progressive=True)

            if not filtered.all():
                continue

            num_downloaded += 1

            itag = filtered.first().itag
            filename = '{}.mp4'.format(video_id)
            original_video_location = f'{prefix}_videos/{filename}'
            keyframe_video_location = f'{prefix}_videos/key-{filename}'
            download_video(url, itag, filename=original_video_location)

            start_min = start_time//60
            start_sec = start_time%60
            subprocess.call(f'ffmpeg -y -i {original_video_location} -force_key_frames 00:{start_min}:{start_sec} {keyframe_video_location}', shell=True)
            #ffmpeg_extract_subclip(keyframe_video_location,
            #        start_time,
            #        end_time,
            #        targetname=filename_mp4)

            subprocess.call(f'ffmpeg -y -ss {start_time} -i {keyframe_video_location} -t 1 -vcodec copy -acodec copy -y {filename_mp4}', shell=True)

            extract_audio(prefix, video_id)

            #os.remove(original_video_location)
            os.remove(keyframe_video_location)

            if num_videos and num_downloaded == num_videos:
                break

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Please provide the name of a file to download from, ' +
        'a prefix to use when storing them,' +
        'and, a number of videos to download')
        sys.exit()
    download_file(sys.argv[1], sys.argv[2], num_videos=int(sys.argv[3]))
