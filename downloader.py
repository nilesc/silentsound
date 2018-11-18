import csv
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import multiprocessing as mp
from math import ceil

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

if __name__ == '__main__':
    with open('avspeech_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            video_id = row[0]
            start_time = string_to_int(row[1])
            end_time = string_to_int(row[2])

            url = 'http://youtube.com/watch?v={}'.format(video_id)

            try:
                yt = YouTube(url)
            except:
                print('Error when trying to fetch video')
                continue

            print(yt.length)
            if int(yt.length) > 500:
                continue
    
            print(i)
            filtered = yt.streams.filter(res='240p', mime_type='video/3gpp', fps=30, adaptive=False)

            if not filtered.all():
                continue

            itag = filtered.first().itag
            filename = '{}.3gpp'.format(str(i))
            filename_mp4 = '{}.mp4'.format(str(i))
            download_video(url, itag, filename='videos/{}'.format(filename))

            ffmpeg_extract_subclip('videos/{}'.format(filename), 
                    start_time,
                    end_time,
                    targetname='cropped_videos/{}'.format(filename_mp4))
