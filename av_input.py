import csv
import os
import subprocess
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2

# Install ffmpeg with Homebrew
# brew install ffmpeg

dir_path = os.path.dirname(os.path.realpath(__file__)) # Uses directory the script is in
csvfile_path = dir_path + '/avspeech_train.csv' # AV Speech CSV file
vid_path = dir_path + '/videos' # Directory to hold video downloads

num_videos = 1 # Number of videos to download

with open(csvfile_path) as csv_file:
	# Format of CSV is YouTube ID, start segment, end segment, X coordinate, Y coordinate
    csv_reader = csv.reader(csv_file, delimiter=',')

    for i, row in enumerate(csv_reader):
    	if i > num_videos:
    		break

    	print str(i) + ": " + str(row)

    	url = 'http://youtube.com/watch?v=' + row[0]
    	
    	# Download video
    	stream = YouTube(url).streams.first()
    	stream.download(output_path=vid_path, filename=row[0])

    	# Trim clip to relevant segment, appends "-1" to filename
    	ffmpeg_extract_subclip(vid_path + "/" + row[0] + ".mp4", float(row[1]), float(row[2]), vid_path + "/" + row[0] + "-1.mp4")

    	## Using youtube-dl and ffmpeg, speed up download process by only downloading relevant clip segment 
    	# print "youtube-dl --get-url " + url
    	# Download best video only format but no bigger than 50 MB
    	# youtube-dl -f 'best[filesize<50M]'
    	# direct_url = subprocess.call("youtube-dl --get-url " + url, shell=True)
    	# subprocess.call('ffmpeg -ss (start time) -i (direct video link) -t (duration needed) -c:v copy -c:a copy (destination file))

    	## View video
    	cap = cv2.VideoCapture(vid_path + "/" + row[0] + "-1.mp4")

    	# Check if camera opened successfully
    	if cap.isOpened() == False: 
    		print("Error opening video file.")

    	# Read until video is completed
    	while cap.isOpened():
    		ret, frame = cap.read()
    		
    		cv2.imshow('frame',frame)

    		## Use video data ##
    		## Will research audio processing ##

    		# Press q on keyboard to exit
    		if cv2.waitKey(1) & 0xFF == ord('q'):
    			break

    	cap.release()
    	cv2.destroyAllWindows()