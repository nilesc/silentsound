In order to download dependencies:
pip install -r requirements.txt

To run our network:
python3 silentsound.py

Create any necessary directories if errors appear.

To download necesssary videos:
python3 downloader.py <avspeech_filename> <output_prefix> <Number of videos to download>

To convert videos to inputs:
python3 to_inputs.py <avspeech_filename> <output_prefix>

To generate outputs from trained network:
python3 generator.py <weights_file> <avspeech_file>
