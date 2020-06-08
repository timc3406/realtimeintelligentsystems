# Realtime Intelligent Systems

## How to use

Create virtual environment and install packages

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run video stream

```bash
python main.py run
```

To update model using jpg images inside dataset

```bash
#first embed images into single file
python main.py embed
#then update model using embeddings
python main.py update-jpg
```

To update model using csv files from webcam

```bash
python main.py update-csv
```

To add more csv files for more people

```bash
python main.py add -n nameofperson
```

To create csv file of random subset of unknown people

```bash
python main.py embed-unknown
```
