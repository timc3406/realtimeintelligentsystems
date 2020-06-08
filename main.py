
import argparse
from src import add_person, embed_pics, run_video, embed_pics_unknown, update_model, update_model2

parser = argparse.ArgumentParser(prog='realtime', description="RealTime Intelligent Systems Project")
parser.add_argument(
    'action', choices=['run', 'embed', 'update-jpg', 'add','embed-unknown','update-csv'], help='run or update the model')
parser.add_argument("-d", "--detector", default="face_detection_model",
                    help="path to OpenCV's deep learning face detector")
parser.add_argument("-m", "--embedding-model", default="openface_nn4.small2.v1.t7",
                    help="path to OpenCV's deep learning face embedding model")
parser.add_argument("-r", "--recognizer", default="output/recognizer.pickle",
                    help="path to model trained to recognize faces")
parser.add_argument("-l", "--le", default="output/le.pickle",
                    help="path to label encoder")
parser.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
parser.add_argument("-n", "--name", help="name of person")
parser.add_argument("-s", "--dataset", default="dataset")
parser.add_argument("-e", "--embeddings", default="output/embeddings.pickle")

args = vars(parser.parse_args())

# MAIN COMMAND CONTROL
if args["action"] == "run":
    run_video.run(args)
elif args["action"] == "embed":
    embed_pics.embed(args)
elif args["action"] == "update-jpg":
    update_model.update_model(args)
elif args["action"] == "add":
    add_person.create_csv(args)
elif args["action"] == "embed-unknown":
    print('sdf')
    embed_pics_unknown.embed_unknown(args)
elif args["action"] == "update-csv":
    update_model2.update_model2(args)
