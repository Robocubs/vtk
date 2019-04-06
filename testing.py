import tensorflow as tf
import numpy as np
import cv2, os, time
from termcolor import cprint
from vtk.inferrers.tensorflow import TensorFlowInferrer
start = time.time()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
cprint("[0/6] Capturing frame...", "green", attrs=["bold"])
status, frame = cv2.VideoCapture(0).read()
cprint("[1/6] Loading graph into inference class...", "green", attrs=["bold"])
inferrer = TensorFlowInferrer("../vision/frozen_inference_graph.pb")
cprint("[2/6] Preparing graph in memory...", "green", attrs=["bold"])
inferrer.prepare()
cprint("[3/6] Running inference on frame...", "green", attrs=["bold"])
results = inferrer.run(frame)
cprint("[4/6] Drawing on frame...", "green", attrs=["bold"])
for i in results["detections"]:
    cv2.rectangle(frame, (i["bbox"][0], i["bbox"][1]), (i["bbox"][2], i["bbox"][3]), 2, (125, 125, 0))
cprint("[5/6] Displaying result, press Q to quit...", "green", attrs=["bold"])
end = time.time()
while not cv2.waitKey(1) & 0xFF == ord("q"):
    cv2.imshow("Output", frame)
cprint("[6/6] Cleaning up...", "green", attrs=["bold"])
cv2.destroyAllWindows()
cprint("Successfully completed test!", "blue", attrs=["bold"])
cprint("Took {s} seconds.".format(s=str(round(end - start, 2))), "blue", attrs=["bold"])
