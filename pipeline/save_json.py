import os
import json
import torch
import numpy as np
from pipeline.pipeline import Pipeline


class SaveJSON(Pipeline):
    """Pipeline task to save keypoints as JSON file."""

    def __init__(self, filename):
        self.filename = filename
        self.summary = {}
        super(SaveJSON, self).__init__()

    def map(self, data):
        #image = data["image"]
        image_id = data["image_id"]
        predictions = data["predictions"]
        instances = predictions["instances"]
        keypoints = instances.pred_keypoints.cpu().numpy()
        num_instances = len(keypoints)
        pose_flows = data["pose_flows"]

        # Loop over all detected person and buffer summary results
        self.summary[image_id] = {}

        # Save results
        for idx, pose_flow in enumerate(pose_flows):
            pid = pose_flow["pid"]
            #box, confidence = pose_flow
            #(start_x, start_y, end_x, end_y) = box.astype("int")
            print(pose_flow)
            (start_x, start_y, end_x, end_y) = pose_flow["box"].astype("int")
            instance_keypoints = keypoints[idx]
            # Save bounding box
            self.summary[image_id][pid] = {
                "box": np.array([start_x, start_y, end_x, end_y], dtype=int).tolist(),
                #"confidence": confidence.item()
                #"keypoint": np.array(instance_keypoints)
            }

        self.write()
        print(f"[INFO] Saving summary to {self.filename}...")
        return data


    def write(self):
        dirname = os.path.dirname(os.path.abspath(self.filename))
        os.makedirs(dirname, exist_ok=True)
        with open(self.filename, 'w') as json_file:
            json_file.write(json.dumps(self.summary))

    # # Storage for JSON summary
    # summary = {}
    # for image_file in input_image_files:
    #     summary[image_file] = {}
    #     # Loop over all detected faces
    #     #See how image save or video save

    #     summary[image_file][face_file] = np.array([x, y, w, h], dtype=int).tolist()

    #  # Save summary data
    # if args["out_summary"]:
    #     summary_file = os.path.join(args["output"], args["out_summary"])
    #     print(f"[INFO] Saving summary to {summary_file}...")
    #     with open(summary_file, 'w') as json_file:
    #         json_file.write(json.dumps(summary))