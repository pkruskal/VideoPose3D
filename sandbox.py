import sys
import argparse
from loguru import logger
import PIL
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def parse_args(cfg : str,im_or_folder : str) -> argparse:
	"""

	:param
	im_or_folder: path to the image or folder for analysis
	cfg: configuration to run

	:return:
	"""

	parser = argparse.ArgumentParser(description='End-to-end inference')

	parser.add_argument(
		'--cfg',
		dest='cfg',
		help='cfg model file (/path/to/model_config.yaml)',
		default=cfg,
		type=str
	)
	parser.add_argument(
		'--output-dir',
		dest='output_dir',
		help='directory for visualization pdfs (default: /tmp/infer_simple)',
		default='/tmp/infer_simple',
		type=str
	)
	parser.add_argument(
		'--image-ext',
		dest='image_ext',
		help='image file name extension (default: mp4)',
		default='mp4',
		type=str
	)
	parser.add_argument(
		'--im_or_folder',
		dest='im_or_folder',
		help='image or folder of images',
		default=im_or_folder,
		type=str
	)

	# these are needed for running in py consol
	parser.add_argument(
		'--mode',
	)

	# these are needed for running in py consol
	parser.add_argument(
		'--port',
	)

	return parser.parse_args()

def setup_detection_config(args):
	cfg = get_cfg()

	# default is cuda, change to CPU if needed
	cfg.MODEL.DEVICE = "cpu"

	# sets the configuration yaml by configuration_file
	cfg.merge_from_file(model_zoo.get_config_file(args.cfg))

	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

	# set the model waits from the pretrained model in the configuration
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)

	return cfg

def load_results(
		load_file = "/Users/pkruskal/Documents/Projects/AItools_downloaded/VideoPose3D/BroomDance_croped_1_50fps.mp4.npz"
):
	import numpy as np
	f = open(load_file,"rb")
	a = np.load(f, allow_pickle=True)
	a['boxes']
	f.close()
	return a

def example():
	"""
	Terminal call python infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir output_directory --image-ext mp4 input_directory
	:return:

	"""

	setup_logger()

	im_or_folder = "./images/soccer.jpeg"


	# see detectron2/detectron2/config for all different yamls
	configuration_file = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"

	args = parse_args(configuration_file,im_or_folder)

	cfg = setup_detection_config(args)
	cfg.MODEL.DEVICE = "cpu"

	# set configs from the default yaml configuration
	cfg.merge_from_file(model_zoo.get_config_file(args.cfg))

	# set thresholding
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4

	# set the weights to the pretrained models
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)

	# detection class that stores the model and handles image formating (detectron2/engine/defaults)
	predictor = DefaultPredictor(cfg)


	keypoints = np.array([
		"nose", "left_eye", "right_eye", "left_ear", "right_ear",
		"left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
		"left_wrist", "right_wrist", "left_hip", "right_hip",
		"left_knee", "right_knee", "left_ankle", "right_ankle"
	])

	# analyze the image
	im_2 = PIL.Image.open(im_or_folder)
	image_mat = np.asarray(im_2)
	outputs = predictor(image_mat)['instances'].to('cpu')

	has_bbox = False
	if outputs.has('pred_boxes'):
		bbox_tensor = outputs.pred_boxes.tensor.numpy()
		if len(bbox_tensor) > 0:
			has_bbox = True
			scores = outputs.scores.numpy()[:, None]
	if has_bbox:
		kps = outputs.pred_keypoints.numpy()
		kps_xy = kps[:, :, :2]
		kps_prob = kps[:, :, 2:3]
		kps_logit = np.zeros_like(kps_prob)  # Dummy
	else:
		kps = []
		bbox_tensor = []

	plt.imshow(image_mat)
	plot_box= False
	key_threshold = 0.2
	if plot_box:
		legend = ['bounding box']
		for ibox in range(bbox_tensor.shape[0]):
			plt.plot([bbox_tensor[ibox, 2], bbox_tensor[ibox, 3],bbox_tensor[ibox, 3], bbox_tensor[ibox, 2], bbox_tensor[ibox, 2]],
			         [bbox_tensor[ibox, 0], bbox_tensor[ibox, 0], bbox_tensor[ibox, 1], bbox_tensor[ibox, 1], bbox_tensor[ibox, 0]],
			         )
	else:
		legend = []
	for ikey in range(len(keypoints)):
		if kps_prob[0][ikey] > key_threshold:
			plt.plot(kps_xy[0][ikey][0] ,kps_xy[0][ikey][1], '.',alpha=0.5)

	legend.extend(keypoints[[b[0]>key_threshold for b in kps_prob[0]]])
	plt.legend(legend,bbox_to_anchor=(1.05, 1))

	# from key points determin

	# 1) based on left and right face marks determin if facing camera

	# 2) based on left and right body marks determin if facing camera

	# Face
	# 3) based on eye to ear distances on left and right determin gaze angle
	# 4) based on eye y axis determin



	# plot skeliton
	skeliton_threshold = 0.25
	face_points = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
	chest = ["left_shoulder","right_shoulder"]
	left_arm_1 = ["left_shoulder","left_elbow"]
	left_arm_2 = ["left_elbow","left_wrist"]
	torso = ["left_hip", "right_hip"]


image = load_image()
outputs = predictor(image)['instances'].to('cpu')