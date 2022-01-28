import argparse
import time

import cv2

from models import *
from utils.datasets import *
from utils.utils import *
import zivid
from pathlib import Path
import math
import glob
import re
from numba import jit

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('-image_folder', type=str, default='tir/', help='path to images')
parser.add_argument('-output_folder', type=str, default='outputs/', help='path to outputs')
parser.add_argument('-plot_flag', type=bool, default=False)
parser.add_argument('-txt_out', type=bool, default=True)
parser.add_argument('-cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('-weights_path', type=str, default='weights/best.pt', help='weight file path')
parser.add_argument('-class_path', type=str, default='cfg/icdar.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.01, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.2, help='iou threshold for non-maximum suppression')
parser.add_argument('-batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('-img_size', type=int, default=608, help='size of each image dimension')
opt = parser.parse_args()
print(opt)


def detect(opt):
	# Bounding-box colors
	color_list = [0,0,255]
	# Load model
	model = Darknet(opt.cfg, opt.img_size)

	weights_path = opt.weights_path
	if weights_path.endswith('.weights'):  # saved in darknet format
		load_weights(model, weights_path)
	else:  # endswith('.pt'), saved in pytorch format
		checkpoint = torch.load(weights_path, map_location='cpu')
		model.load_state_dict(checkpoint['model'])
		del checkpoint

	model.to(device).eval()

	# Set Dataloader
	classes = load_classes(opt.class_path)# Extracts class labels from file
	dataloader = load_images(opt.image_folder, batch_size=opt.batch_size, img_size=opt.img_size)

	imgs = []  # Stores image paths
	img_detections = []  # Stores detections for each image index
	for batch_i, (img_paths, img) in enumerate(dataloader):
		print(batch_i, img.shape, end=' ')
                
		# Get detections
		with torch.no_grad():
			chip = torch.from_numpy(img).unsqueeze(0).to(device)
			pred = model(chip)
			pred = pred[pred[:, :, 8] > opt.conf_thres]

			if len(pred) > 0:
				detections = non_max_suppression(pred.unsqueeze(0), 0.1, opt.nms_thres);print(detections)
                
				img_detections.extend(detections)
				imgs.extend(img_paths)





	if len(img_detections) == 0:
		return

	# Iterate through images and save plot of detections
	xy_up_all = []
	xy_down_all = []
	for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
		print("image %g: '%s'" % (img_i, path))
		img = cv2.imread(path)
		# The amount of padding that was added
		pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
		pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
		# Image height and width after padding is removed
		unpad_h = opt.img_size - pad_y
		unpad_w = opt.img_size - pad_x

		# Draw bounding boxes and labels of detections
		if detections is not None:
			unique_classes = detections[:, -1].cpu().unique()
			bbox_colors = random.sample(color_list, len(unique_classes))

			# write results to .txt file
			results_img_path =path.replace(opt.image_folder,opt.output_folder) 
            
			results_txt_path = results_img_path.replace('png', 'txt')
			if os.path.isfile(results_txt_path):
				os.remove(results_txt_path)

			for i in unique_classes:
				n = (detections[:, -1].cpu() == i).sum()
				print('%g %ss' % (n, classes[int(i)]))

			for P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y, conf, cls_conf, cls_pred in detections:
				P1_y = max((((P1_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P1_x = max((((P1_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				P2_y = max((((P2_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P2_x = max((((P2_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				P3_y = max((((P3_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P3_x = max((((P3_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				P4_y = max((((P4_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P4_x = max((((P4_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)

				#计算出预测框四条边的长度
				d_x1x2 = (P1_x - P2_x) * (P1_x - P2_x) + (P1_y - P2_y) * (P1_y - P2_y)
				d_x2x3 = (P2_x - P3_x) * (P2_x - P3_x) + (P2_y - P3_y) * (P2_y - P3_y)
				d_x3x4 = (P3_x - P4_x) * (P3_x - P4_x) + (P3_y - P4_y) * (P3_y - P4_y)
				d_x4x1 = (P4_x - P1_x) * (P4_x - P1_x) + (P4_y - P1_y) * (P4_y - P1_y)

				#预测框四条边相加
				d_all = d_x1x2 + d_x2x3 + d_x3x4 + d_x4x1

				#判断四条边中哪两条边是长边，然后在两条长边上分别取等间距的三个点，并把三点的坐标分别放入xy_up、和xy_down两个数组中
				#列如：xy_up[x1, y1, x2, y2, x3, y3], 分别包含三个点的坐标
				if d_x1x2 > (d_all/4):
					x1_up = P2_x + ((P1_x - P2_x) / 4)
					x2_up = P2_x + ((P1_x - P2_x) / 2)
					x3_up = P2_x + ((3 * P1_x - 3 * P2_x) / 4)
					y1_up = (((x1_up - P1_x) * (P2_y - P1_y)) / (P2_x - P1_x)) + P1_y
					y2_up = (((x2_up - P1_x) * (P2_y - P1_y)) / (P2_x - P1_x)) + P1_y
					y3_up = (((x3_up - P1_x) * (P2_y - P1_y)) / (P2_x - P1_x)) + P1_y
					if (P1_x - P2_x) > 0:
						xy_up = [x1_up, y1_up, x2_up, y2_up, x3_up, y3_up]
					if (P1_x - P2_x) < 0:
						xy_up = [x3_up, y3_up, x2_up, y2_up, x1_up, y1_up]

				if d_x2x3 > (d_all/4):
					x1_up = P3_x + ((P2_x - P3_x) / 4)
					x2_up = P3_x + ((P2_x - P3_x) / 2)
					x3_up = P3_x + ((3 * P2_x - 3 * P3_x) / 4)
					y1_up = (((x1_up - P2_x) * (P3_y - P2_y)) / (P3_x - P2_x)) + P2_y
					y2_up = (((x2_up - P2_x) * (P3_y - P2_y)) / (P3_x - P2_x)) + P2_y
					y3_up = (((x3_up - P2_x) * (P3_y - P2_y)) / (P3_x - P2_x)) + P2_y
					if (P2_x - P3_x) > 0:
						xy_up = [x1_up, y1_up, x2_up, y2_up, x3_up, y3_up]
					if (P2_x - P3_x) < 0:
						xy_up = [x3_up, y3_up, x2_up, y2_up, x1_up, y1_up]

				if d_x3x4 > (d_all/4):
					x1_down = P4_x + ((P3_x - P4_x) / 4)
					x2_down = P4_x + ((P3_x - P4_x) / 2)
					x3_down = P4_x + ((3 * P3_x - 3 * P4_x) / 4)
					y1_down = (((x1_down - P3_x) * (P4_y - P3_y)) / (P4_x - P3_x)) + P3_y
					y2_down = (((x2_down - P3_x) * (P4_y - P3_y)) / (P4_x - P3_x)) + P3_y
					y3_down = (((x3_down - P3_x) * (P4_y - P3_y)) / (P4_x - P3_x)) + P3_y
					if (P3_x - P4_x) > 0:
						xy_down = [x1_down, y1_down, x2_down, y2_down, x3_down, y3_down]
					if (P3_x - P4_x) < 0:
						xy_down = [x3_down, y3_down, x2_down, y2_down, x1_down, y1_down]

				if d_x4x1 > (d_all / 4):
					x1_down = P1_x + ((P4_x - P1_x) / 4)
					x2_down = P1_x + ((P4_x - P1_x) / 2)
					x3_down = P1_x + ((3 * P4_x - 3 * P1_x) / 4)
					y1_down = (((x1_down - P4_x) * (P1_y - P4_y)) / (P1_x - P4_x)) + P4_y
					y2_down = (((x2_down - P4_x) * (P1_y - P4_y)) / (P1_x - P4_x)) + P4_y
					y3_down = (((x3_down - P4_x) * (P1_y - P4_y)) / (P1_x - P4_x)) + P4_y
					if (P4_x - P1_x) > 0:
						xy_down = [x1_down, y1_down, x2_down, y2_down, x3_down, y3_down]
					if (P4_x - P1_x) < 0:
						xy_down = [x3_down, y3_down, x2_down, y2_down, x1_down, y1_down]

				xyint_up = []
				xyint_down = []

				#计算出来的xy_up和xy_down是float型对其进行四舍五入
				for ii in xy_up:
					xyint_up.append(round(ii))

				for jj in xy_down:
					xyint_down.append(round(jj))

				xy_up_all.append(xyint_up)
				xy_down_all.append(xyint_down)
				# write to file
				if opt.txt_out:
					with open(results_txt_path, 'w') as f:
						f.write(('%s %.2f %g %g %g %g %g %g %g %g  \n') % \
							(classes[int(cls_pred)], cls_conf * conf, P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y ))
				
					# Add the bbox to the plot
				label = '%s %.2f' % (classes[int(cls_pred)], conf)
				color = bbox_colors[int(np.where(unique_classes == int(cls_pred))[0])]
				plot_one_box([P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y], img, label=None, color=color)

	return xy_up_all, xy_down_all

def get_sample_data_path():
    """Get sample data path for your OS.

    Returns:
        path: Sample data path

    """
    if os.name == "nt":
        # path = Path(os.environ["PROGRAMDATA"]) / "Zivid"
        path = Path(r"C:\project\yolov3_pytorch_zhawa\test")
    else:
        path = Path("/usr/share/Zivid/data")
    return path

def get_xyz_for_zdf(xyint, zdf_path):
	"""
	从zdf文件中获取xyz信息
	xyint：2d图像中点的xy坐标，xyint[x, y]
	zdf_path：zdf的路径信息
	return: 2D图像中点对应的zdf中的xyz坐标xyz_cloud[x, y, z]
	"""
	app = zivid.Application() #不能删除
	#print(f"Reading point cloud from file: {zdf_path}")
	frame = zivid.Frame(zdf_path)
	#print("Getting point cloud from frame")
	point_cloud = frame.point_cloud()
	xyz = point_cloud.copy_data("xyz")
	xyz = np.array(xyz, dtype=int)
	xyz_cloud = []
	for i in range(3):
		xyz_cloud.append(xyz[xyint[1], xyint[0], i])

	return xyz_cloud

def dist(point11, point22):
	"""
	计算两个三维点的距离，输入x,y,z，输出两个点的距离
	"""
	ddd = 0
	for i in range(len(point11)):
		point1 = point11[i]
		disttt = 0
		for j in range(len(point22)):
			point2 = point22[j]
			x1 = point1[0]
			y1 = point1[1]
			z1 = point1[2]
			x2 = point2[0]
			y2 = point2[1]
			z2 = point2[2]
			x1x2 = x1 - x2
			y1y2 = y1 - y2
			z1z2 = z1 - z2
			distt = math.sqrt(x1x2**2 + y1y2**2 + z1z2**2)
			disttt = disttt + distt
		ddd = ddd + disttt
	return ddd/(len(point11)*len(point22))

def circle_pixel(x, y, radius):
	"""
	给定一个点x，y和半径radius,得出半经内的点的坐标
	return：[(x1, y1), (x2, y2), ...]
	"""
	list = []
	for i in range(x - radius, x + radius + 1):
		for j in range(y - radius, y + radius +1):
			list.append((i, j))

	return list

def get_all_xyz(xy_t, zdf_path):
	"""
	输入[（x1, y1）， （x2, y2）， ...]
	return: 除去nan值的列表
	"""
	xyz = []
	nanxyz = []
	for i in range(len(xy_t)):
		xy = xy_t[i]
		a = get_xyz_for_zdf(xy, zdf_path)
		if True in np.isnan(np.array(a)):
			nanxyz.append(a)
		else:
			xyz.append([a[0], a[1], a[2]])
	return xyz

if __name__ == '__main__':

	strat_time1 = time.clock()
	torch.cuda.empty_cache()
	xy_up_all, xy_down_all = detect(opt)
	zdf_dir = []
	end_time1 = time.clock()
	print('目标检测时间：', end_time1-strat_time1)
	for zdf in glob.glob("./tir/*.zdf"): #.zdf文件路径
		zdf_dir.append(str(zdf))
	for i in range(len(xy_up_all)):
		end_time2 = time.clock()
		xy_up = xy_up_all[i]
		xy_down = xy_down_all[i]
		end_time3 = time.clock()
		print("选择线上的一对点的时间:", end_time3 - end_time2)
		xy_upx = circle_pixel(xy_up[0], xy_up[1], 3)
		xy_upx = np.array(xy_upx, dtype=int)
		xy_downx = circle_pixel(xy_down[0], xy_down[1], 3)
		xy_downx = np.array(xy_downx, dtype=int)
		end_time4 = time.clock()
		print("选择平面上的96个点:", end_time4 - end_time3)
		txt_name = zdf_dir[i].split('\\')[-1].split('.')[0]
		a = get_all_xyz(xy_upx, zdf_dir[i])
		b = get_all_xyz(xy_downx, zdf_dir[i])
		end_time5 = time.clock()
		print("根据平面上的96个点索引对应的xyz值:", end_time5 - end_time4)
		with open("./tir/" + txt_name + '.txt', 'a') as f: #检测结果的.txt文件路径
			zhawa_dis = str(dist(a, b))
			f.write('\n' + zhawa_dis + 'mm')
			print(txt_name + '闸瓦厚度：' + zhawa_dis + ' mm')
		end_time6 = time.clock()
		print("计算一对点的距离：", end_time6 - end_time5)
	end_time = time.clock()
	print('times: ', (end_time-strat_time1), ' s')
	print("Done")