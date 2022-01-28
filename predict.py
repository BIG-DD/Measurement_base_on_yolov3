import argparse
from models import *
from utils.datasets import *
from utils.utils import *
import numpy as np
import os
import cv2

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('-image_folder', type=str, default='./datasets/DOTA_data/ImageSets/test.txt', help='path to images')
parser.add_argument('-output_folder', type=str, default='./outputs/', help='path to outputs')
parser.add_argument('-plot_flag', type=bool, default=False)
parser.add_argument('-txt_out', type=bool, default=True)
parser.add_argument('-cfg', type=str, default='./cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('-weights_path', type=str, default='./weights/latest.pt', help='weight file path')
parser.add_argument('-class_path', type=str, default='./cfg/icdar.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.6, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.1, help='iou threshold for non-maximum suppression')
parser.add_argument('-batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('-img_size', type=int, default=608, help='size of each image dimension')
opt = parser.parse_args()
print(opt)

if __name__ == '__main__':
    # Bounding-box colors
    color_list = [(0,0,255)]
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
    classes = load_classes(opt.class_path)  # Extracts class labels from file

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    test_file = open(opt.image_folder, 'r')
    for img_path in test_file.readlines():
        img_path = img_path.replace('\n', '')
        img0 = cv2.imread(img_path)  # BGR
        # Padded resize
        img, _, _, _ = resize_square(img0, height=opt.img_size, color=(127.5, 127.5, 127.5))
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0        
        # Get detections
        with torch.no_grad():
            chip = torch.from_numpy(img).unsqueeze(0).to(device)
            pred = model(chip)
            pred = pred[pred[:, :, 8] > opt.conf_thres]

            if len(pred) > 0:
                detections = non_max_suppression(pred.unsqueeze(0), 0.1, opt.nms_thres)[0]
        
        img=img0
        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = opt.img_size - pad_y
        unpad_w = opt.img_size - pad_x
        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_classes=[0]
            # write results to .txt file
            img_name = img_path.split('/')[-1]
            results_img_path = opt.output_folder + img_name
            results_txt_path = opt.output_folder + img_name.replace('png', 'txt')
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
                # write to file
                if opt.txt_out:
                    with open(results_txt_path, 'w') as f:
                        f.write(('%s %.2f %g %g %g %g %g %g %g %g  \n') % \
                                (classes[int(cls_pred)], cls_conf * conf, P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y ))
				
				# Add the bbox to the plot
                label = '%s %.2f' % (classes[int(cls_pred)], conf)
                color = color_list[int(cls_pred)]
                plot_one_box([P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y], img, label=None, color=color)
            if opt.plot_flag:
                cv2.imshow(img_path, img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        cv2.imwrite(results_img_path, img)