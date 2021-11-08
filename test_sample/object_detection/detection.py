import os, sys, os.path
import argparse
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork
from IPython import display
from PIL import Image, ImageDraw
import urllib, shutil, json
import yaml
from yaml.loader import SafeLoader

#Helper functions
def image_preprocess(input_image, size):
    img = cv2.resize(input_image, (size,size))
    img = np.transpose(img, [2,0,1]) / 255
    img = np.expand_dims(img, 0)
    ##NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    img_mean = np.array([0.485, 0.456,0.406]).reshape((3,1,1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
    img -= img_mean
    img /= img_std
    return img.astype(np.float32)

def draw_box(img, results, label_list, scale_x, scale_y):
    for i in range(len(results)):
        #print(results[i])
        bbox = results[i, 2:]
        label_id = int(results[i, 0])
        score = results[i, 1]
        if(score>0.20):
            xmin, ymin, xmax, ymax = [int(bbox[0]*scale_x), int(bbox[1]*scale_y),
                                      int(bbox[2]*scale_x), int(bbox[3]*scale_y)]
            cv2.rectangle(img,(xmin, ymin),(xmax, ymax),(0,255,0),3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_text = label_list[label_id];
            cv2.rectangle(img, (xmin, ymin), (xmax, ymin-70), (0,255,0), -1)
            cv2.putText(img, "#"+label_text,(xmin,ymin-10), font, 1.2,(255,255,255), 2,cv2.LINE_AA)
            cv2.putText(img, str(score),(xmin,ymin-40), font, 0.8,(255,255,255), 2,cv2.LINE_AA)
    return img


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model_file", type=str, default='', help="model filename, default(None)")
    return parser.parse_args()

def main():
    args = parse_args()
    pdmodel_file = args.model_file
    dir_name = os.path.dirname(pdmodel_file)
    pdmodel_config = dir_name + "/infer_cfg.yml"

    if not os.path.exists(pdmodel_file) or not os.path.exists(pdmodel_config) or not os.path.exists("./horse.jpg"):
        print('model params file "{}" or "{}" or jpg file "./horse.jpg" not exists. Please check them.'.format(pdmodel_file, pdmodel_config))
        return

    device = 'CPU'

    #load the data from config, and setup the parameters
    label_list=[]
    with open(pdmodel_config) as f:
        data = yaml.load(f, Loader=SafeLoader)
    label_list = data['label_list'];

    ie = IECore()
    net = ie.read_network(pdmodel_file)

    net.reshape({'image': [1, 3, 608, 608], 'im_shape': [
        1, 2], 'scale_factor': [1, 2]})

    exec_net = ie.load_network(net, device)
    assert isinstance(exec_net, ExecutableNetwork)

    input_image = cv2.imread("horse.jpg")
    test_image = image_preprocess(input_image, 608)
    test_im_shape = np.array([[608, 608]]).astype('float32')
    test_scale_factor = np.array([[1, 2]]).astype('float32')
    #print(test_image.shape)

    inputs_dict = {'image': test_image, "im_shape": test_im_shape,
            "scale_factor": test_scale_factor}

    output = exec_net.infer(inputs_dict)
    result_ie = list(output.values())

    result_image = cv2.imread("horse.jpg")
    scale_x = result_image.shape[1]/608*2
    scale_y = result_image.shape[0]/608
    result_image = draw_box(result_image, result_ie[0], label_list, scale_x, scale_y)
    _,ret_array = cv2.imencode('.jpg', result_image)
    i = display.Image(data=ret_array)
    display.display(i)

    cv2.imwrite("result.png",result_image)
    print('Done. result save in ./result.png.')

if __name__ == "__main__":
    main()

