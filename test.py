import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *

GLOBAL_IMG_ID = 0  # global image id.
GLOBAL_ANN_ID = 0  # global annotation id.
def get_image_id():
    global GLOBAL_IMG_ID
    GLOBAL_IMG_ID += 1
    return GLOBAL_IMG_ID


def get_ann_id():
    global GLOBAL_ANN_ID
    GLOBAL_ANN_ID += 1
    return GLOBAL_ANN_ID

label_map_dict = {
    'car': 0, 'bus': 1, 'person': 2, 'bike': 3, 'truck': 4,
    'motor': 5, 'train': 6, 'rider': 7, 'traffic sign': 8, 'traffic light': 9, 
}
ann_json_dict = {
      'images': [],
      'type': 'instances',
      'annotations': [],
      'categories': []
}
for class_name, class_id in label_map_dict.items():
      cls = {'supercategory': 'none', 'id': class_id, 'name': class_name}
      ann_json_dict['categories'].append(cls)


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=416,
         iou_thres=0.5,
         conf_thres=0.001,
         nms_thres=0.5,
         save_json=True,
         model=None):

    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device()
        verbose = True
        print("model is None....") 
        print("image size in testMain:", img_size) 
        # Initialize model 
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)
            print("loaded weights")

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    test_path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, batch_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=min(os.cpu_count(), batch_size),
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    print("model eval....")
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1')
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    print('...........init...........')
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        targets = targets.to(device)
        imgs = imgs.to(device)
        # print(imgs.shape) torch.Size([3, 3, 384, 640])
        _, _, height, width = imgs.shape  # batch size, channels, height, width  dataloader.dataset.img_files
        # Plot images with bounding boxes
        if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
            # print("batch_i == 0 and not os.path.exists('test_batch0.jpg')")
            plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

        # Run model
        inf_out, train_out = model(imgs)  # inference and training outputs

        # Compute loss
        if hasattr(model, 'hyp'):  # if model has loss hyperparameters
            loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls

        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)
        # print("run NMS....")
        # Statistics per image
        for si, pred in enumerate(output): # pred(prediction == output) 
            # print("pred:", len(pred))
            # print("width: ", width, "height: ", height)
            if ann_json_dict:
                image = {
                    'file_name': dataloader.dataset.img_files[si],
                    'height': height,
                    'width': width,
                    'id': si,
                }
            ann_json_dict['images'].append(image)

            xmin = []
            ymin = []
            xmax = []
            ymax = []
            classes = []
            classes_text = []
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue
            
            # box = pred[:, :4].clone()  # xyxy
            # scale_coords(imgs[si].shape[1:], box, shapes[si])  # to original shape
            # box = xyxy2xywh(box)  # xywh
            # box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            # id = get_ann_id()
            # category_id = coco91class[int(d[6])]
            # print("#############")
            # print("#############")
            # print(image_id + '\n')
            # print("category_id: ", category_id + "\n")
            # print("bbox", bbox)
            # print("id", id)
            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Append to pycocotools JSON dictionary
            # it = 0
            # print("save JSON....")
            # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
            id = get_image_id()
            box = pred[:, :4].clone()  #xyxy
            scale_coords(imgs[si].shape[1:], box, shapes[si])  #to original shape
            box = xyxy2xywh(box)  #xywh
            box[:, :2] -= box[:, 2:] / 2 # xy center to top-left corner
            print(box)
            # print("img_id" + str(batch_i)  + '\n' )
            # print("id", id)
            # dataloader.dataset.img_files[si])
            for di, d in enumerate(pred):
                id = get_ann_id()

                ann = {
                    'area': 11,
                    'image_id': si,
                    'bbox': [floatn(x, 3) for x in box[di]],
                    'category_id': int(d[6]),
                    'id': id,
                }
                ann_json_dict['annotations'].append(ann)
                # # print(pred[di])
                # jdict.append({'image_id': si, #dataloader.dataset.img_files,
                #             'category_id': coco91class[int(d[6])],
                #             'bbox': [floatn(x, 3) for x in box[di]],
                #             'id': id
                #             })
        
            # print(jdict)
            # print("after save", it)
            # Clip boxes to image bounds
            # print("get out of save json....")
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))
    # print(batch_i)
    # print(dataloader.dataset.img_files)
    with open('results.json', 'w') as file:
            print("++++++++saving json++++++++++")
            json.dump(ann_json_dict, file)
            print("-----finish saving json------")
   
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
    print(".....################")
    # Save JSON
    if True:
        imgIds = [imgid for imgid, x in enumerate(dataloader.dataset.img_files)]
        print('\n')
        # print(jdict)
        # print(imgIds)
        with open('results.json', 'w') as file:
            json.dump(ann_json_dict, file)

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        print("reading COCO JSON.....################")
        
        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        cocoGt = COCO('/cluster/home/qiaotianwei/yolo/coco/annotations/val.json')  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api 

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        map = cocoEval.stats[1]  # update mAP to pycocotools mAP

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=300, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/bdd100k.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='/cluster/home/qiaotianwei/yolo/yolov3/bdd100k_yolov3-spp3_final.weights', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save_json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.iou_thres,
             opt.conf_thres,
             opt.nms_thres,
             opt.save_json)
