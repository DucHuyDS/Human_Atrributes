import argparse
import os
import numpy as np
import torch
from dataset.AttrDataset import AttrDataset, get_transform
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50
from tools.function import get_model_log_path
from PIL import Image
import pickle
from tools.function import get_pkl_rootpath


def main(args):
    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    save_model_path = os.path.join(model_dir, 'ckpt_max.pth')


    data_path = get_pkl_rootpath(args.dataset)

    dataset_info = pickle.load(open(data_path, 'rb+'))
    
    atrrid = np.array(dataset_info.attr_name)
    _, valid_tsfm = get_transform(args)

    train_set = AttrDataset(args=args, split=args.train_split, transform=_)

    backbone = resnet50()
    classifier = BaseClassifier(nattr=train_set.attr_num)
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()


    checkpoint = torch.load(save_model_path)
    model.load_state_dict(checkpoint['state_dicts'])
    model.eval()

    img = Image.open(args.img_path).convert('RGB') #data\PA100k\data\000001.jpg
    
    img = valid_tsfm(img)
     
    img = img.cuda()

    with torch.no_grad():
        test_logits = model(img.unsqueeze(0))
        test_probs = torch.sigmoid(test_logits)
        test_probs = test_probs.cpu().numpy()
        
    print(atrrid[test_probs[0] > 0])
    
    print(test_probs[0][test_probs[0] > 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset", type=str, default="RAP")
    parser.add_argument("--threshold", type=int, default=0.5)
    parser.add_argument("--img_path", type=str, required = True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)

    args = parser.parse_args()
    main(args)

    # os.path.abspath()

"""
python infer.py PA100k --img_path data\PA100k\data\000001.jpg
"""
