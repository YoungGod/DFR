import argparse
from anoseg_dfr import AnoSegDFR
import os


def config():
    parser = argparse.ArgumentParser(description="Settings of DFR")

    # positional args
    parser.add_argument('--mode', type=str, choices=["train", "evaluation"],
                        default="train", help="train or evaluation")

    # general
    parser.add_argument('--model_name', type=str, default="", help="specifed model name")
    parser.add_argument('--save_path', type=str, default=os.getcwd(), help="saving path")
    parser.add_argument('--img_size', type=int, nargs="+", default=(256, 256), help="image size (hxw)")
    parser.add_argument('--device', type=str, default="cuda:0", help="device for training and testing")

    # parameters for the regional feature generator
    parser.add_argument('--backbone', type=str, default="vgg19", help="backbone net")

    cnn_layers = ('relu4_1', 'relu4_2', 'relu4_3', 'relu4_4')
    parser.add_argument('--cnn_layers', type=str, nargs="+", default=cnn_layers, help="cnn feature layers to use")
    parser.add_argument('--upsample', type=str, default="bilinear", help="operation for resizing cnn map")
    parser.add_argument('--is_agg', type=bool, default=True, help="if to aggregate the features")
    parser.add_argument('--featmap_size', type=int, nargs="+", default=(256, 256), help="feat map size (hxw)")
    parser.add_argument('--kernel_size', type=int, nargs="+", default=(4, 4), help="aggregation kernel (hxw)")
    parser.add_argument('--stride', type=int, nargs="+", default=(4, 4), help="stride of the kernel (hxw)")
    parser.add_argument('--dilation', type=int, default=1, help="dilation of the kernel")

    # training and testing
    # default values
    data_name = "bottle"
    train_data_path = "/home/jie/Datasets/MVAomaly/" + data_name + "/train/good"
    test_data_path = "/home/jie/Datasets/dataset/MVAomaly/" + data_name + "/test"

    parser.add_argument('--data_name', type=str, default=data_name, help="data name")
    parser.add_argument('--train_data_path', type=str, default=train_data_path, help="training data path")
    parser.add_argument('--test_data_path', type=str, default=test_data_path, help="testing data path")

    # CAE
    parser.add_argument('--latent_dim', type=int, default=None, help="latent dimension of CAE")
    parser.add_argument('--is_bn', type=bool, default=True, help="if using bn layer in CAE")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epochs', type=int, default=150, help="epochs for training")    # default 700, for wine 150

    # segmentation evaluation
    parser.add_argument('--thred', type=float, default=0.5, help="threshold for segmentation")
    parser.add_argument('--except_fpr', type=float, default=0.005, help="fpr to estimate segmentation threshold")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    #########################################
    #    On the whole data
    #########################################
    cfg = config()
    cfg.save_path = "/home/jie/Python-Workspace/Pycharm-Projects/Anomaly-2020/DFR-Baseline"
    # cfg.model_name = ""

    # feature extractor
#     cfg.cnn_layers = ('relu4_1', 'relu4_2', 'relu4_3', 'relu4_4')
    cfg.cnn_layers = ('relu1_1', 'relu1_2', 'relu2_1', 'relu2_2',
                  'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4',
                  'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4')

    # dataset
    textures = ['carpet', 'grid', 'leather', 'tile', 'wood']
    objects = ['bottle','cable', 'capsule','hazelnut', 'metal_nut',
               'pill', 'screw', 'toothbrush', 'transistor', 'zipper'] 
    data_names = objects + textures

    # train or evaluation
    for data_name in data_names:
        cfg.data_name = data_name
        cfg.train_data_path = "/home/jie/Datasets/MVAomaly/" + data_name + "/train/good"
        cfg.test_data_path = "/home/jie/Datasets/MVAomaly/" + data_name + "/test"

        dfr = AnoSegDFR(cfg)
        if cfg.mode == "train":
            dfr.train()
        else:
            dfr.metrics_evaluation()
#             dfr.metrics_detecion()
