import torch
import torch.nn as nn
import torch.nn.functional as F
#from extractors.feature import Extractor
from feature import Extractor
from torch.utils.data import DataLoader
import torch.optim as optim
#from data.MVTec import NormalDataset, TrainTestDataset
from MVTec import NormalDataset, TrainTestDataset, TestDataset

import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage import measure
from skimage.transform import resize
import pandas as pd

from feat_cae import FeatCAE

from sklearn.externals import joblib
from sklearn.decomposition import PCA

from utils import *


class AnoSegDFR():
    """
    Anomaly segmentation model: DFR.
    """
    def __init__(self, cfg):
        super(AnoSegDFR, self).__init__()
        self.cfg = cfg
        self.path = cfg.save_path    # model and results saving path

        self.n_layers = len(cfg.cnn_layers)
        self.n_dim = cfg.latent_dim

        self.log_step = 10
        self.data_name = cfg.data_name

        self.img_size = cfg.img_size
        self.threshold = cfg.thred
        self.device = torch.device(cfg.device)

        # feature extractor
        self.extractor = Extractor(backbone=cfg.backbone,
                 cnn_layers=cfg.cnn_layers,
                 upsample=cfg.upsample,
                 kernel_size=cfg.kernel_size,
                 stride=cfg.stride,
                 dilation=cfg.dilation,
                 featmap_size=cfg.featmap_size,
                 device=cfg.device).to(self.device)

        # datasest
        self.train_data_path = cfg.train_data_path
        self.test_data_path = cfg.test_data_path
        self.train_data = self.build_dataset(is_train=True)
        self.test_data = self.build_dataset(is_train=False)

        # dataloader
        self.train_data_loader = DataLoader(self.train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
        self.test_data_loader = DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=1)
        self.eval_data_loader = DataLoader(self.train_data, batch_size=10, shuffle=False, num_workers=2)

        # autoencoder classifier
        self.autoencoder, self.model_name = self.build_classifier()

        # optimizer
        self.lr = cfg.lr
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.lr, weight_decay=0)

        # saving paths
        self.subpath = self.data_name + "/" + self.model_name
        self.model_path = os.path.join(self.path, "models/" + self.subpath + "/model")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.eval_path = os.path.join(self.path, "models/" + self.subpath + "/eval")
        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path)

    def build_classifier(self):
        # self.load_dim(self.model_path)
        if self.n_dim is None:
            print("Estimating one class classifier AE parameter...")
            feats = torch.Tensor()
            for i, normal_img in enumerate(self.eval_data_loader):
                i += 1
                if i > 1:
                    break
                normal_img = normal_img.to(self.device)
                feat = self.extractor.feat_vec(normal_img)
                feats = torch.cat([feats, feat.cpu()], dim=0)
            # to numpy
            feats = feats.detach().numpy()
            # estimate parameters for mlp
            pca = PCA(n_components=0.90)
            pca.fit(feats)
            n_dim, in_feat = pca.components_.shape
            print("AE Parameter (in_feat, n_dim): ({}, {})".format(in_feat, n_dim))
            self.n_dim = n_dim
        else:
            for i, normal_img in enumerate(self.eval_data_loader):
                i += 1
                if i > 1:
                    break
                normal_img = normal_img.to(self.device)
                feat = self.extractor.feat_vec(normal_img)
            in_feat = feat.shape[1]


        autoencoder = FeatCAE(in_channels=in_feat, latent_dim=self.n_dim).to(self.device)

        model_name = "AnoSegDFR_{}Pad_l{}_d{}_s{}_k{}".format(self.cfg.backbone, self.n_layers,
                                                                 self.n_dim, self.cfg.stride[0],
                                                                 self.cfg.kernel_size[0])
        return autoencoder, model_name

    def build_dataset(self, is_train):
        normal_data_path = self.train_data_path
        abnormal_data_path = self.test_data_path
        if is_train:
            dataset = NormalDataset(normal_data_path, normalize=True)
        else:
            # dataset = TrainTestDataset(abnormal_path=abnormal_data_path,
            #                            normal_path=None,
            #                            is_train=is_train, train_size=0., seed=0)
            dataset = TestDataset(path=abnormal_data_path)
        return dataset

    def train(self):
        if self.load_model():
            # return
            print("Model Loaded.")

        start_time = time.time()

        # train
        iters_per_epoch = len(self.train_data_loader)  # total iterations every epoch
        epochs = self.cfg.epochs  # total epochs
        for epoch in range(0, epochs):
            self.extractor.train()
            self.autoencoder.train()
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            for i, normal_img in enumerate(self.train_data_loader):
                normal_img = normal_img.to(self.device)
                # forward and backward
                total_loss = self.optimize_step(normal_img)

                # statistics and logging
                loss = {}
                loss['total_loss'] = total_loss.data.item()

                # print out log info
                if (i + 1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    total_time = ((epochs * iters_per_epoch) - (epoch * iters_per_epoch + i)) * elapsed / (
                            epoch * iters_per_epoch + i + 1)
                    epoch_time = (iters_per_epoch - i) * elapsed / (epoch * iters_per_epoch + i + 1)

                    epoch_time = str(datetime.timedelta(seconds=epoch_time))
                    total_time = str(datetime.timedelta(seconds=total_time))
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, epoch_time, total_time, epoch + 1, epochs, i + 1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                    self.save_model()

        # save model
        self.save_model()
        print("Cost total time {}s".format(time.time() - start_time))
        print("Done.")

    def optimize_step(self, input_data):
        self.extractor.train()
        self.autoencoder.train()

        self.optimizer.zero_grad()

        # forward
        input_data = self.extractor(input_data)
        # # random select 10000 samples
        # n = input_data.shape[0]
        # idx = torch.randint(low=0, high=n, size=(min(20000, n),)).to(self.device)
        # input_data = torch.index_select(input_data, dim=0, index=idx)

        # print(input_data.size())
        dec = self.autoencoder(input_data)

        # loss
        total_loss = self.autoencoder.loss_function(dec, input_data.detach().data)

        # self.reset_grad()
        total_loss.backward()

        self.optimizer.step()

        return total_loss

    def score(self, input):
        """
        Args:
            input: image with size of (img_size_h, img_size_w, channels)
        Returns:
            score map with shape (img_size_h, img_size_w)
        """
        self.extractor.eval()
        self.autoencoder.eval()

        input = self.extractor(input)
        dec = self.autoencoder(input)

        # sample energy
        scores = self.autoencoder.compute_energy(dec, input)
        scores = scores.reshape((1, 1, self.extractor.out_size[0], self.extractor.out_size[1]))    # test batch size is 1.
        scores = nn.functional.interpolate(scores, size=self.img_size, mode="bilinear").squeeze()
        # print("score shape:", scores.shape)
        return scores

    def segment(self, input, threshold=0.5):
        """
        Args:
            input: image with size of (img_size_h, img_size_w, channels)
        Returns:
            score map and binary score map with shape (img_size_h, img_size_w)
        """
        # predict
        scores = self.score(input).data.cpu().numpy()

        # binary score
        print("threshold:", threshold)
        binary_scores = np.zeros_like(scores)    # torch.zeros_like(scores)
        binary_scores[scores <= threshold] = 0
        binary_scores[scores > threshold] = 1

        return scores, binary_scores

    def segment_evaluation(self):
        i = 0
        metrics = []
        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)
            scores, binary_scores = self.segment(img, threshold=self.threshold)

            # show something
            #     plt.figure()
            #     ax1 = plt.subplot(1, 2, 1)
            #     ax1.imshow(resize(mask[0], (256, 256)))
            #     ax1.set_title("gt")

            #     ax2 = plt.subplot(1, 2, 2)
            #     ax2.imshow(scores)
            #     ax2.set_title("pred")

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            self.save_seg_results(normalize(scores), binary_scores, mask, name)
            # metrics of one batch
            if name.split("/")[-2] != "good":
                specificity, sensitivity, accuracy, coverage, auc = spec_sensi_acc_iou_auc(mask, binary_scores, scores)
                metrics.append([specificity, sensitivity, accuracy, coverage, auc])
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))
        # metrics over all data
        metrics = np.array(metrics)
        metrics_mean = metrics.mean(axis=0)
        metrics_std = metrics.std(axis=0)
        print("metrics: specificity, sensitivity, accuracy, iou, auc")
        print("mean:", metrics_mean)
        print("std:", metrics_std)
        print("threshold:", self.threshold)

    def save_paths(self):
        # generating saving paths
        score_map_path = os.path.join("Results", self.subpath + "/score_map")
        if not os.path.exists(score_map_path):
            os.makedirs(score_map_path)

        binary_score_map_path = os.path.join("Results", self.subpath + "/binary_score_map")
        if not os.path.exists(binary_score_map_path):
            os.makedirs(binary_score_map_path)

        gt_pred_map_path = os.path.join("Results", self.subpath + "/gt_pred_score_map")
        if not os.path.exists(gt_pred_map_path):
            os.makedirs(gt_pred_map_path)

        mask_path = os.path.join("Results", self.subpath + "/mask")
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

        gt_pred_seg_image_path = os.path.join("Results", self.subpath + "/gt_pred_seg_image")
        if not os.path.exists(gt_pred_seg_image_path):
            os.makedirs(gt_pred_seg_image_path)

        return score_map_path, binary_score_map_path, gt_pred_map_path, mask_path, gt_pred_seg_image_path

    def save_seg_results(self, scores, binary_scores, mask, name):
        score_map_path, binary_score_map_path, gt_pred_score_map, mask_path, gt_pred_seg_image_path = self.save_paths()
        img_name = name.split("/")
        img_name = "-".join(img_name[-2:])
        print(img_name)
        # score map
        imsave(os.path.join(score_map_path, "{}".format(img_name)), scores)

        # binary score map
        imsave(os.path.join(binary_score_map_path, "{}".format(img_name)), binary_scores)

        # mask
        imsave(os.path.join(mask_path, "{}".format(img_name)), mask)

        # # pred vs gt map
        # imsave(os.path.join(gt_pred_score_map, "{}".format(img_name)), normalize(binary_scores + mask))
        visulization_score(img_file=name, mask_path=mask_path,
                     score_map_path=score_map_path, saving_path=gt_pred_score_map)
        # pred vs gt image
        visulization(img_file=name, mask_path=mask_path,
                     score_map_path=binary_score_map_path, saving_path=gt_pred_seg_image_path)

    def save_model(self, epoch=0):
        # save model weights
        torch.save({'extractor': self.extractor.state_dict(),
                    'autoencoder': self.autoencoder.state_dict()},
                   os.path.join(self.model_path, 'autoencoder.pth'))
        np.save(os.path.join(self.model_path, 'n_dim.npy'), self.n_dim)

    def load_model(self, path=None):
        print("Loading model...")
        if path is None:
            model_path = os.path.join(self.model_path, 'autoencoder.pth')
            print("model path:", model_path)
            if not os.path.exists(model_path):
                print("Model not exists.")
                return False

            if torch.cuda.is_available():
                data = torch.load(model_path)
            else:
                data = torch.load(model_path,
                                  map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU, using a function

            self.extractor.load_state_dict(data['extractor'])
            self.autoencoder.load_state_dict(data['autoencoder'])
            print("Model loaded:", model_path)
        return True

    # def save_dim(self):
    #     np.save(os.path.join(self.model_path, 'n_dim.npy'))

    def load_dim(self, model_path):
        dim_path = os.path.join(model_path, 'n_dim.npy')
        if not os.path.exists(dim_path):
            print("Dim not exists.")
            self.n_dim = None
        else:
            self.n_dim = np.load(os.path.join(model_path, 'n_dim.npy'))

    ########################################################
    #  Evaluation (testing)
    ########################################################
    def segmentation_results(self):
        def normalize(x):
            return x/x.max()

        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)
            scores, binary_scores = self.segment(img, threshold=self.threshold)

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            if name[0].split("/")[-2] != "good":
                self.save_seg_results(normalize(scores), binary_scores, mask, name)
            # self.save_seg_results((scores-score_min)/score_range, binary_scores, mask, name)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))


    ########################################################
    #  Evaluation (AUC and PRO-AUC)
    ########################################################
    def pro_auc_evaluation(self, expect_fpr=0.3):
        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return

        from sklearn.metrics import auc
        i = 0
        time_start = time.time()
        masks = []
        scores = []
        for i, (img, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
            i += 1
            # data
            img = img.to(self.device)
            mask = mask.squeeze().numpy()

            # score
            score = self.score(img).data.cpu().numpy()

            # save results, save score as .npy
            # score_path = os.path.join("Results", self.subpath + "/array_scores")
            # if not os.path.exists(score_path):
            #     os.makedirs(score_path)
            # mask_path = os.path.join("Results", self.subpath + "/array_masks")
            # if not os.path.exists(mask_path):
            #     os.makedirs(mask_path)
            # print(name)
            # img_name = name[0].split("/")
            # img_name = "-".join(img_name[-2:])
            # np.save(os.path.join(score_path, "{}".format(img_name)), score)
            # np.save(os.path.join(mask_path, "{}".format(img_name)), mask.astype(np.bool))

            masks.append(mask)
            scores.append(score)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time() - time_start))

        # as array
        masks = np.array(masks)
        masks[masks <= 0.5] = 0
        masks[masks > 0.5] = 1
        masks = masks.astype(np.bool)
        scores = np.array(scores)

        # auc score
        auc_score, roc = auc_roc(masks, scores)
        # metrics over all data
        print("auc:",  auc_score)

        # per region overlap
        max_step = 5000
        max_th = scores.max()
        min_th = scores.min()
        delta = (max_th - min_th) / max_step

        pros_mean = []
        pros_std = []
        threds = []
        fprs = []
        binary_score_maps = np.zeros_like(scores, dtype=np.bool)
        for step in range(max_step):
            thred = max_th - step * delta
            #     print(thred)
            # segmentation
            binary_score_maps[scores <= thred] = 0
            binary_score_maps[scores > thred] = 1

            pro = []
            for i in range(len(binary_score_maps)):
                #         print(i)
                label_map = measure.label(masks[i], connectivity=2)
                props = measure.regionprops(label_map, binary_score_maps[i])
                for prop in props:
                    pro.append(prop.intensity_image.sum() / prop.area)
            pros_mean.append(np.array(pro).mean())
            pros_std.append(np.array(pro).std())
            # fpr
            masks_neg = ~masks
            fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
            fprs.append(fpr)
            threds.append(thred)
        # as array
        threds = np.array(threds)
        pros_mean = np.array(pros_mean)
        pros_std = np.array(pros_std)
        fprs = np.array(fprs)

        # save results
        data = np.vstack([threds, fprs, pros_mean, pros_std])
        df_metrics = pd.DataFrame(data=data.T, columns=['thred', 'fpr',
                                                        'pros_mean', 'pros_std'])
        df_metrics.to_csv(os.path.join(self.eval_path, '{}_pro_fpr.csv'.format(self.model_name)),
                          sep=',', index=False)


        # default 30% fpr vs pro, pro_auc
        idx = fprs <= expect_fpr    # # rescale fpr [0,0.3] -> [0, 1]
        fprs_selected = fprs[idx]
        fprs_selected = rescale(fprs_selected)
        pros_mean_selected = pros_mean[idx]   
        pro_auc_score = auc(fprs_selected, pros_mean_selected)
        print("pro auc ({}% FPR):".format(int(expect_fpr*100)), pro_auc_score)

        # save results
        data = np.vstack([threds[idx], fprs[idx], pros_mean[idx], pros_std[idx]])
        df_metrics = pd.DataFrame(data=data.T, columns=['thred', 'fpr',
                                                        'pros_mean', 'pros_std'])
        df_metrics.to_csv(os.path.join(self.eval_path, '{}_pro_fpr_{}.csv'.format(self.model_name, expect_fpr)),
                          sep=',', index=False)

        # save auc, pro as 30 fpr
        np.savetxt(os.path.join(self.eval_path, '{}_auc_pro_auc_{}.txt'.format(self.model_name, expect_fpr)),
                   np.vstack([auc_score, pro_auc_score]), delimiter=',')

    ######################################################
    #  Evaluation of segmentation
    ######################################################
    def save_segment_paths(self, fpr):
        # generating saving paths
        binary_score_map_path = os.path.join("Results", self.subpath + "/fpr_{}/binary_score_map".format(fpr))
        if not os.path.exists(binary_score_map_path):
            os.makedirs(binary_score_map_path)

        mask_path = os.path.join("Results", self.subpath + "/fpr_{}/mask".format(fpr))
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

        gt_pred_seg_image_path = os.path.join("Results", self.subpath + "/fpr_{}/gt_pred_seg_image".format(fpr))
        if not os.path.exists(gt_pred_seg_image_path):
            os.makedirs(gt_pred_seg_image_path)

        return binary_score_map_path, mask_path, gt_pred_seg_image_path

    def save_segment_results(self, binary_scores, mask, name, fpr):
        binary_score_map_path, mask_path, gt_pred_seg_image_path = self.save_segment_paths(fpr)
        img_name = name.split("/")
        img_name = "-".join(img_name[-2:])
        print(img_name)
        # binary score map
        imsave(os.path.join(binary_score_map_path, "{}".format(img_name)), binary_scores)

        # mask
        imsave(os.path.join(mask_path, "{}".format(img_name)), mask)

        # pred vs gt image
        visulization(img_file=name, mask_path=mask_path,
                     score_map_path=binary_score_map_path, saving_path=gt_pred_seg_image_path)

    def estimate_thred_with_fpr(self, expect_fpr=0.05):
        """
        Use training set to estimate the threshold.
        """
        threshold = 0
        scores_list = []
        for i, normal_img in enumerate(self.train_data_loader):
            normal_img = normal_img[0:1].to(self.device)
            scores_list.append(self.score(normal_img).data.cpu().numpy())
        scores = np.concatenate(scores_list, axis=0)

        # find the optimal threshold
        max_step = 100
        min_th = scores.min()
        max_th = scores.max()
        delta = (max_th - min_th) / max_step
        for step in range(max_step):
            threshold = max_th - step * delta
            # segmentation
            binary_score_maps = np.zeros_like(scores)
            binary_score_maps[scores <= threshold] = 0
            binary_score_maps[scores > threshold] = 1

            # estimate the optimal threshold base on user defined min_area
            fpr = binary_score_maps.sum() / binary_score_maps.size
            print(
                "threshold {}: find fpr {} / user defined fpr {}".format(threshold, fpr, expect_fpr))
            if fpr >= expect_fpr:  # find the optimal threshold
                print("find optimal threshold:", threshold)
                print("Done.\n")
                break
        return threshold

    def segment_evaluation_with_fpr(self, expect_fpr=0.05):
        # estimate threshold
        thred = self.estimate_thred_with_fpr(expect_fpr=expect_fpr)

        # segment
        i = 0
        metrics = []
        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)
            scores, binary_scores = self.segment(img, threshold=thred)

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            self.save_segment_results(binary_scores, mask, name, expect_fpr)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))
        print("threshold:", thred)

    def segment_evaluation_with_otsu_li(self, seg_method='otsu'):
        """
        ref: skimage.filters.threshold_otsu
        skimage.filters.threshold_li
        e.g.
        thresh = filters.threshold_otsu(image) #返回一个阈值
        dst =(image <= thresh)*1.0 #根据阈值进行分割
        """
        from skimage.filters import threshold_li
        from skimage.filters import threshold_otsu

        # segment
        thred = 0
        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)

            # estimate threshold and seg
            if seg_method == 'otsu':
                thred = threshold_otsu(img.detach().cpu().numpy())
            else:
                thred = threshold_li(img.detach().cpu().numpy())
            scores, binary_scores = self.segment(img, threshold=thred)

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            self.save_segment_results(binary_scores, mask, name, seg_method)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))
        print("threshold:", thred)

    def segmentation_evaluation(self):
        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return
        self.segment_evaluation_with_fpr(expect_fpr=self.cfg.except_fpr)


if __name__ == "__main__":
    anosegpcaae = AnoSegDFR(cfg=None)
    anosegpcaae.train()
    # anosegpcaae.evaluation_avg_img()
    # anosegpcaae.threshold = 0.7
    # anosegpcaae.segment_evaluation()
    # anosegpcaae.pro_auc_evaluation()
    # anosegpcaae.segmentation_results()
    # anosegpcaae.estimate_thred_with_fpr(expect_fpr=0.1)
    anosegpcaae.segment_evaluation_with_fpr(expect_fpr=0.005)
    # anosegpcaae.segment_evaluation_with_otsu_li(seg_method='li')


    ##################################################
    # evaluation on all datasets
    ##################################################
    # textures: carpet, grid, leather, tile, wood
    # objects: bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper
    level1 = ('relu1_1', 'relu1_2')
    level2 = ('relu1_1', 'relu1_2', 'relu2_1', 'relu2_2')
    level3 = ('relu1_1', 'relu1_2', 'relu2_1', 'relu2_2',
              'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4')
    level4 = ('relu1_1', 'relu1_2', 'relu2_1', 'relu2_2',
              'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4',
              'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4')
    level5 = ('relu1_1', 'relu1_2', 'relu2_1', 'relu2_2',
              'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4',
              'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4',
              'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4')
    levels = [level1, level2, level3, level4, level5]

    textures = ['carpet', 'grid', 'leather', 'tile', 'wood']
    objects = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
               'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    data_names = textures + objects

