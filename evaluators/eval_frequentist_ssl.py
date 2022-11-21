from pathlib import Path

from . import Evaluator

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, auc
from numba import njit
from multiprocessing import Pool

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

from data import MVTecDataset, MVTecTestDataset, PatchDataset
from globals import MVTEC_CLASSES
from statistics_np import AdaptableRHT
from parse import (
    parse_frequentist_model
)
from metrics.pro import compute_pro

import os
import seaborn as sns

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class FrequentistSSLEvaluator(Evaluator):
    def __init__(self, config):
        super().__init__(config)

        # Load image size and window size
        self.image_size = self.config_data["image_size"]
        self.window_size = self.config_data["window_size"]

        # Number of rows and columns of the grid of patches
        self.n_rows = self.image_size // self.window_size
        self.n_cols = self.image_size // self.window_size

        self.classes = list(MVTEC_CLASSES.values())
        print(self.classes)
        self.checkpoint_manager.n_models = len(self.classes)

        # Load model and optimizer
        self.models = [
            parse_frequentist_model(self.config_train).to(self.device)
            for _ in self.classes
        ]


        # Grid of lambdas of adaptive RHT
        low = 7
        range_ = 6
        lambdas = [10 ** i for i in range(low, low + range_)] + \
                  [10 ** i / 2 for i in range(low, low + range_)] + \
                  [10 ** i / 3 for i in range(low, low + range_)] + \
                  [10 ** i / 5 for i in range(low, low + range_)]
        self.all_lambdas = np.sort(lambdas)
        self.lambdas_list = [
            self.all_lambdas[i:i + 8] for i in range(len(self.all_lambdas) - 7)
        ]
        self.prior_weights = self.config_eval["prior_weights"]

        # Number of samples
        self.n_normal_samples = self.config_eval["n_normal_samples"]
        self.n_test_samples = self.config_eval["n_test_samples"]

        # visualization or not
        self.has_visualization = self.config_eval["has_visualization"]

        # Load trained models
        state_dicts = self.checkpoint_manager.load_model()
        [m.load_state_dict(s) for m, s in zip(self.models, state_dicts)]
        [m.eval() for m in self.models]

        # Global significance level
        self.alpha = self.config_eval["alpha"]

    def get_normal_dataloader(self, normal_dir):
        normal_data = MVTecDataset(normal_dir, self.image_size, test=True)
        n = len(normal_data.data_paths)
        normal_dataloader = DataLoader(normal_data, batch_size=n, shuffle=True)
        print('Loaded normal data of length', n)
        return normal_dataloader

    def get_eval_dataloader(self, eval_dir, gt_dir):
        # Eval Path
        testing_data = MVTecTestDataset(eval_dir, gt_dir, self.image_size)
        n = len(testing_data.data_paths)
        test_dataloader = DataLoader(testing_data, batch_size=n, shuffle=True)
        print('length of test image = ', n)
        return test_dataloader

    def compute_normal_posterior(self, model, normal_dataloader):
        # Compute posterior from normal images
        for data_n, l_n in normal_dataloader:
            data_n = data_n.to(self.device)

            with torch.no_grad():
                map = model.inference(data_n)

            embeddings = [self.infer_embs(map, i, j)
                          for i in range(self.n_rows)
                          for j in range(self.n_cols)
                          ]
            with Pool(2) as p:
                posteriors = p.map(self.compute_mean_cov, embeddings)

            self.n_1 = embeddings[0].shape[0]
            self.p = embeddings[0].shape[1]

            return posteriors

    def infer_embs(self, maps, i, j, tp="normal"):

        all_feat = []
        feat = []

        for m in maps:
            win_size = m.shape[2] / self.n_cols

            start_i = min(m.shape[2] - 1, i * round(win_size))
            end_i = min(m.shape[2], (i+1) * round(win_size))
            start_j = min(m.shape[3] - 1, j * round(win_size))
            end_j = min(m.shape[3], (j+1) * round(win_size))

            f = m[:, :, start_i:end_i, start_j:end_j]
            f = f.mean(axis=(2, 3))
            f = F.normalize(torch.from_numpy(f), dim=1, p=2).numpy()

            feat.append(f)

        feat = np.concatenate(feat, axis=1)
        all_feat.append(feat)

        if tp == "normal":
            feat = np.concatenate(all_feat)
        else:
            feat = np.stack(all_feat)

        return feat

    def compute_mean_cov(self, embs):
        """
        Compute posterior distributions from the training data
        Sample n_sample parameters from the posterior distribution and compute posterior sampling distribution of X
        The testing will be performed on the posterior sampling distribution
        :return: list of posterior distributions
        """

        # Compute posterior mean and variance
        mean = np.mean(embs, axis=0)
        # p by p covariance matrix
        cov = np.matmul((embs - mean).T, (embs - mean))

        return mean, cov

    def compute_pvalue(self, idx, args):
        testers, mu, test_mu, n_1, n_2, p = args

        b = idx // len(self.lambdas_list)
        l = idx % len(self.lambdas_list)

        lambda_arr = self.lambdas_list[l]
        test_mu = test_mu[b]

        tester = testers[b]

        # Tune lambda using the covariance matrix
        lamb = self.tune_lambda(tester, lambda_arr)
        t_stat = tester.adaptive_rht(lamb, mu, test_mu, n_1, n_2, p).item()
        pvalue = min(stats.norm.cdf(t_stat), 1 - stats.norm.cdf(t_stat))
        return pvalue

    def tune_lambda(self, tester, lambdas):
        q_values = [tester.Q_function(lamb, self.prior_weights) for lamb in lambdas]
        indx = np.argmax(q_values)
        return lambdas[indx]

    def test_one_step(self, test_embs, posterior):
        # Read batch size
        batch_size = test_embs.shape[0]

        # test_embs = self.infer_embs(test_data, tp="testing")
        test_mu = np.mean(test_embs, axis=1)

        # Compute covariance matrix of the testing embeddings
        covs = []
        var_test_embs = []
        for b in range(batch_size):
            _, cov = self.compute_mean_cov(test_embs[b])
            covs.append(cov)
            var_test_embs.append(np.trace(cov))

        # Compute the pool variance
        n_1, p = self.n_1, self.p
        n_2 = test_embs.shape[1]
        mu, sigma_1 = posterior
        sigma_2 = np.stack(covs, axis=0)
        sigma = (np.stack([sigma_1] * batch_size) + sigma_2) / (n_1 + n_2 - 2)

        # Compute total variations and l2 distance
        var_test_embs = np.stack(var_test_embs, axis=0)
        l2_distance = np.sqrt(np.sum((test_mu - np.stack([mu] * batch_size)) ** 2, axis=1))

        testers = [AdaptableRHT(self.all_lambdas, sigma[b], n_1 + n_2, p) for b in range(batch_size)]

        args = (testers, mu, test_mu, n_1, n_2, p)

        # with Pool(4) as p:
        #     pvalues = p.map(self.compute_pvalue, args)
        # pvalues = [self.compute_pvalue(*arg) for arg in args]
        pvalues = np.arange(batch_size * len(self.lambdas_list)).reshape(
            (batch_size, len(self.lambdas_list)))
        v_func = np.vectorize(self.compute_pvalue)
        v_func.excluded.add(1)
        pvalues = v_func(pvalues, args)

        # pvalues = np.array(pvalues).reshape(batch_size, -1)
        return (
            pvalues,
            l2_distance,
            var_test_embs
        )

    def compute_anomaly_score(self, pvalues):

        score_map = np.arange(self.image_size * self.image_size).reshape((self.image_size, self.image_size))
        x_refs = [self.window_size // 2 + i * self.window_size for i in range(self.n_rows)]
        y_refs = [self.window_size // 2 + j * self.window_size for j in range(self.n_cols)]
        xcoords = score_map // self.image_size
        ycoords = score_map % self.image_size
        distances = []
        for x_ref in x_refs:
            for y_ref in y_refs:
                distances.append(np.sqrt((xcoords - x_ref) ** 2 + (ycoords - y_ref) ** 2))
        weights = [d / sum(distances) for d in distances]
        score_map = sum(w * p for w, p in zip(weights, pvalues))

        return score_map

    def plot_heatmap(self, input_img, pvalues):
        """
        Plot heatmap of the pvalues on the input image
        :param input_img:
        :param pvalues:
        :return:
        """
        # overlap heatmap on input image
        heat_map = input_img.clone()
        p_idx = 0
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                heat_map[i * self.window_size:(i + 1) * self.window_size,
                j * self.window_size:(j + 1) * self.window_size] = pvalues[p_idx]
                p_idx += 1

        return heat_map

    def plot_bh_img(self, input_img, r_set):
        """
        Blacken the rejected patches with
        :param input_img:
        :param r_set:
        :return:
        """

        bh_img = input_img.clone()

        for patch in r_set:
            i = patch // self.n_cols
            j = patch % self.n_cols

            bh_img[i * self.window_size:(i + 1) * self.window_size,
            j * self.window_size:(j + 1) * self.window_size, :] = 0

        return bh_img

    def image_level_testing(self):
        pass

    def visualize_result(self, input_img, r_set,
                         score_map, pvalues, l2_distance,
                         var_test_embs, cl, def_type,
                         image_idx, lambda_idx=1):
        fig_img, ax_img = plt.subplots(2, 4, figsize=(25, 10))
        ax_img[0, 0].imshow(input_img)
        ax_img[0, 0].title.set_text('Image')

        bh_img = self.plot_bh_img(input_img, r_set)
        ax_img[0, 1].imshow(bh_img)
        ax_img[0, 1].title.set_text('After BH Testing Result')

        ax_img[0, 2].imshow(input_img, cmap='gray', interpolation='none')
        heat_map = self.plot_heatmap(input_img, pvalues)
        ax_img[0, 2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[0, 2].title.set_text('P-Value Heatmap')

        # plot p-value list
        pvalues_list = [float("{0:.2f}".format(v)) for v in pvalues]
        pvalues_list = np.array(pvalues_list).reshape((self.n_rows, self.n_cols))
        sns.heatmap(pvalues_list, annot=True, cmap='RdBu', linecolor='white', ax=ax_img[0, 3])
        ax_img[0, 3].title.set_text('P Value')

        l2_distance_list = [float("{0:.2f}".format(v)) for v in l2_distance]
        l2_distances = np.array(l2_distance_list).reshape((self.n_rows, self.n_cols))
        sns.heatmap(l2_distances, annot=True, cmap='RdBu', linecolor='white', ax=ax_img[1, 0])
        ax_img[1, 0].title.set_text('l2 distance')

        # Plot hte covariance matrix
        var_test_embs_list = [float("{0:.2f}".format(v)) for v in var_test_embs]
        var_test_embs_list = np.array(var_test_embs_list).reshape((self.n_rows, self.n_cols))
        sns.heatmap(var_test_embs_list, annot=True, cmap='RdBu', linecolor='white', ax=ax_img[1, 1])
        ax_img[1, 1].title.set_text('Embeding Var')

        # Plot heatmap by IDW
        ax_img[1, 2].imshow(score_map)
        ax_img[1, 2].title.set_text('Anomaly score map')

        results_output_path = os.path.join(self.config_data['output_path'], str(lambda_idx), cl, def_type)
        if not os.path.exists(results_output_path):
            os.makedirs(results_output_path)
        img_name = os.path.join(results_output_path, str(image_idx) + '.png')
        fig_img.savefig(img_name, dpi=100)
        plt.close(fig_img)

    def get_score_map(self, pvalues_list, batch_size):
        out = np.zeros((len(self.lambdas_list), batch_size, self.image_size, self.image_size))
        score_maps = []
        for lambda_idx in range(len(self.lambdas_list)):
            for b in range(batch_size):
                s = [p[b][lambda_idx] for p in pvalues_list]
                s = torch.Tensor(s).reshape((self.n_rows, self.n_cols))
                score_maps.append(s)
        score_maps = torch.stack(score_maps)
        score_maps = torch.unsqueeze(score_maps, 1)
        score_maps = F.interpolate(score_maps, self.image_size, mode='bilinear').numpy()
        score_maps = np.squeeze(score_maps, axis=1)

        for lambda_idx in range(len(self.lambdas_list)):
            for b in range(batch_size):
                out[lambda_idx, b, :, :] = score_maps[lambda_idx * batch_size + b, :, :]

        out = 1 - out

        return out

    def plot_tpr_fpr(self):
        data_root = self.config_data['data_root']
        # Create results path
        results_path = self.config_eval["eval_results_path"]
        if not Path(results_path).exists():
            Path(results_path).mkdir(parents=True)
        rf = open(results_path + "results.txt", "a")
        with open(results_path + "summary.txt", "a") as f:
            f.write("Class; Image AUC; Pixel AUC; PROAUC\n")

        for cl_idx, cl in enumerate(self.classes):
            rf.write(cl + "\n")
            def_path = os.path.join(data_root, cl, 'test')
            normal_path = os.path.join(data_root, cl, 'train', 'good')
            normal_dataloader = self.get_normal_dataloader(normal_path)
            posteriors = self.compute_normal_posterior(self.models[cl_idx], normal_dataloader)

            num_p_sample = {}
            num_n_sample = {}
            tp = {}
            fp = {}
            pixel_auc = {}
            pro_all = {}
            for lambda_idx in range(len(self.lambdas_list)):
                num_p_sample[lambda_idx] = 0
                num_n_sample[lambda_idx] = 0
                tp[lambda_idx] = 0
                fp[lambda_idx] = 0
                pixel_auc[lambda_idx] = []
                pro_all[lambda_idx] = []

            for def_type in os.listdir(def_path):
                test_path = os.path.join(data_root, cl, 'test', def_type)
                gt_path = os.path.join(data_root, cl, 'ground_truth', def_type)
                test_dataloader = self.get_eval_dataloader(test_path, gt_path)

                for idx, (ts, ts_gt, gt_label) in enumerate(test_dataloader):
                    batch_size = ts.shape[0]
                    with torch.no_grad():
                        d = ts.to(self.device)
                        map = self.models[cl_idx].inference(d)

                    embeddings = [self.infer_embs(map, i, j, tp="testing")
                                  for i in range(self.n_rows)
                                  for j in range(self.n_cols)
                                  ]
                    embs = np.zeros((batch_size, int(self.n_rows * self.n_cols), self.n_test_samples, self.p))

                    # Allocate embeddings
                    for b in range(batch_size):
                        for i in range(self.n_rows):
                            for j in range(self.n_cols):
                                embs[b, i * self.n_cols + j, :, :] = embeddings[i * self.n_cols + j][:, b, :]

                    arg_list = [
                        (
                            embs[:, i * self.n_cols + j, :, :],
                            posteriors[i * self.n_cols + j]
                        )
                        for i in range(self.n_rows)
                        for j in range(self.n_cols)
                    ]
                    return_list = [self.test_one_step(*args) for args in arg_list]

                    pvalues_list = [t[0] for t in return_list]
                    l2_distance_list = [t[1] for t in return_list]
                    var_test_embs_list = [t[2] for t in return_list]

                    # Scores are negative of pvalues as anomaly is larger when pvalue is smaller
                    score_maps = self.get_score_map(pvalues_list, batch_size)
                    score_maps = np.nan_to_num(score_maps)

                    for lambda_idx in range(len(self.lambdas_list)):
                        for b in range(ts.shape[0]):
                            # Load metrics
                            pvalues = [p[b][lambda_idx] for p in pvalues_list]
                            l2_distance = [l[b] for l in l2_distance_list]
                            var_test_embs = [v[b] for v in var_test_embs_list]

                            # Using BH Test controling the FDR
                            n = len(pvalues)
                            thresholds = np.array([0.05 * (k + 1) / n for k in range(n)])
                            maxk = np.argmax(np.where(np.sort(pvalues) < thresholds, np.sort(pvalues), 0))
                            r_set = np.argsort(pvalues)[:maxk]

                            # calculate image-level TRP and FPR
                            if gt_label[b] == 0:
                                num_n_sample[lambda_idx] += 1
                            else:
                                num_p_sample[lambda_idx] += 1

                            if gt_label[b] == 1 and len(r_set) > 0:
                                tp[lambda_idx] += 1
                            elif gt_label[b] == 0 and len(r_set) > 0:
                                fp[lambda_idx] += 1

                            score_map = score_maps[lambda_idx, b, :, :]

                            if "good" not in def_type:
                                # Compute pixel-wise AUCROC and PRO for defect images
                                try:
                                    pixauc = roc_auc_score(ts_gt[b].type(torch.int).flatten(), score_map.flatten())
                                    pro = compute_pro(def_type, ts_gt[b].type(torch.int).numpy(),
                                                      np.expand_dims(score_map, 0))
                                except (ValueError, AssertionError):
                                    pixauc = 0.5
                                    pro = 0
                                pixel_auc[lambda_idx].append(pixauc)
                                pro_all[lambda_idx].append(pro)
                                # print(f"Pixel-AUC: {pixauc}; PRO: {pro}")

                            # Visualization
                            if self.has_visualization:
                                input_img = ts[b].permute(1, 2, 0)
                                self.visualize_result(
                                    input_img, r_set, score_map, pvalues,
                                    l2_distance, var_test_embs, cl,
                                    def_type, b, lambda_idx
                                )

            TPRs = []
            FPRs = []
            for lambda_idx in range(len(self.lambdas_list)):
                tpr = tp[lambda_idx] / num_p_sample[lambda_idx]
                fpr = fp[lambda_idx] / num_n_sample[lambda_idx]
                TPRs.append(tpr)
                FPRs.append(fpr)
                pixel_auc[lambda_idx] = np.mean(pixel_auc[lambda_idx])
                pro_all[lambda_idx] = np.mean(pro_all[lambda_idx])
                print('Lamb idx ', lambda_idx, 'TPR = ', tpr, ', FPR = ', fpr, 'Pixel-AUROC = ', pixel_auc[lambda_idx],
                      ' PRO = ', pro_all[lambda_idx], "\n")
                rf.write(f"lambda idx = {lambda_idx}; TPR = {tpr}; FPR = {fpr}; Pixel-AUROC = {pixel_auc[lambda_idx]}"
                         f"; PRO = {pro_all[lambda_idx]}\n")

            # Compute image level AUC
            TPRs = np.array(TPRs)
            FPRs = np.array(FPRs)
            p = FPRs.argsort()
            TPRs = TPRs[p]
            FPRs = FPRs[p]

            print(
                f"Image AUC:{auc(FPRs, TPRs)}; Pixel AUC: {np.nanmax(list(pixel_auc.values()))}; PRO: {np.nanmax(list(pro_all.values()))}")
            rf.write(
                f"Image AUC:{auc(FPRs, TPRs)}; Pixel AUC: {np.nanmax(list(pixel_auc.values()))}; PRO: {np.nanmax(list(pro_all.values()))}\n")
            with open(results_path + "summary.txt", "a") as f:
                f.write(
                    f"{cl}; {auc(FPRs, TPRs)}; {np.nanmax(list(pixel_auc.values()))}; {np.nanmax(list(pro_all.values()))}\n")
