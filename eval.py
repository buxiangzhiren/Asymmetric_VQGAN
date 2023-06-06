#!/usr/bin/env python3

import os, wandb
import pandas as pd
from tqdm import tqdm
from saicinpainting.evaluation.data import PrecomputedInpaintingResultsDataset
from saicinpainting.evaluation.evaluator import InpaintingEvaluator, lpips_fid100_f1
from saicinpainting.evaluation.losses.base_loss import SegmentationAwareSSIM, \
    SegmentationClassStats, SSIMScore, LPIPSScore, FIDScore, SegmentationAwareLPIPS, SegmentationAwareFID
from saicinpainting.evaluation.utils import load_yaml


def main(args):
    wandb.login(key='49222ad51163763788e59460ea91552f32605e38')
    run = wandb.init(
        id=args.tag + "_eval",
        name=args.tag + "_eval",
        entity='buxiangzhiren',
        project='Inpainting',
        job_type='train_model',
        config=args,
    )
    config = load_yaml(args.config)
    predict_all = os.path.join(args.predictdir, args.tag, "image_results")
    all_results = os.listdir(predict_all)
    for i_s, each_dir in enumerate(all_results):
        dir = os.path.join(predict_all, each_dir)
        masks = sorted(glob.glob(os.path.join(args.datadir, "*_mask.png")))
        images = [x.replace("_mask.png", ".png") for x in masks]
        pred_filename = [dir + os.path.splitext(fname[len(args.datadir):])[0] + '.png'
                         for fname in images]
        error_list = []
        for i in tqdm(range(len(images))):
            image = images[i]
            mask = masks[i]
            pred = pred_filename[i]
            # image_dir = image
            image = load_image(image)
            # image = image.astype(np.float32) / 255.0
            predmap = load_image(pred)
            mask = np.array(Image.open(mask).convert("L"))
            # mask = np.transpose(mask, (2, 0, 1))
            mask = mask.astype(np.float32) / 255.0
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            masked_predmap = (1 - mask[None, :, :]) * predmap
            masked_img = (1 - mask[None, :, :]) * image
            error = np.sum((masked_predmap - masked_img) ** 2)
            error_list.append(error)
        total_error = np.mean(error_list)
        wandb.log({f'pre_{i_s}': total_error})
        print(total_error)
        dataset = PrecomputedInpaintingResultsDataset(args.datadir,dir, **config.dataset_kwargs)

        metrics = {
            'ssim': SSIMScore(),
            'lpips': LPIPSScore(),
            'fid': FIDScore()
        }
        # enable_segm = config.get('segmentation', dict(enable=False)).get('enable', False)
        # if enable_segm:
        #     weights_path = os.path.expandvars(config.segmentation.weights_path)
        #     metrics.update(dict(
        #         segm_stats=SegmentationClassStats(weights_path=weights_path),
        #         segm_ssim=SegmentationAwareSSIM(weights_path=weights_path),
        #         segm_lpips=SegmentationAwareLPIPS(weights_path=weights_path),
        #         segm_fid=SegmentationAwareFID(weights_path=weights_path)
        #     ))
        evaluator = InpaintingEvaluator(dataset, scores=metrics,
                                        integral_title='lpips_fid100_f1', integral_func=lpips_fid100_f1,
                                        **config.evaluator_kwargs)
        results = evaluator.evaluate()
        wandb.log({f'liips_{i_s}': results[('lpips', 'total')]["mean"]})
        print(f"The results of {i_s}-th epoch is", results)


    # results = pd.DataFrame(results).stack(1).unstack(0)
    # results.dropna(axis=1, how='all', inplace=True)
    # results.to_csv(args.outpath, sep='\t', float_format='%.4f')
    #
    # if enable_segm:
    #     only_short_results = results[[c for c in results.columns if not c[0].startswith('segm_')]].dropna(axis=1, how='all')
    #     only_short_results.to_csv(args.outpath + '_short', sep='\t', float_format='%.4f')
    #
    #     print(only_short_results)
    #
    #     segm_metrics_results = results[['segm_ssim', 'segm_lpips', 'segm_fid']].dropna(axis=1, how='all').transpose().unstack(0).reorder_levels([1, 0], axis=1)
    #     segm_metrics_results.drop(['mean', 'std'], axis=0, inplace=True)
    #
    #     segm_stats_results = results['segm_stats'].dropna(axis=1, how='all').transpose()
    #     segm_stats_results.index = pd.MultiIndex.from_tuples(n.split('/') for n in segm_stats_results.index)
    #     segm_stats_results = segm_stats_results.unstack(0).reorder_levels([1, 0], axis=1)
    #     segm_stats_results.sort_index(axis=1, inplace=True)
    #     segm_stats_results.dropna(axis=0, how='all', inplace=True)
    #
    #     segm_results = pd.concat([segm_metrics_results, segm_stats_results], axis=1, sort=True)
    #     segm_results.sort_values(('mask_freq', 'total'), ascending=False, inplace=True)
    #
    #     segm_results.to_csv(args.outpath + '_segm', sep='\t', float_format='%.4f')
    # else:



if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('--config', type=str, default="configs/autoencoder/eval2_gpu.yaml", help='Path to evaluation config')
    aparser.add_argument('--datadir', type=str, default='/mnt/output/my_dataset/val',
                         help='Path to folder with images and masks (output of gen_mask_dataset.py)')
    aparser.add_argument('--predictdir', type=str, default='/mnt/output/logs/',
                         help='Path to folder with predicts (e.g. predict_hifill_baseline.py)')
    # aparser.add_argument('outpath', type=str, help='Where to put results')
    aparser.add_argument('--tag', type=str, help='wandb log')

    main(aparser.parse_args())
