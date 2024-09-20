import argparse
import datetime
import json
import logging
import os
import random
import time
import numpy as np
import unicom
import yaml
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from torch import distributed, optim
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Subset
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torchvision import transforms
from dataset import create_loader, create_sampler, get_dataset, cross_flickr_dataset, cross_coco_dataset, \
    cross_ia_dataset, cross_nuswide_dataset
from evaluation_cvpr import evaluation, itm_eval, evaluate
from model import ImageMlp, TextMlp, FuseTransEncoder, ImageMlpSingle, TextMlpSingle


import utils
from scheduler import create_scheduler
from optim import create_optimizer

from sentence_transformers import SentenceTransformer

from torch.utils.tensorboard import SummaryWriter




def main(args, config,t):
    device = torch.device(args.gpu)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if args.resume:
        # try to continue training
        print('load checkpoint from %s' % args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        start_epoch = checkpoint['epoch'] + 1
        best = checkpoint['best']
        best_epoch = checkpoint['best_epoch']
        # config = checkpoint['config']
    else:
        start_epoch = 0
        best = 0
        best_epoch = 0
        state_dict = None

    print("args: ", args)
    print("config: ", config)
    print("config prefix: ", json.dumps(config, indent=4))

    # for training
    # get model
    # when resume, state_dict is not None, so we can load model from state_dict
    print("Creating model")
    nbits = args.bits#128
    feat_lens = 512
    num_layers, token_size, nhead = 2, 1024, 4
    myFuseTrans = FuseTransEncoder(num_layers, token_size, nhead).to(device)
    myImageMlp = ImageMlp(feat_lens, nbits).to(device)
    myTextMlp = ImageMlp(feat_lens, nbits).to(device)
    if config['dataset_name'] == "flickr25k" :
        from unire.model_25k import unire
        model = unire(args, config, myImageMlp, myTextMlp, myFuseTrans)
    elif config['dataset_name'] == "cross_coco":
        model = unire(args, config, myImageMlp, myTextMlp, myFuseTrans)
    elif config['dataset_name'] == "ia":
        from unire.model_ia import unire
        model = unire(args, config, myImageMlp, myTextMlp, myFuseTrans)
    elif config['dataset_name'] == "nuswide":
        from unire.model_nus import unire
        model = unire(args, config, myImageMlp, myTextMlp, myFuseTrans)

    msg = model.load_state_dict(state_dict)
    print(msg)
    model.to(device)
    unicommodel ,  unicomtransform = unicom.load('ViT-B/32')
    unicommodel =  unicommodel.cuda()
    unicommodel .  eval()



    samplers = [None, None, None]
    if config['dataset_name'] == "flickr25k" :
        train_loader, val_loader, test_loader = cross_flickr_dataset.load_dataset('mirflickr',config['batch_size_train'],model.preprocess)
    elif config['dataset_name'] == "cross_coco":
        train_loader, val_loader, test_loader = cross_coco_dataset.load_dataset(config['data_path'],config['batch_size_train'],model.preprocess,unicomtransform)
    elif config['dataset_name'] == "ia":
        train_loader, val_loader, test_loader = cross_ia_dataset.load_dataset(config['dataset_name'],config['batch_size_train'],model.preprocess)
    elif config['dataset_name'] == "nuswide":
        train_loader, val_loader, test_loader = cross_nuswide_dataset.load_dataset(config['dataset_name'],config['batch_size_train'],model.preprocess)

    else:
    # get dataset
        print("Creating dataset")
        if args.experiment:
            train_dataset, val_dataset, test_dataset = [get_dataset(config['dataset_name'], config['data_path'], split, model.preprocess) for split in [
                'experiment', 'val', "test"]]
        else:
            train_dataset, val_dataset, test_dataset = [get_dataset(
                config['dataset_name'], config['data_path'], split, model.preprocess) for split in ['train', 'val', "test"]]
        # get loader
        train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers, batch_size=[config['batch_size_train'], config[
            'batch_size_test'], config['batch_size_testall']], num_workers=[4, 4, 4], is_trains=[True, False, False], collate_fns=[None, None, None])

    # get distributed model
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # assisant model
    # use sentence transformer to get text softlabel
    txt_enc_assisant = SentenceTransformer('all-mpnet-base-v2').to(device=device)
    if args.distributed:
        txt_enc_assisant = torch.nn.parallel.DistributedDataParallel(txt_enc_assisant, device_ids=[args.gpu])

    # train setting
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)

    # scheduler
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    #  train
    print("Start training")
    start_time = time.time()

    if args.eval:
        print("Start eval")
        score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, args)
        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, args)
        if utils.is_main_process():
            val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
            print(val_result)
            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
            print(test_result)
        # synchronize()
        dist.barrier()
        # release gpu memory
        torch.cuda.empty_cache()
        return


    paramsImage = list(myImageMlp.parameters())
    paramsText = list(myTextMlp.parameters())
    paramsFuse_to_update = list(myFuseTrans.parameters())

    optimizer_FuseTrans = optim.Adam(paramsFuse_to_update, lr=1e-4, betas=(0.5, 0.999))
    optimizer_ImageMlp = optim.Adam(paramsImage, lr=1e-3, betas=(0.5, 0.999))
    optimizer_TextMlp = optim.Adam(paramsText, lr=1e-3, betas=(0.5, 0.999))
    MAX_I2T = 0.0
    MAX_T2I = 0.0
    MAX_EPOCH = 0
    global_step = 0
    # 假设 model 是你的 PyTorch 模型
    writer = SummaryWriter(f"./logs/{config['dataset_name']}/{t}/{args.bits}")
    #writer.add_graph(model)
    reset = 0
    for epoch in range(start_epoch, max_epoch):
        lr_scheduler.step(epoch)
        # set epoch
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, train_loader, optimizer, lr_scheduler, epoch, warmup_steps, device, config,unicommodel,txt_enc_assisant,myImageMlp,myTextMlp,optimizer_ImageMlp,optimizer_TextMlp,myFuseTrans,optimizer_FuseTrans)
        for key, value in train_stats.items():
            if key == 'lr':
                writer.add_scalar(f'lr', float(value), epoch)
            else:
                writer.add_scalar(f'{config["dataset_name"]}/loss/{key}', float(value), epoch)
        if epoch % 10==0 or epoch>=80:
            MAP_I2T, MAP_T2I = evaluate(device,model,myImageMlp,myTextMlp,val_loader,test_loader,myFuseTrans)
            writer.add_scalar(f'{config["dataset_name"]}/train_i2t_{args.bits}', MAP_I2T, epoch)
            writer.add_scalar(f'{config["dataset_name"]}/train_t2i_{args.bits}', MAP_T2I, epoch)

            if(MAP_I2T + MAP_T2I>MAX_I2T+MAX_T2I):
                MAX_I2T =MAP_I2T
                MAX_T2I = MAP_T2I
                MAX_EPOCH = epoch
                reset = 0
                #保存模型
                save_obj = {
                    'model': model.state_dict(),
                    'fuse_model': myFuseTrans.state_dict(),
                    'mlp_image_model': myImageMlp.state_dict(),
                    'mlp_txt_model': myTextMlp.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'best': {'i2t':MAX_I2T,'t2i':MAP_T2I},
                    'best_epoch': MAX_EPOCH,
                }
                path = os.path.join(config['model_name'],config["dataset_name"],str(args.bits))
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(save_obj, os.path.join(path,'checkpoint_best.pth'))

            reset = reset +1

            print(MAP_I2T,MAP_T2I,MAX_I2T,MAX_T2I,MAX_EPOCH)
            if(reset>2):

                break
        # eval
        #score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, args)
       # score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, args)

        # save model and log
        # if utils.is_main_process():
        #     #val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
        #     #print(val_result)
        #     test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
        #     print(test_result)
        #     print("Train stats:", train_stats)
        #
        #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                  #**{f'val_{k}': v for k, v in val_result.items()},
        #                  **{f'test_{k}': v for k, v in test_result.items()},
        #                  'epoch': epoch,
        #                  }
        #     with open(os.path.join(config['logger_name'], "log.txt"), "a") as f:
        #         f.write(json.dumps(log_stats) + "\n")
        #
        #     if test_result['r_mean'] > best:
        #         save_obj = {
        #             'model': model_without_ddp.state_dict(),
        #             'config': config,
        #             'epoch': epoch,
        #             'best': best,
        #             'best_epoch': best_epoch,
        #         }
        #         torch.save(save_obj, os.path.join(config['model_name'], 'checkpoint_best.pth'))
        #         best = test_result['r_mean']
        #         best_epoch = epoch
        #
        #     save_obj = {
        #         'model': model_without_ddp.state_dict(),
        #         'config': config,
        #         'epoch': epoch,
        #         'best': best,
        #         'best_epoch': best_epoch,
        #     }
        #     torch.save(save_obj, os.path.join(
        #         config['model_name'], 'checkpoint_{}.pth'.format(str(epoch).zfill(2))))

        # synchronize()
        #dist.barrier()
        # release gpu memory
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # 关闭 SummaryWriter
    writer.close()
    print('Training time {}'.format(total_time_str))

    # if utils.is_main_process():
    with open( "log.txt", "a") as f:
        f.write("best epoch: %d,i2t:%.4f,t2i:%.4f,%.4f,%.4f,%.4f,%.4f\n" % (MAX_EPOCH,MAX_I2T,MAX_T2I,args.bits,args.i,args.j,args.k))


def train(model, train_loader, optimizer, lr_scheduler, epoch, warmup_steps, device, config, unicommodel,txt_enc_assisant,myImageMlp,myTextMlp,optimizer_ImageMlp,optimizer_TextMlp,myFuseTrans,optimizer_FuseTrans):

    model.train()
    myImageMlp.train()
    myTextMlp.train()
    myFuseTrans.train()
    # set metric logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_contrastive', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cross_modal', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_uni_modal', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('my_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metrics = [
        "tau",
        "cross_tau", "cross_tau_image", "cross_tau_text", "cross_the_softlabel_tau", "cross_the_softlabel_tau_image", "cross_the_softlabel_tau_text",
        "uni_tau", "uni_tau_image", "uni_tau_text", "uni_the_softlabel_tau", "uni_the_softlabel_tau_image", "uni_the_softlabel_tau_text"
    ]
    for val in metrics:
        if hasattr(model.modules, val):
            metric_logger.add_meter(val, utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, image_features, raw_captions, idx,lab) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        # n_clusters = 10  # 假设我们希望将数据分为3个类
        # sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0)
        # labels = sc.fit_predict(image_features)
        # score = silhouette_score(lab, labels)
        # print(f'Silhouette Score: {score}')

        image = image.to(device, non_blocking=True)
        caption = caption.to(device, non_blocking=True)
        # softlabel feature for cross-modal retrieval and uni-modal retrieval
        with torch.no_grad():
            image_features = image_features.to(device, non_blocking=True)
            start_time = time.time()

            if config['dataset_name'] != "ia":
                caption_features = txt_enc_assisant.encode(
                    raw_captions,batch_size=config['batch_size_train'], device=device, show_progress_bar=False, convert_to_tensor=True).to(device, non_blocking=True)
            else:
                caption_features = raw_captions.to(device, non_blocking=True)
            # 结束时间
            end_time = time.time()
            if config['dataset_name'] == "cross_coco":
                image_features = unicommodel.forward_features(image_features)
                image_features = image_features.to(device, non_blocking=True)


        # get loss
        cross_modal_loss, uni_modal_loss, contrastive_loss,my_loss = model(image, caption, image_features, caption_features, epoch, idx)

        loss = my_loss + contrastive_loss+cross_modal_loss + uni_modal_loss
        optimizer_FuseTrans.zero_grad()
        optimizer.zero_grad()
        optimizer_ImageMlp.zero_grad()
        optimizer_TextMlp.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_ImageMlp.step()
        optimizer_TextMlp.step()
        optimizer_FuseTrans.step()
        #print(model.cross_tau)
        # update metric logger
        for val in metrics:
            if hasattr(model.modules, val):
                metric_logger.update(**{val: getattr(model.module, val).item()})
        metric_logger.update(loss_cross_modal=cross_modal_loss.item())
        metric_logger.update(loss_uni_modal=uni_modal_loss.item())
        metric_logger.update(loss_contrastive=contrastive_loss.item())
        metric_logger.update(my_loss=my_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def parser_args():
    parser = argparse.ArgumentParser(description="PyTorch Image Retrieval Training")
    parser.add_argument('--config', type=str, default='configs/vitb32/nuswide/cusa.yaml', help='The config file.')
    parser.add_argument('--eval', action='store_true', help='Is eval?')
    parser.add_argument('--experiment', action='store_true', help='Is experiment?')
    parser.add_argument('--resume', action='store_true', help='Is resume?')
    parser.add_argument('--seed', default=23, type=int, help='Seed for initializing training.')
    parser.add_argument("--num_workers", default=4, type=int, help="The number of workers to use for data loading.")
    parser.add_argument('--distributed', default=False, type=bool, help='Is distributed?')
    parser.add_argument('--checkpoint', type=str, default='', help='The checkpoint file to resume from.')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--gpu', default='cuda:0', help='url used to set up distributed training')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # set env
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # set args
    args = parser_args()
    # set distributed
    utils.init_distributed_mode(args)
    print(torch.cuda.is_available())
    assert not (args.config == '' and args.checkpoint == ''), "config and checkpoint cannot be empty at the same time"
    config = None
    if args.config != '':
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
            config['save_path'] = config['save_path'] + "_seed" + str(args.seed)
            config['logger_name'] = os.path.join(config['save_path'], "log")
            config['model_name'] = os.path.join(config['save_path'], "checkpoints")

    if args.resume and args.checkpoint == '':
        modelList = os.listdir(config['model_name'])
        modelList.sort()
        modelPath = modelList[-2]
        args.checkpoint = os.path.join(config['model_name'], modelPath)

    if utils.is_main_process():
        if not os.path.exists(config['save_path']):
            os.makedirs(config['save_path'])
        # Copy the configuration file to storage
        try:
            # If the file exists
            if os.path.exists(args.config):
                os.system("cp -f %s %s" % (args.config, os.path.join(config['save_path'])+"/"))
        except:
            pass
        if not os.path.exists(config['model_name']):
            os.makedirs(config['model_name'])
        if not os.path.exists(config['logger_name']):
            os.makedirs(config['logger_name'])
        # for i in np.arange(0.2,1.2,0.2):
        #     for j in np.arange(0.1, 1.2, 0.2):
        #         #for k in np.arange(0.1, 1.2 ,0.2):
        # 获取当前时间
        current_time = datetime.datetime.now()

        # 提取年、月、日、时、分、秒
        current_year = current_time.year
        current_month = current_time.month
        current_day = current_time.day
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_second = current_time.second
        # 打印年月日时分秒
        t = f"{current_year}-{current_month}-{current_day} {current_hour}.{current_minute}.{current_second}"
        for bits in [16,32,64,128]:
            for i in[0.1,]:
                for j in[0.1]:
                    for k in [0.01]:
                        args.bits = bits
                        args.i = 2.5
                        args.j = 1
                        args.k= 0.01

        #nus
                        # args.i = 2.5
                        # args.j = 1
                        # args.k= 1
                        # #ia
                        # args.i = 2.5
                        # args.j = 1
                        # args.k= 0.001
                    #args.k = k
                        main(args, config,t)


