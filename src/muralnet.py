import os
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy


class MuralNet():
    def __init__(self, config):
        self.config = config
        model_name=config.MODELNAME

        print("loading \""+model_name +'\" model')
        self.debug = False
        self.model_name = model_name
        # self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config)
        self.inpaint_model.to(config.DEVICE)
        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # if len(config.GPU) > 1:
        #     self.inpaint_model = nn.DataParallel(self.inpaint_model, config.GPU)
        #     self.psnr = nn.DataParallel(self.psnr, config.GPU)
        #     self.edgeacc = nn.DataParallel(self.edgeacc, config.GPU)
        # self.inpaint_model = nn.DataParallel(self.inpaint_model)
        # self.inpaint_model = self.inpaint_model.cuda()

        print(str(self.config.MODE))
        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.merged_results_path = os.path.join(config.PATH, 'merged_outputs')

        if config.RESULTS is not None:
            self.merged_results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        self.inpaint_model.load()

    def save(self):
        self.inpaint_model.save()


    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.inpaint_model.train()

                images, images_gray, edges, masks = self.cuda(*items)
                print(type(images))

                # inpaint model
                # train
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

                # backward
                self.inpaint_model.backward(gen_loss, dis_loss)
                iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        total = len(self.val_dataset)

        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, images_gray, edges, masks = self.cuda(*items)

            # inpaint model
            # eval
            outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

            # metrics
            psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
            mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
            logs.append(('psnr', psnr.item()))
            logs.append(('mae', mae.item()))

            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

    def test(self):
        self.inpaint_model.eval()

        create_dir(self.merged_results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )
        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, edges, masks = self.cuda(*items)
            index += 1

            mask1 = masks
            images1 = images

            outputs1, outputs2 = self.inpaint_model(images, edges, masks, returnInput=False,coarseOnly=False)

            outputs_merged = (outputs2 * mask1) + (images1 * (1 - mask1))

            output_merged = self.postprocess(outputs_merged)[0]

            path_merged = os.path.join(self.merged_results_path, name)
            print(index, name)

            imsave(output_merged, path_merged)

            if self.debug:
                edges = self.postprocess(1 - edges)[0]
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')

                imsave(edges, os.path.join(self.merged_results_path, fname + '_edge.' + fext))
                imsave(masked, os.path.join(self.merged_results_path, fname + '_masked.' + fext))

        if index == 0:
            print("check the test folder to make sure the folder is not none")


    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, images_gray, edges, masks = self.cuda(*items)


        # inpaint model
        iteration = self.inpaint_model.iteration
        coarseonly=False
        if iteration<self.config.COARSE_ITERS:
            coarseonly=True
        inputs = (images * (1 - masks)) + masks
        coarse, outputs, inputs2 = self.inpaint_model(images, edges, masks, returnInput=True,coarseOnly=coarseonly)
        outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 8:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(edges),
            self.postprocess(coarse),
            self.postprocess(inputs2),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )


        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
