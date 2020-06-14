"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util import util
import torch
import numpy as np
from PIL import Image
from metric.metric import calculate_fid_given_paths

def save_image_(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    image_pil.save(image_path)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    ## Your Implementation Here ##
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        for label, img, in visuals.items():
            im = util.tensor2im(img)
            name = img_path[0].split('/')[-1]
            name = name.split('.')[0]
            img_name = image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(opt.results_dir, label)
            if os.path.isdir(save_path):
                save_image_(im, os.path.join(save_path, img_name))
            else:
                os.makedirs(save_path, exist_ok=True)
                save_image_(im, os.path.join(save_path, img_name))


        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    # evaluate by fid score (target -> source)
    ## Your Implementation Here ##
    target_path_A = os.path.join(opt.results_dir, 'real_A')
    pred_path_A = os.path.join(opt.results_dir, 'fake_A')
    fid_scoreA = calculate_fid_given_paths([target_path_A, pred_path_A], 50, True, 2048)


    target_path_B = os.path.join(opt.results_dir, 'real_B')
    pred_path_B = os.path.join(opt.results_dir, 'fake_B')
    # evaluate by fid score (source -> target)
    ## Your Implementation Here ##
    fid_scoreB = calculate_fid_given_paths([target_path_B, pred_path_B], 50, True, 2048)

    print('source-like target / source fid score = %d \n' % (fid_scoreA))
    print('target-like source / target fid score = %d \n' % (fid_scoreB))

    webpage.save()  # save the HTML
