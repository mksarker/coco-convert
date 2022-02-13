from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys

from pycocotools import mask as mask_util
import imageio
import json
import numpy as np
import scipy.ndimage

from mask_encoding import encode_rle, parse_segments_from_outlines, parse_xml_annotations, regions_to_rle
from utils import rescale_0_255, image_ids_in


def load_images(raw_train_images_dir, raw_train_annotations_dir, ids, dataset):
    images = []
    rles_list = []
    image_sizes = []
    im_ids = []
    class_names = []
    cat_ids = []
    for id in ids:
        class_n= id.split('_')[0]
        print(class_n)
        class_names.append(class_n)
        if class_n=='benign':
            cat_i=1
            print(cat_i)
        elif class_n=='malignant':
            cat_i=2
            print(cat_i)
        cat_ids.append(cat_i)



        im_path = os.path.join(raw_train_images_dir, id)
        image = imageio.imread(im_path)

        if len(image.shape) == 2:
            image = np.stack([image, image, image]).transpose((1, 2, 0))

        image = image[:, :, :3]  # remove the alpha channel as it is not used

        if dataset == args.dataset_name:
            outline_path = (raw_train_annotations_dir +'/'+ (id.split('.')[0] + '.png')) 

        rles = parse_segments_from_outlines(outline_path, dataset)
        if not rles:
            continue

        rles_list.append(rles)
        images.append(image)
        image_sizes.append(image.shape[:2])
        im_ids.append(id)

    return images, rles_list, image_sizes, im_ids, class_names, cat_ids


def tile_image(I, sz=512, resize=None, order=3):
    print(I.shape)
    height, width, _= I.shape
    import scipy.ndimage

    chunks = []
    names = []
    for h in range(0, height, sz):
        for w in range(0, width, sz):
            w_end = w + sz
            h_end = h + sz
            c = I[w:w_end, h:h_end]
            n = '{}_{}_x_{}_{}'.format(w, w_end, h, h_end)
            if resize:
                c = scipy.ndimage.zoom(c, (resize / float(sz), resize / float(sz), 1), order=order)
            chunks.append(c)
            names.append(n)

    return chunks, names


def get_all_tiles(I, sizes, resize, order=3):
    tiles = []
    names = []

    for sz in sizes:  # 
        c, n = tile_image(I, sz, resize, order=order)
        tiles.extend(c)
        names.extend(n)
        print('chunk created')
    return tiles, names


def filter_masks(M):
    masks = []
    for idx in range(M.shape[2]):  # for each mask channel
        if M[:, :, idx].sum() < 5:
            continue
        masks.append(M[:, :, idx])
    if masks:
        return True, np.stack(masks).transpose((1, 2, 0))  # put channels back to place
    return False, None


def convert_union_mask_to_masks(mask_union):
    from skimage import measure
    assert mask_union.shape[2] == 1

    blobs_labels = measure.label(mask_union[:, :, 0], background=0)
    masks = []
    for idx in range(1, blobs_labels.max() + 1):  # for each mask channel
        masks.append(blobs_labels == idx)

    return np.stack(masks).transpose((1, 2, 0))  # put channels back to place


def preprocess_as_tiles(orig_images, orig_rles_list, orig_im_ids):
    images = []
    rles_list = []
    image_sizes = []
    image_names = []

    for I, rles, im_name in zip(orig_images, orig_rles_list, orig_im_ids):
        M = mask_util.decode(rles)

        mtiles, _ = get_all_tiles(M, [512], 512, order=1)
        tiles, names = get_all_tiles(I, [512], 512, order=3)

        for t, m, n in zip(tiles, mtiles, names):
            success, m = filter_masks(m)
            if success:
                rles_list.append(mask_util.encode(np.asarray(m, order='F')))
                images.append(t)
                image_sizes.append(t.shape[:2])
                image_names.append('{}:{}'.format(im_name, n))
                print('Image')
            else:
                print('Failed Image')
    return images, rles_list, image_sizes, image_names


def get_image_data(image, rles, image_id, image_filename, size, class_name, train_image_dir, cat_id):
    im_metadata = {
        'file_name': image_filename,
        'height': size[0],
        'id': image_id,
        'width': size[1],
        'nuclei_class': class_name,
        # 'is_grey_scale': is_grey_scale_mat(image)
    }

    annotations = []
    global annotation_id
    for rle in rles:
        encoded_segment = encode_rle(rle, annotation_id, image_id, cat_id)
        if encoded_segment['area'] > 0:
            annotations.append(encoded_segment)
            annotation_id += 1
        else:
            from pprint import pprint as pp
            pp(encoded_segment)

    if annotations:
        file_name = os.path.join(train_image_dir , (image_filename ))
        imageio.imsave(file_name, image)

    return im_metadata, annotations


annotation_id = 0


def prepare_cd3(args):
    dataset_name = args.dataset_name
    raw_train_images_dir = os.path.join(dataset_name, args.image_dir)
    raw_train_annotations_dir = os.path.join(dataset_name, args.masks_dir)

    train_images_output_dir = os.path.join(dataset_name, dataset_name) 

    try:
        os.mkdir(train_images_output_dir)
    except:
        pass

    annotations_file_path = os.path.join(dataset_name, args.save_file)

    im_ids = image_ids_in(raw_train_images_dir)
    images, rle_lists, image_sizes, train_image_ids, class_names, cat_ids = load_images(raw_train_images_dir,
                                                                  raw_train_annotations_dir,
                                                                  im_ids,
                                                                  dataset_name)

    im_count = 0
    dataset_structure = {
        'annotations': [],
        'categories': [
             {
            "id": 1,
            "name": "benign"
        },
        {
            "id": 2,
            "name": "malignant"
        }
        ],
        'images': [],
        'info': {
            'description': 'coco format Breast Cancer Ultrasound Dataset',
        }
    }

    for im, im_id, masks, sz, class_name, cat_id in zip(images, train_image_ids, rle_lists, image_sizes, class_names, cat_ids ):
        im_metadata, annotations_res = get_image_data(im, masks, im_count, im_id, sz, class_name,
                                                      train_images_output_dir, cat_id )
        dataset_structure['images'].append(im_metadata)
        dataset_structure['annotations'].extend(annotations_res)
        im_count += 1

    json.dump(dataset_structure, open(annotations_file_path, 'w'))


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare external datasets')
    parser.add_argument('--dataset_name', type=str, default='Aldhayani',  help='dataset name' )

    parser.add_argument('--image_dir', type=str, default='data/test/images',  help='dataset image dir path' )
    parser.add_argument('--masks_dir', type=str, default='data/test/masks',  help='dataset masks dir path' )

    parser.add_argument('--save_file', type=str,default='Aldhayani_test.json',  help='save .json file' )
    return parser.parse_args()


def main(args):
    prepare_cd3(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
