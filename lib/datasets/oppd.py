from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
from os import listdir
import os
import pickle
import pdb


import numpy as np
import scipy.sparse
import json

from model.config import cfg
from datasets.imdb import imdb


class oppd(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'oppd_frag_' + image_set)

        # name, paths
        self._year = '2020'
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, 'OPPD_devkit', 'OPPD_DATA')
        self._devkit_path = osp.join(cfg.DATA_DIR, 'OPPD_devkit')

        # API
        # TODO
        # self._OPPD=OPPD()

        self._classes = ('__background__', 
                                'ALOMY', 'ANGAR', 'APESV','ARTVU','AVEFA','BROST','BRSNN',
                                'CAPBP','CENCY','CHEAL','CHYSE','CIRAR','CONAR', 'EPHHE',
                                'EPHPE', 'EROCI', 'FUMOF','GALAP', 'GERMO', 'LAPCO','LOLMU',
                                'LYCAR', 'MATCH', 'MATIN', 'MELNO', 'MYOAR', 'PAPRH', 'PLALA', 
                                'PLAMA', 'POAAN','POLAV','POLCO','POLLA','POLPE','RUMCR',
                                'SENVU','SINAR','SOLNI','SONAS','SONOL','STEME','THLAR',
                                'URTUR','VERAR','VERPE','VICHI','VIOAR','PPPMM', 'PPPDD', 'other')

        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        
        print('list of classes: ', self._classes)
        print('num of classes: ', self.num_classes)

        self._image_ext = '.jpg'

        # dict of all the data in each class
        self._image_index = self.list_imgs() # index images
        self._roidb_handler = self.gt_roidb # load labels


    def list_imgs(self):
        list_imgs=[]
        for filename in listdir(self._data_path):
            if osp.splitext(filename)[1] == self._image_ext:
                list_imgs.append(filename)
        return list_imgs

    def image_path_at(self, i):
        return osp.join(self._data_path, self._image_index[i])

    def gt_roidb(self):
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # FIXME DBG
        # cache_file="/home/yl/pj29/weeds/tf-faster-rcnn-cpu/data/cache/voc_2007_trainval_gt_roidb.pkl"
        
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
            # gt_roidb=roidb
        else:
            # ground truth dictionaries
            # FIXME
            gt_roidb = [self._load_oppd_annotation(index)
                    for index in self._image_index]

            # TODO do I need to shuffle it here?

            with open(cache_file, 'wb') as fid:
              pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb 

        assert osp.exists(self._devkit_path), \
          'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert osp.exists(self._data_path), \
          'Path does not exist: {}'.format(self._data_path)


    def _load_oppd_annotation(self, img_filename):
        """
        load dataset data
        """        
        name=osp.splitext(img_filename)[0]
        annotation_path=name+'.json'

        # TODO load the annotations
        annotation_path=img_filename[:-3]+'json'

        f = open(osp.join(self._data_path ,annotation_path))
        annotations = json.load(f)

        if annotations['filename'] != img_filename:
            raise Exception("error, annotation file and img file does not match")

        num_objs= len(annotations['plants'])
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        for ix, annotation in enumerate(annotations['plants']):
            x1=annotation['bndbox']['xmin']
            y1=annotation['bndbox']['ymin']
            x2=annotation['bndbox']['xmax']
            y2=annotation['bndbox']['ymax']
            
            class_name = annotation['eppo'].strip()

            cls=self._class_to_ind[class_name]

            if not (cls == 10 or cls == 48 or cls == 14 or cls == 49):
                pdb.set_trace()


            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        f.close()

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}


    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)

        # TODO write the eval script
        self._do_python_eval(output_dir)

        # FIXME DBG
        # if self.config['cleanup']:
        if False:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)


    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
          if cls == '__background__':
            continue
          print('Writing {} OPPD results file'.format(cls))
          filename = osp.join(self._devkit_path, 'eval_oppd_'+cls+'.txt')
          with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self.image_index):
              dets = all_boxes[cls_ind][im_ind]
              if dets == []:
                continue
              # the VOCdevkit expects 1-based indices
              for k in range(dets.shape[0]):
                f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                        format(index, dets[k, -1],
                               dets[k, 0] + 1, dets[k, 1] + 1,
                               dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
          self._devkit_path,
          'VOC' + self._year,
          'Annotations',
          '{:s}.xml')
        imagesetfile = os.path.join(
          self._devkit_path,
          'VOC' + self._year,
          'ImageSets',
          'Main',
          self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
          os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
          if cls == '__background__':
            continue
          filename = self._get_voc_results_file_template().format(cls)
          rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
          aps += [ap]
          print(('AP for {} = {:.4f}'.format(cls, ap)))
          with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
          print(('{:.3f}'.format(ap)))
        print(('{:.3f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')