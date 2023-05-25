#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
import mmcv
from mmseg.core.evaluation import get_palette
import os
import numpy as np
import glob

from PIL import Image
import os.path as osp
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         build_runner, get_dist_info)
import importlib_metadata
from mmcv.utils import build_from_cfg
from mmseg import digit_version
from mmseg.core import DistEvalHook, EvalHook, build_optimizer
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import find_latest_checkpoint, get_root_logger

from sahi.utils.mmdet import (
    download_mmdet_cascade_mask_rcnn_model,
    download_mmdet_config,
)

# import required functions, classes
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image_as_pil
from sahi.slicing import slice_image
import fiona
import rasterio
import rasterio.mask
from osgeo import gdal
from osgeo import gdal, ogr

import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon, Point,LinearRing # (pip install Shapely)


classes = ('Background', 'Palm_Tree')

palette = [[0, 0, 0], [0, 137, 37]]


# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMSegmentation installation
import mmseg
print(mmseg.__version__)


# In[2]:


from sahi.predict import get_sliced_prediction, predict, get_prediction


config_file = r"C:\Users\ahmed.mansour\Example_instance_segm\swin\mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py"#"segformer_mit-b2_512x512_160k_ade20k.py"
checkpoint_file=r'C:\Users\ahmed.mansour\Example_instance_segm\best_bbox_mAP_iter_86000.pth'
detection_model = AutoDetectionModel.from_pretrained(
    model_type='mmdet',
    model_path=checkpoint_file,
    config_path=config_file,
    confidence_threshold=0.4,
    image_size=640,
    device="cuda", # or 'cuda:0'
)


# In[8]:


result = get_sliced_prediction(
   r'000000000007.tif',#r'C:\Users\ahmed.mansour\Subset\Subset.tif',# 
    detection_model,
    slice_height = 512,
    slice_width = 512,
    overlap_height_ratio = 0.3,
    overlap_width_ratio = 0.3,
    perform_standard_pred=False
)


# In[5]:


result.export_visuals(export_dir="",file_name="newimage_slice")


# In[5]:


def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)
        
        # Make a polygon and simplify it
#         poly = Polygon(contour)
#         print(contour)
#         poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(contour)
    return polygons


# In[7]:


schema =  {
      "type": "Feature",
      "geometry": "Polygon",
      "properties": {'name':'str'}

    }

         

#open a fiona object
polyShp = fiona.open('masks.shp', mode='w', driver='ESRI Shapefile',
          schema = schema, crs = "EPSG:4326")
dataset = rasterio.open(r'000000000007.tif')

feature = []
#get list of points
for i,box in enumerate(result.object_prediction_list):
#     print(box.bbox)
    rect = box.bbox
    r_b_p = rect.maxx,rect.maxy
    l_b_p = rect.minx,rect.maxy
    l_u_p = rect.minx,rect.miny
    r_u_p = rect.maxx,rect.miny

    r_b =  dataset.transform * (rect.maxx, rect.maxy) 
    l_b = dataset.transform * (rect.minx,rect.maxy)
    l_u = dataset.transform * (rect.minx,rect.miny)
    r_u = dataset.transform * (rect.maxx,rect.miny)
#     lineString =  [l_u,l_b,r_b,r_u]
#     rowName = 'box'+str(i)
    cat = ""#box.mask.to_coco_segmentation()
    polog = create_sub_mask_annotation(box.mask.bool_mask)
    allpols = []
    for pol in polog:
        subpol = []
        for point in pol:
            subpol.append(dataset.transform *point)
        allpols.append(subpol)
    #save record and close shapefile
    feature.append( {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": allpols
      },
      "properties": {'name' : "mask"}
    })
    
#     feature.append({
#       "type": "Feature",
#       "geometry": {
#         "type": "LineString",
#         "coordinates": lineString
#       },
#       "properties": []
#     })
    rowDict = {
    'geometry' : {'type':'Polygon',
                     'coordinates': allpols
        }, #Here the xyList is in brackets
    'properties': {'name' : "mask"},
    }
    polyShp.write(rowDict)
#close fiona object
polyShp.close()


# In[8]:


with fiona.open("masks.shp", "r") as shapefile:
    print(shapefile.schema)
    shapes = [feature['geometry'] for feature in shapefile]


# In[9]:


with fiona.open("box.shp", "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]


# In[10]:


schema =  {
      "type": "Feature",
      "geometry": "Polygon",
      "properties": {'name':'str'}

    }

         

#open a fiona object
polyShp = fiona.open('boundingbox.shp', mode='w', driver='ESRI Shapefile',
          schema = schema, crs = "EPSG:4326")
dataset = rasterio.open(r'000000000007.tif')

feature = []
#get list of points
for i,box in enumerate(result.object_prediction_list):
#     print(box.bbox)
    rect = box.bbox
    r_b_p = rect.maxx,rect.maxy
    l_b_p = rect.minx,rect.maxy
    l_u_p = rect.minx,rect.miny
    r_u_p = rect.maxx,rect.miny

    r_b =  dataset.transform * (rect.maxx, rect.maxy) 
    l_b = dataset.transform * (rect.minx,rect.maxy)
    l_u = dataset.transform * (rect.minx,rect.miny)
    r_u = dataset.transform * (rect.maxx,rect.miny)
    lineString =  [l_u,l_b,r_b,r_u]
#     print(lineString)
#     rowName = 'box'+str(i)
    cat = ""#box.mask.to_coco_segmentation()
    #save record and close shapefile
#     feature.append( {
#       "type": "Feature",
#       "geometry": {
#         "type": "Polygon",
#         "coordinates": allpols
#       },
#       "properties": {'name' : "mask"}
#     })
    
#     feature.append({
#       "type": "Feature",
#       "geometry": {
#         "type": "LineString",
#         "coordinates": lineString
#       },
#       "properties": []
#     })
    rowDict = {
    'geometry' : {'type':'Polygon',
                     'coordinates': [lineString]
        }, #Here the xyList is in brackets
    'properties': {'name' : "box"},
    }
    polyShp.write(rowDict)
#close fiona object
polyShp.close()


# In[11]:


with fiona.open("masks.shp", "r") as shapefile:
    shapes = [feature['geometry'] for feature in shapefile]
with rasterio.open(r'000000000007.tif') as src:
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=False)
    out_meta = src.meta
    print(out_image.shape,out_image.max() )
    out_meta.update({"driver": "GTiff",
             "height": out_image.shape[1],
             "width": out_image.shape[2],
             "transform": out_transform})

    with rasterio.open("masked.tif", "w", **out_meta) as dest:
        dest.write(out_image)


# In[12]:


# dataset = gdal.Open(file_name)
# band1 = dataset.GetRasterBand(1)
# data = band1.ReadAsArray()
vmin = 0 # minimum value in your data (will be black in the output)
vmax = 256 # minimum value in your data (will be white in the output)
ds = gdal.Translate('fused1.png',  'masked.tif', format='PNG', outputType=gdal.GDT_Byte, scaleParams=[[vmin,vmax]])

