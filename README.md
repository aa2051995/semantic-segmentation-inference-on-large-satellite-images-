# semantic-segmentation-inference-on-large-satellite-images-
semantic segmentation inference on large satellite images using Sahi method and mmsegmentaion model and saving the prediction as georeferenced files using GDAl libraries
the challenge was to do inference on large image in the size of two giga. I had to use sahi method to slice the image and do inference on every slice then combine the results.
I used the Gdal library to load the slices to make inference, So I don't need to load the whole image.
