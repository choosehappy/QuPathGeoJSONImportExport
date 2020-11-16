# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +

openslidelevel=0 #level from openslide to read 
tilesize=10000 #size of the tile to load from openslide 
patchsize=32 #patch size needed by our DL model 

minhits=100 #the minimum number of objects needed to be present within a tile for the tile to be computed on
batchsize=1024 #how many patches we want to send to the GPU at a single time
nclasses=2 #number of output classes our model is providing
classnames=["Other","Lymphocyte"] #the names of those classes which will appear in QuPath later on 
colors=[-377282,-9408287] #their associated color, see selection of different color values at the bottom of the file

mask_patches=False #if we woud like to blackout the space around the object of interest, this is determined by how the model was trained

json_fname=r'1L1_nuclei_reg.json' #input geojson file
json_annotated_fname=r'1L1_nuclei_reg_anno.json' #target output geojson file
model_fname="lymph_model.pth" #DL model to use
wsi_fname="1L1_-_2019-09-10_16.44.58.ndpi" #whole slide image fname to load cells from which coincide with the json file

# -

import os
os.environ['PATH'] = 'C:\\research\\openslide\\bin' + ';' + os.environ['PATH'] #can either specify openslide bin path in PATH, or add it dynamically
import openslide
from tqdm.autonotebook import tqdm
from math import ceil
import matplotlib.pyplot as plt

import geojson
from shapely.geometry import shape
from shapely.strtree import STRtree
from shapely.geometry import Point
from shapely.geometry import Polygon

# +
import torch
from torch import nn
from torchsummary import summary
import numpy as np
import cv2
import gzip

device = torch.device('cuda')

def divide_batch(l, n): 
    for i in range(0, l.shape[0], n):  
        yield l[i:i + n,::] 
        
#--- Load your model here    
#model = LoadYourModelHere().to(device)    
#checkpoint = torch.load(model_fname, map_location=lambda storage, loc: storage)  # load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666
#model.load_state_dict(checkpoint["model_dict"])
#model.eval()
#summary(model, (3, 32, 32))

# +

if json_fname.endswith(".gz"):
    with gzip.GzipFile(json_fname, 'r') as f:
        allobjects = geojson.loads(f.read(), encoding= 'ascii')
else:
    with open(json_fname) as f:
        allobjects = geojson.load(f)

print("done loading")
# -

allshapes=[shape(obj["nucleusGeometry"] if "nucleusGeometry" in obj.keys() else obj["geometry"]) for obj in allobjects]
allcenters=[ s.centroid  for s in allshapes]
print("done converting")

for i in range(len(allshapes)):
        allcenters[i].id=i

searchtree = STRtree(allcenters)
print("done building tree")

# +
osh  = openslide.OpenSlide(wsi_fname)
nrow,ncol = osh.level_dimensions[0]
nrow=ceil(nrow/tilesize)
ncol=ceil(ncol/tilesize)

scalefactor=int(osh.level_downsamples[openslidelevel])
paddingsize=patchsize//2*scalefactor

int_coords = lambda x: np.array(x).round().astype(np.int32)

# +
for y in tqdm(range(0,osh.level_dimensions[0][1],round(tilesize * scalefactor)), desc="outer" , leave=False):
    for x in tqdm(range(0,osh.level_dimensions[0][0],round(tilesize * scalefactor)), desc=f"inner {y}", leave=False):            

        tilepoly = Polygon([[x,y],[x+tilesize*scalefactor,y],
                            [x+tilesize*scalefactor,y+tilesize*scalefactor],
                            [x,y+tilesize*scalefactor]])
        hits=searchtree.query(tilepoly)

        if len(hits) < minhits:
            continue

        tile  = np.asarray(osh.read_region((x-paddingsize, y-paddingsize), openslidelevel, 
                                           (tilesize+2*paddingsize,tilesize+2*paddingsize)))[:,:,0:3] #trim alpha

        if mask_patches:
            mask = np.zeros((tile.shape[0:2]),dtype=tile.dtype)
            exteriors = [int_coords(allshapes[hit.id].boundary.coords) for hit in hits]
            exteriors_shifted=[(ext-np.asarray([(x-paddingsize),(y-paddingsize)]))//scalefactor for ext in exteriors]
            cv2.fillPoly(mask, exteriors_shifted,1)

        arr_out = np.zeros((len(hits),patchsize,patchsize,3))
        id_out = np.zeros((len(hits),1))


        #---- get patches from hits within this tile and stick them (and their ids) into matricies
        for hit,arr,id in zip(hits,arr_out,id_out):
            px,py=hit.coords[:][0]  #this way is faster than using hit.x and hit.y, likely because of call stack overhead
            c=int((px-x+paddingsize)//scalefactor)
            r=int((py-y+paddingsize)//scalefactor)
            patch = tile[r - patchsize // 2:r + patchsize // 2, c - patchsize // 2:c + patchsize // 2, :]
            
            if mask_patches:
                maskpatch = mask[r - patchsize // 2:r + patchsize // 2, c - patchsize // 2:c + patchsize // 2]
                patch = np.multiply(patch, maskpatch[:, :, None])
                
            arr[:] = patch
            
            id[:]=hit.id


        #---- process batch

        classids=[]
        for batch_arr in tqdm(divide_batch(arr_out,batchsize),leave=False):
            batch_arr_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2)).type('torch.FloatTensor').to(device)/255
            
            # ---- get results
            #classids.append(torch.argmax( model.img2class(batch_arr_gpu),dim=1).detach().cpu().numpy())
            classids.append(np.random.choice([0,1],arr_out_gpu.shape[0]))
        classids=np.hstack(classids)

        for id,classid in zip(id_out,classids):
            allobjects[int(id)]["properties"]['classification']={'name':classnames[classid],'colorRGB':colors[classid]}

        


# +
# # for debugging
# for i,(c,a) in enumerate(zip(classids,arr_out)):
#     plt.imshow(a/255)
#     plt.show()
#     print(c)
#     if i > 10:
#         break

# +
# # for timing
# # %load_ext line_profiler
# # %lprun -f makeoutput makeoutput()
# makeoutput()
# -

if json_annotated_fname.endswith(".gz"):
    with gzip.open(json_annotated_fname, 'wt', encoding="ascii") as zipfile:
        geojson.dump(allobjects, zipfile)
else:
    with open(json_annotated_fname, 'w') as outfile:
        geojson.dump(allobjects,outfile)



# +

#         "name": "Positive", # add incolors
#         "colorRGB": -377282

#         "name": "Other",
#         "colorRGB": -14336

#         "name": "Stroma",
#         "colorRGB": -6895466


#         "name": "Necrosis",
#         "colorRGB": -13487566

#         "name": "Tumor",
#         "colorRGB": -3670016

#         "name": "Immune cells",
#         "colorRGB": -6268256


#         "name": "Negative",
#         "colorRGB":-9408287


# +
#This code to perform entire polygon with complex objects
#exteriors = [int_coords(geo.coords) for hit in hits for geo in hit.boundary.geoms ] #need this modificatrion for complex structures

#This code to perform by center with complex objects
#exteriors = [int_coords(geo.coords) for hit in hits for geo in allshapes[hit.id].boundary.geoms ]

