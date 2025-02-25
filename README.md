﻿READ ME

This github repository has been created for the course: Capstone at TU Delft. 

DISCLAIMER: The structure of the code will not change, however, there could be data uploaded after the deadline for ease of visualization. These files (such as ‘Final Result.html’) can be generated by the code in this repo but can take around 2 hours to do so. This file is simply to save time.

DISCLAIMER: Due do Google's Terms of Service, the actual images cannot be uploaded to the github so the visualization step cannot work without these images but this is beyond our control.

The project has been subdivided into five main files:

Main: ties all functions together

Database: handles the creation of the neighbourhoods, based on what shapes are requested, and the image data given.

Embedder: Handles the creation, saving and loading of embeddings based on the images passed to it

Summarization: Takes the embeddings and creates a summary for each neighbourhood

Visualization: Visualizes the results of the summarization.

The main file contains a function to tie together all other files and pass them their required arguments. The arguments are implemented using kwargs, this means all arguments for all functions can be put in the main ‘oneRUNtoRUNthemALL’. Four main arguments are used to define what version of each part to use. defined in the following way:

database\_version (1.0, 2.0, 2.1, 2.2, 3.0)

embedder\_version (1.0, 2.0, 2.1)

summarization\_version (1.0, 2.0 - 2.7, 3.0 - 3.2)

visualization\_version (1.0, 2.0, 3.0)

To use: 

To get a new output to be generated by the software

Database:

Using specific files, it generates a json file with the neighbourhoods and which images are assigned to that neighbourhood. The different version uses different neighbourhoods. A range of years for the image can be selected. It returns a tuple with a dictionary with as keys the neighbourhood id and as value a list with the image ids in a neighbourhood, and 2 GeoDataFrame, one for the images and one for the neighbourhoods.

Version 1.0 uses user made hexagons

Version 2.0 uses Uberhex hexagons with resolution 7

Version 2.1 uses Uberhex hexagons with resolution 8

Version 2.2 uses Uberhex hexagons with resolution 9

Version 3.0 uses GIS data taken from the municipality for the definitions of real-world neighbourhoods

Embeddings:

Version 1.0:

This version uses the full ResNet152 model to make the embeddings. The input is the path to an image file, the output a tensor of size 2048, which represents the embedding.

Version 2.0:

This version uses the ResNet152 model without the last 2 layers, the average pooling layer and the fully connected layer.

Version 2.1:

represents a specific version of the embedding model that uses ResNet152 architecture with triplet loss.

Embedding triplet trainer

This class implements the Triplet Loss method, which is commonly used in metric learning tasks. Triplet Loss aims to learn embeddings such that the distance between the anchor and the positive sample is minimized, while the distance between the anchor and the negative sample is maximized.

FineTuneResNet152: This class is used to fine-tune a pretrained ResNet152 model.

To use this module, create an instance of the TripletLoss class and call the forward method with the anchor, positive, and negative samples. The method will return the triplet loss.

The number of epochs for training is defined at the top of the module. To change the number of epochs, modify the num\_epochs variable.


Summarization:

Version 1.0: applies PCA and KMeans to form clusters and selects the central data point to form the summary

Version 2.x: the different versions run different combinations of algorithms. PCA, TSNE or UMAP for dimensionality reduction and KMeans, Hierarchical clustering or Density clustering for clustering.

Version 3.x: same as 2.x but uses density clustering to first detect the amount of clusters instead of it being an argument


visualization:

version 1.0:

This version saves a visual summary based on the summary provided to it in the form of a dict containing each cluster, which are in turn dicts with keys: cluster, and selected. the cluster holds the IDs of images which are in the cluster, and the selected contains the ID of the image representing this cluster. It saves the image to a PNG file in a subfolder. The filename will be neighborhood {neighbourhood\_id} e {embedding versions} s {summarization version} as the visualization will vary based on what embedding en summarization versions are used for. This image will be used in version 3.0 to visually represent the summaries on a map.

version 2.0:

This version function is to be able to verify the embeddings and summarizations. The summarization will be verified by averaging all embeddings contained in the neighbourhood, and the embeddings which are in the summary, the distance between these two points is the score for the summary. To be able to visually verify the embedding is capturing data about the neighbourhoods, all embeddings of that neighbourhood are averaged and reduced to 3 dimensions using PCA. This way each dimension can be used as one of the three RGB channels. This version works as a support class, it returns its colour and distance in a tuple per neighbourhood but is not visualized itself. The data it calculates is passed on to other functions, to verify the embeddings and summarisation. Its data is also used in version 3.0 to visualize this data on an interactive map.

Version 3.0:

This version generates an interactive KeplerGL map where the summarized neighbourhoods are visible. The neighbourhoods are represent with a color that comes from version 2.0. When hovering over the neighbourhood, an image is shown showing the images that summarise the neighbourhood. These images are also represented as points on the map. This saves the map as an html file and opens the html file.
