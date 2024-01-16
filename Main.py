from Database_Classes import DatabaseParent # If you get en error with not being able to find subclasses
from Embedding_Classes import EmbeddingParent # Change the imports to: from ..... import *
from Summarization_Classes import SummarizationParent
from Visualization_Classes import VisualizationParent

from geopandas import GeoDataFrame
import pandas as pd
from tqdm import tqdm
import torch

def OneRUNtoRUNthemALL(**kwargs):
	print('START')
	database_parent = DatabaseParent()
	embedding_parent = EmbeddingParent()
	summarization_parent = SummarizationParent()
	visualization_parent = VisualizationParent()

	print('collecting kwargs')
	database_version = kwargs['database_version'] if 'database_version' in kwargs else -1
	embedder_version = kwargs['embedder_version'] if 'embedder_version' in kwargs else -1
	summarization_version = kwargs['summarization_version'] if 'summarization_version' in kwargs else -1
	visualization_version = kwargs['visualization_version'] if 'visualization_version' in kwargs else -1

	file_name = kwargs['file_name'] if 'file_name' in kwargs else 'Empty'

	print('Database rev:', database_version, 'Embedder rev:', embedder_version, 'Summarization rev:', summarization_version, 'visualization rev:', visualization_version, )

	neighbourhood_images, images, neighbourhoods = database_parent.run(database_version, **kwargs) # type: tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]
 
	start_hood = min(len(neighbourhood_images), max(0, kwargs['start_hood'])) if 'start_hood' in kwargs else 0
	stop_hood = min(len(neighbourhood_images), max(0, kwargs['stop_hood'])) if 'stop_hood' in kwargs else len(neighbourhood_images)
	step_size = step_size = min(len(neighbourhood_images), max(0, kwargs['step_size'])) if 'step_size' in kwargs else 1
    
	embeddings = dict()

	wanted_hoods = [i for i in range(start_hood, stop_hood, step_size)]

for neighbourhood_id, image_ids in tqdm(neighbourhood_images.items(), total=len(neighbourhood_images)):
	neighbourhood_id = '1680'
	image_ids = neighbourhood_images[neighbourhood_id]
	print(image_ids)
	for image_id in image_ids:
		for index, path in enumerate(images.loc[(images['img_id'] == image_id), 'path']):
			unique_img_id = int(image_id) * 4 + index
			embeddings[str(unique_img_id)] = embedding_parent.run(1.0, image_id=unique_img_id, img_path=path, resnet=152) # list[float]
	
	break

# Specify the file path
file_path = 'summarization_data.pth'

# Save the dictionary to a file
torch.save(embeddings, file_path)

	for neighbourhood_id, image_ids in tqdm(neighbourhood_images.items(), total=len(neighbourhood_images)):
		if int(neighbourhood_id) not in range(start_hood, stop_hood, step_size):
			continue

		image_ids = neighbourhood_images[neighbourhood_id]
  
		for image_id in image_ids:
			for index, path in enumerate(images.loc[(images['img_id'] == image_id), 'path']):
				unique_img_id = int(image_id) * 4 + index
				embeddings[str(unique_img_id)] = embedding_parent.run(embedder_version, image_id=unique_img_id, img_path=path, **kwargs) # list[float]

	summarization_parent.run(summarization_version, visualize=True, data=embeddings, **kwargs)

	visualization_parent.run(visualization_version, **kwargs)
 

OneRUNtoRUNthemALL(database_version = 1.0, start_hood = 1, stop_hood = 2, step_size = 1, 
				   embedder_version = 1.0, rerun = False, 
				   summarization_version = 1.0, K_images = 5, N_clusters = 10, N_dimensions = 10, 
				   visualization_version = 1.0, visualize = True,
				   file_name = '')
