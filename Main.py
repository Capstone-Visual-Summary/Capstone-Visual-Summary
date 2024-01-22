from Database_Classes import DatabaseParent # If you get en error with not being able to find subclasses
from Embedding_Classes import EmbeddingParent # Change the imports to: from ..... import *
from Summarization_Classes import SummarizationParent
from Visualization_Classes import VisualizationParent

from geopandas import GeoDataFrame
from tqdm import tqdm

def OneRUNtoRUNthemALL(**kwargs) -> None:
	"""
	Runs a series of operations on the given input parameters.

	Parameters:
		**kwargs: Keyword arguments for specifying the input parameters. The following parameters are supported:
			- database_version (float): The version of the database to use. Defaults to the newest version.
			- start_hood (int): The starting neighborhood ID. Defaults to 0.
			- stop_hood (int): The stopping neighborhood ID. Defaults to the total number of neighborhoods.
			- step_size (int): The step size for iterating over the neighborhoods. Defaults to 1.
			- start_year (int): From what year onwards the images should be from. Defualts to 2008.
			- end_year (int): Until (inclusive) what year the images should be from. Defualts to 2022.
			- embedder_version (float): The version of the embedder to use. Defaults to the newest version.
			- max_files (int): Limits the number of embedding files loaded to memory. Defaults to 1000.
			- summarization_version (float): The version of the summarization algorithm to use. Defaults to the newest version.
			- K_images (int): The number of images to include in each summary. Defaults to 5.
			- N_clusters (int): The number of clusters to create in the summarization algorithm. Defaults to 3.
			- N_dimensions (int): The number of dimensions to use in the embedding algorithm. Defaults to 5.
			- visualization_version (float): The version of the visualization algorithm to use. Defaults to the newest version.
			- visualize (bool): Whether to visualize the results. Defaults to True.
			- file_name (str): The name of the file to save the results. Defaults to an empty string.

	Returns:
		None
	"""
	print('START')
	database_parent = DatabaseParent()
	embedding_parent = EmbeddingParent()
	summarization_parent = SummarizationParent()
	visualization_parent = VisualizationParent()

	print('Database rev:', kwargs['database_version'] if 'database_version' in kwargs else 'Newest', 
		'| Embedder rev:', kwargs['embedder_version'] if 'embedder_version' in kwargs else 'Newest', 
		'| Summarization rev:', kwargs['summarization_version'] if 'summarization_version' in kwargs else 'Newest', 
		'| visualization rev:', kwargs['visualization_version'] if 'visualization_version' in kwargs else 'Newest', 
		'| file name:', kwargs['file_name'] if 'file_name' in kwargs and kwargs['file_name'] != '' else 'Empty')

	neighbourhood_images, images, neighbourhoods = database_parent.run(**kwargs) # type: tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]

	start_hood = min(len(neighbourhood_images), max(0, kwargs['start_hood'])) if 'start_hood' in kwargs else 0
	stop_hood = min(len(neighbourhood_images), max(0, kwargs['stop_hood'])) if 'stop_hood' in kwargs else len(neighbourhood_images)
	step_size = min(len(neighbourhood_images), max(0, kwargs['step_size'])) if 'step_size' in kwargs else 1

	wanted_hoods = [i for i in range(start_hood, stop_hood, step_size)]
	print('Using neighbourhood ids:', wanted_hoods)


	embedding_neighbourhood = dict()

	for neighbourhood_id, image_ids in tqdm(neighbourhood_images.items(), total=len(neighbourhood_images)):
		if int(neighbourhood_id) not in range(start_hood, stop_hood, step_size):
			continue

		embeddings = dict()

		image_ids = neighbourhood_images[neighbourhood_id]

		for image_id in image_ids:
			# TODO Remove path input to embeddings as embeddings are no longer made during runtime for v1.x
            #  remove once v2.x has support for this
			path = images.loc[(images['img_id_com'] == image_id), 'path'].iloc[0]
			embeddings[str(image_id)] = embedding_parent.run(image_id=image_id, img_path=path, **kwargs) # list[float]
		
		embedding_neighbourhood[neighbourhood_id] = embeddings

	summaries = dict()

	for neighbourhood_id in tqdm(embedding_neighbourhood):
		# summaries is of type: dict[str, dict[str, str | list[str]]], so summaries[neighbourhood_id] = {Cluster X: {selected: image_id, cluster: [image_ids]}}
		summaries[str(neighbourhood_id)] = summarization_parent.run(data=embedding_neighbourhood[str(neighbourhood_id)], **kwargs)

	for neighbourhood_id in embedding_neighbourhood:
		visualization_parent.run(summary = summaries[neighbourhood_id], images = images, **kwargs)
	print('DONE')
 
if __name__ == '__main__':
	OneRUNtoRUNthemALL(database_version=1.0, start_hood=7, stop_hood=8, step_size=1, start_year=2008, end_year=2022,
                       embedder_version=1.0, max_files=1000,
                       summarization_version=1.0, K_images=5, N_clusters=5, N_dimensions=5,
                       visualization_version=1.0, visualize=True,
                       file_name='')


