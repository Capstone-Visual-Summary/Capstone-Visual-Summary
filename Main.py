from Database_Classes import DatabaseParent # If you get en error with not being able to find subclasses
from Embedding_Classes import EmbeddingParent # Change the imports to: from ..... import *
from Summarization_Classes import SummarizationParent
from Visualization_Classes import VisualizationParent

from geopandas import GeoDataFrame
import pandas as pd
from tqdm import tqdm

database_parent = DatabaseParent()
embedding_parent = EmbeddingParent()
summarization_parent = SummarizationParent()
visualization_parent = VisualizationParent()

# database_parent.run()  # Grabs latsest version that is not WIP
# database_parent.run("1.0")  # Grabs specific version

neighbourhood_images, images, neighbourhoods = database_parent.run(1.0) # type: tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]

embeddings = dict()

# embeddings[str(int(0) * 4 + 0)] = embedding_parent.run(img_path='image_0_s_a.png', resnet=152)
# print(embeddings)

for neighbourhood_id, image_ids in tqdm(neighbourhood_images.items(), total=len(neighbourhood_images)):
	neighbourhood_id = '1680'
	image_ids = neighbourhood_images[neighbourhood_id]
	print(image_ids)
	for image_id in image_ids:
		for index, path in enumerate(images.loc[(images['img_id'] == image_id), 'path']):
			embeddings[str(int(image_id) * 4 + index)] = embedding_parent.run(img_path=path, resnet=152) # list[float]
	
	break
# print(embeddings)
df = pd.DataFrame.from_dict(embeddings).T
# print(df)
summarization_parent.run(visualize=True, data=df)
