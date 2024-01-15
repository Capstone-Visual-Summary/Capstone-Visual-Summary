from Database_Classes import DatabaseParent # If you get en error with not being able to find subclasses
from Embedding_Classes import EmbeddingParent # Change the imports to: from ..... import *
from Summarization_Classes import SummarizationParent
from Visualization_Classes import VisualizationParent

from geopandas import GeoDataFrame
import pandas as pd

database_parent = DatabaseParent()
embedding_parent = EmbeddingParent()
summarization_parent = SummarizationParent()
visualization_parent = VisualizationParent()

# database_parent.run()  # Grabs latsest version that is not WIP
# database_parent.run("1.0")  # Grabs specific version

neighbourhood_images, images, neighbourhoods = database_parent.run(1.0) # type: tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]

embeddings = dict()

for neighbourhood_id, image_ids in neighbourhood_images.items():
	for image_id in image_ids:
		for index, path in enumerate(images.loc[(images['img_id'] == 0), 'path']):
			embeddings[str(int(image_id) * 4 + 1)] = embedding_parent.run(img_path=path) # list[float]

		break

print(embeddings)
