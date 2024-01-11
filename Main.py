from Database_Classes import DatabaseParent # If you get en error with not being able to find subclasses
from Embedding_Classes import EmbeddingParent # Change the imports to: from ..... import *
from Summerization_Classes import SummerizationParent
from Visualization_Classes import VisualizationParent

from geopandas import GeoDataFrame

database_parent = DatabaseParent()
embedding_parent = EmbeddingParent()
summerization_parent = SummerizationParent()
visualization_parent = VisualizationParent()

# database_parent.run()  # Grabs latsest version that is not WIP
# database_parent.run("1.0")  # Grabs specific version

neighbourhood_images, images, neighbourhoods = database_parent.run(1.0) # type: tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]
print('gdsffg')
