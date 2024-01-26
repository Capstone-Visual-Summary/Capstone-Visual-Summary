from sympy import E
# If you get en error with not being able to find subclasses
from Database_Classes import DatabaseParent
# Change the imports to: from ..... import *
from Embedding_Classes import EmbeddingParent
from Summarization_Classes import SummarizationParent
from Visualization_Classes import VisualizationParent

from geopandas import GeoDataFrame
from tqdm import tqdm


def OneRUNtoRUNthemALL(**kwargs) -> None:

    database_parent = DatabaseParent()
    embedding_parent = EmbeddingParent()
    summarization_parent = SummarizationParent()
    visualization_parent = VisualizationParent()

    neighbourhood_images, images, neighbourhoods = database_parent.run(
        **kwargs)  # type: tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]

    start_hood = min(len(neighbourhood_images), max(
        0, kwargs['start_hood'])) if 'start_hood' in kwargs else 0
    stop_hood = min(len(neighbourhood_images), max(
        0, kwargs['stop_hood'])) if 'stop_hood' in kwargs else len(neighbourhood_images)
    step_size = min(len(neighbourhood_images), max(
        0, kwargs['step_size'])) if 'step_size' in kwargs else 1

    wanted_hoods = [i for i in range(start_hood, stop_hood, step_size)]

    embedding_neighbourhood = dict()

    for neighbourhood_id, image_ids in neighbourhood_images.items():

        if int(neighbourhood_id) not in wanted_hoods:
            continue

        embeddings = dict()

        image_ids = neighbourhood_images[neighbourhood_id]

        for image_id in image_ids:
            #  remove once v2.x has support for this
            path = images.loc[(images['img_id_com'] ==
                               image_id), 'path'].iloc[0]
            embeddings[str(image_id)] = embedding_parent.run(
                image_id=image_id, img_path=path, **kwargs)  # list[float]

        embedding_neighbourhood[neighbourhood_id] = embeddings

    summaries = dict()

    for neighbourhood_id in embedding_neighbourhood:
        # summaries is of type: tuple[dict[str, list[str]], dict[str, str]], so tuple[clusters, centroids]
        summaries[str(neighbourhood_id)] = summarization_parent.run(
            data=embedding_neighbourhood[str(neighbourhood_id)], **kwargs)

    if 'visualization_version' not in kwargs or kwargs['visualization_version'] >= 3.0:
        visualization_parent.run(summaries=summaries, embeddings=embedding_neighbourhood,
                                 images=images, neighbourhoods=neighbourhoods, **kwargs)
    else:
        for neighbourhood_id in embedding_neighbourhood:
            visualization_parent.run(
                summary=summaries[neighbourhood_id], embeddings=embedding_neighbourhood, neighbourhood_id=neighbourhood_id, images=images, **kwargs)


if __name__ == '__main__':

    fails = []

    for version in tqdm([1.0, 2.0, 2.1, 2.2, 3.0], desc='testing database versions'):
        try:
            OneRUNtoRUNthemALL(database_version=version, start_hood=6, stop_hood=7, step_size=1,  # start_year=2008, end_year=2022,
                               embedder_version=1.0, max_files=1000,
                               summarization_version=1.0, n_clusters=5, n_dimensions=5,
                               visualization_version=3.0, visualize=False,
                               file_name='')
            print(f'{version} passed')
        except Exception as e:
            print(f'{version} FAILED: {e}')
            fails.append(f'database: {version}')

    print('done testing database\n')

    for version in tqdm([1.0, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 3.0, 3.1, 3.2], desc='testing summary versions'):
        try:
            OneRUNtoRUNthemALL(database_version=1.0, start_hood=6, stop_hood=7, step_size=1,  # start_year=2008, end_year=2022,
                               embedder_version=1.0, max_files=1000,
                               summarization_version=version, n_clusters=5, n_dimensions=5,
                               visualization_version=3.0, visualize=False,
                               file_name='')
            print(f'{version} passed')
        except Exception as e:
            print(f'{version} FAILED: {e}')
            fails.append(f'summary: {version}')

    print('done testing summary\n')

    for version in tqdm([1.0, 1.1, '1.2WIP', '2.0', 3.0, 3.1, 3.2], desc='testing visualization versions'):
        try:
            OneRUNtoRUNthemALL(database_version=1.0, start_hood=6, stop_hood=7, step_size=1,  # start_year=2008, end_year=2022,
                               embedder_version=1.0, max_files=1000,
                               summarization_version=1.0, n_clusters=5, n_dimensions=5,
                               visualization_version=version, visualize=True,
                               file_name='')
            print(f'{version} passed')
        except Exception as e:
            print(f'{version} FAILED: {e}')
            fails.append(f'visualization: {version}')

    print('done testing vizualization\n')

    print(fails)
