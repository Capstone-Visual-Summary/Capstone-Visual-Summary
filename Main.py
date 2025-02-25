from Database_Classes import DatabaseParent
from Embedding_Classes import EmbeddingParent
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
                    - start_year (int): From what year onwards the images should be from. Defaults to 2008.
                    - end_year (int): Until (inclusive) what year the images should be from. Defaults to 2022.
                    - embedder_version (float): The version of the embedder to use. Defaults to the newest version.
                    - max_files (int): Limits the number of embedding files loaded to memory. Defaults to 1000.
                    - summarization_version (float): The version of the summarization algorithm to use. Defaults to the newest version.
                    - n_clusters (int): The number of clusters to create in the summarization algorithm. Defaults to 5.
                    - n_dimensions (int): The number of dimensions to use in the embedding algorithm. Defaults to 25.
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

    neighbourhood_images, images, neighbourhoods = database_parent.run(
        **kwargs)  # type: tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]
    
    max_neighbourhoods = max([int(neighbourhood_id) for neighbourhood_id in neighbourhood_images])

    start_hood = min(max_neighbourhoods, max(
        0, kwargs['start_hood'])) if 'start_hood' in kwargs else 0
    stop_hood = min(max_neighbourhoods + 1, max(
        0, kwargs['stop_hood'])) if 'stop_hood' in kwargs else max_neighbourhoods
    step_size = min(len(neighbourhood_images), max(
        0, kwargs['step_size'])) if 'step_size' in kwargs else 1
    
    kwargs['stop_hood'] = stop_hood

    wanted_hoods = [i for i in range(start_hood, stop_hood, step_size)]
    print('Using neighbourhood ids:', wanted_hoods)

    embedding_neighbourhood = dict()

    for neighbourhood_id, image_ids in tqdm(neighbourhood_images.items(), total=len(neighbourhood_images), desc='Getting embeddings'):
        if int(neighbourhood_id) not in wanted_hoods:
            continue

        embeddings = dict()

        image_ids = neighbourhood_images[neighbourhood_id]

        for image_id in image_ids:
            embeddings[str(image_id)] = embedding_parent.run(
                image_id=image_id, **kwargs)  # list[float]

        embedding_neighbourhood[neighbourhood_id] = embeddings
    summaries = dict()

    for neighbourhood_id in tqdm(embedding_neighbourhood, desc='Generating summaries'):
        # summaries is of type: tuple[dict[str, list[str]], dict[str, str]], so tuple[clusters, centroids]
        summaries[str(neighbourhood_id)] = summarization_parent.run(
            data=embedding_neighbourhood[str(neighbourhood_id)], neighbourhood_id=str(neighbourhood_id), **kwargs)

    if 'visualization_version' not in kwargs or kwargs['visualization_version'] >= 3.0:
        visualization_parent.run(summaries=summaries, embeddings=embedding_neighbourhood,
                                 images=images, neighbourhoods=neighbourhoods, **kwargs)
    else:
        for neighbourhood_id in embedding_neighbourhood:
            print('viusalising neighbourhood:', neighbourhood_id)
            visualization_parent.run(
                summary=summaries[neighbourhood_id], embeddings=embedding_neighbourhood, neighbourhood_id=neighbourhood_id, images=images, **kwargs)
    print('DONE')


if __name__ == '__main__':
    OneRUNtoRUNthemALL(database_version=3.0, start_hood=10, stop_hood=11, step_size=1, start_year=2008, end_year=2015,
                       embedder_version=2.1, max_files=1000,
                       summarization_version=3.0, n_clusters=5, n_dimensions=25,
                       visualization_version=3.0, visualize=True,
                       file_name='')
