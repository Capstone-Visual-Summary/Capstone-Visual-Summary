def generate_config(colors):
	layers = [{
		'id': 'Summary Images',
		'type': 'geojson',
		'config': {
			'dataId': 'Summary Images',
			'label': 'Summary Images',
			'color': [255, 255, 255],
			'highlightColor': [252, 242, 26, 255],
			'columns': {
				'geojson': 'geometry'
			},
			'isVisible': True,
			'visConfig': {
				'opacity': 0.6,
				'strokeOpacity': 0.8,
				'thickness': 0.5,
				'strokeColor': [180, 180, 180],
				'colorRange': {
					'name': 'Global Warming',
					'type': 'sequential',
					'category': 'Uber',
					'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']
				},
				'strokeColorRange': {
					'name': 'Global Warming',
					'type': 'sequential',
					'category': 'Uber',
					'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']
				},
				'radius': 10,
				'sizeRange': [0, 10],
				'radiusRange': [0, 50],
				'heightRange': [0, 500],
				'elevationScale': 5,
				'enableElevationZoomFactor': True,
				'stroked': False,
				'filled': True,
				'enable3d': False,
				'wireframe': False
			},
			'hidden': False,
			'textLabel': [{
				'field': None,
				'color': [255, 255, 255],
				'size': 18,
				'offset': [0, 0],
				'anchor': 'start',
				'alignment': 'center'
			}]
		},
		'visualChannels': {
			'colorField': None,
			'colorScale': 'quantile',
			'strokeColorField': None,
			'strokeColorScale': 'quantile',
			'sizeField': None,
			'sizeScale': 'linear',
			'heightField': None,
			'heightScale': 'linear',
			'radiusField': None,
			'radiusScale': 'linear'
		}
	}
	]
	layers_neighbourhoods = [{
		'id': layer_name,
		'type': 'geojson',
		'config': {
			'dataId': layer_name,
			'label': layer_name,
			'color': colors[layer_name],
			'highlightColor': [252, 242, 26, 255],
			'columns': {
				'geojson': 'geometry'
			},
			'isVisible': True,
			'visConfig': {
				'opacity': 0.6,
				'strokeOpacity': 0.8,
				'thickness': 0.5,
				'strokeColor': [255, 255, 255],
				'colorRange': {
					'name': 'Global Warming',
					'type': 'sequential',
					'category': 'Uber',
					'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']
				},
				'strokeColorRange': {
					'name': 'Global Warming',
					'type': 'sequential',
					'category': 'Uber',
					'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']
				},
				'radius': 10,
				'sizeRange': [0, 10],
				'radiusRange': [0, 50],
				'heightRange': [0, 500],
				'elevationScale': 5,
				'enableElevationZoomFactor': True,
				'stroked': False,
				'filled': True,
				'enable3d': False,
				'wireframe': False
			},
			'hidden': False,
			'textLabel': [{
				'field': None,
				'color': [255, 255, 255],
				'size': 18,
				'offset': [0, 0],
				'anchor': 'start',
				'alignment': 'center'
			}]
		},
		'visualChannels': {
			'colorField': None,
			'colorScale': 'quantile',
			'strokeColorField': None,
			'strokeColorScale': 'quantile',
			'sizeField': None,
			'sizeScale': 'linear',
			'heightField': None,
			'heightScale': 'linear',
			'radiusField': None,
			'radiusScale': 'linear'
		}
	} for layer_name in colors]


	layers.extend(layers_neighbourhoods)

	config = {
		'version': 'v1',
		'config': {
			'visState': {
				'filters': [],
				'layers': layers,
				'interactionConfig': {
					'tooltip': {
						'fieldsToShow': {
							layer_name: [
								{'name': 'neighbourhood_id', 'format': None},
								{'name': 'BU_NAAM', 'format': None},
								{'name': '<img>-summary', 'format': None}
							]
						for layer_name in colors},
						'compareMode': False,
						'compareType': 'absolute',
						'enabled': True
					},
					'brush': {
						'size': 0.5,
						'enabled': False
					},
					'geocoder': {
						'enabled': False
					},
					'coordinate': {
						'enabled': False
					}
				},
				'layerBlending': 'normal',
				'splitMaps': [],
				'animationConfig': {
					'currentTime': None, 'speed': 1
				}
			},
			'mapState': {
				'bearing': 0,
				'dragRotate': False,
				'latitude': 52.01393523522903,
				'longitude': 4.351426030025631,
				'pitch': 0,
				'zoom': 11.886249027093603,
				'isSplit': False,
			},
			'mapStyle': {
				'styleType': 'dark',
				'topLayerGroups': {},
				'visibleLayerGroups': {
					'label': True,
					'road': True,
					'border': False,
					'building': True,
					'water': True,
					'land': True,
					'3d building': False
				},
				'threeDBuildingColor': [9.665468314072013, 17.18305478057247, 31.1442867897876],
				'mapStyles': {}
			}
		}
	}

	return config
