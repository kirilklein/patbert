from sequence_creators import BaseCreator


class FeatureMaker():
    def __init__(self, config):
        self.config = config

        self.features = {}

        self.pipeline = self.create_pipeline()

    def __call__(self):
        concepts = None
        for creator in self.pipeline:
            concepts = creator(concepts)

        features = self.create_features(concepts)

        return features
    
    def create_pipeline(self):
        features = list(self.config.features.keys())[0]
        if 'background' in features:
            features.remove('background')
            features.append('background')

        creators = {creator.feature: creator for creator in BaseCreator.__subclasses__()}

        pipeline = []
        for feature in self.config.features:
            pipeline.append(creators[feature](self.config))
            if feature != 'background':
                self.features.setdefault(feature, [])

        return pipeline

    def create_features(self, concepts):
        def add_to_features(patient):
            for feature, value in self.features.items():
                value.append(patient[feature.upper()].tolist())
        concepts.groupby('PID').apply(add_to_features)

        return self.features

