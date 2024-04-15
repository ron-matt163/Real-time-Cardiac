class Config:
    def __init__(self, feature_size=None):
        if feature_size is None:
            feature_size = 200
        self.feature_size = feature_size
        self.autoencoder_embedding_size = 16
        self.ip_sample_length = 200

config = Config()