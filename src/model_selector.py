from models.rnn import RNNBaseModel
from models.model_lightning import LitRNNBaseModel
from models.trnasformer_decoder import SimpleDecoderOnlyTransformer
from models.lstm import LSTMBaseModel


class ModelSelector:
    available_models = {
        "rnn": RNNBaseModel,
        "transformer_decoder": SimpleDecoderOnlyTransformer,
        "lstm": LSTMBaseModel,
    }

    def __new__(cls, model_name: str, backbone_kwargs=None, lightning_kwargs=None):
        if model_name not in cls.available_models:
            raise ValueError(
                f"Unknown model: {model_name}. Choose from {list(cls.available_models)}"
            )
        base_model_cls = cls.available_models[model_name]
        if not backbone_kwargs:
            raise TypeError("You must provide backbone_kwargs for this model!")
        base_model = base_model_cls(**backbone_kwargs)
        lit_model = LitRNNBaseModel(backbone=base_model, **(lightning_kwargs or {}))
        return base_model, lit_model
