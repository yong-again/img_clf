from xgboost import XGBClassifier

def get_ensemble_model(config_dict, **kwargs):
    if config_dict["type"] == "XGBoost":
        return XGBClassifier(**config_dict["args"], **kwargs)
    raise NotImplementedError(f"Unknown ensemble model type: {config_dict['type']}")
