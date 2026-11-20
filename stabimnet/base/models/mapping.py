from models.hybrid_models import HybridModel, HybridModelv2, HybridModel_SSL, HybridModelv2_SSL, HybridModelv3
from models.comparative_models import HybridModelv2_Comparative, HybridModelv2_Comparative_SSL, HybridModelv3_Comparative, HybridModelv4_Comparative, HybridModelv2_Comparative_Deterministic

model_map = {
    "HybridModel": HybridModel,
    "HybridModelv2": HybridModelv2,
    "HybridModel_SSL": HybridModel_SSL,
    "HybridModelv2_SSL": HybridModelv2_SSL,
    "HybridModelv3": HybridModelv3,
    "HybridModelv2_Comparative": HybridModelv2_Comparative,
    "HybridModelv2_Comparative_SSL": HybridModelv2_Comparative_SSL,
    "HybridModelv3_Comparative": HybridModelv3_Comparative,
    "HybridModelv4_Comparative": HybridModelv4_Comparative,
    "HybridModelv2_Comparative_Deterministic": HybridModelv2_Comparative_Deterministic,
}
