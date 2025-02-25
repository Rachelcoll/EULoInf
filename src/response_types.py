from dataclasses import dataclass
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Iterable, Callable, Any

@dataclass
class ModelEvalObjects:
    train_losses: List[np.array]
    accuracies: List[float]
    validation_losses: List[float]
    metadata: Optional[Dict[str, str]] = None
