"""
GRCS: Group Relative Completion Selection
------------------------------------------
A training-free way to improve model outputs using contrastive preference steering.
"""

from grcs.engine import GRCSEngine
from grcs.generator import LMStudioGenerator, OpenAIGenerator
from grcs.builder import GRCSBuilder

__all__ = ["GRCSEngine", "LMStudioGenerator", "OpenAIGenerator", "GRCSBuilder"]
