"""
House Price Prediction Project
A lightweight, resource-efficient ML pipeline for predicting house prices.

Author: Claude Code Assistant
Version: 2.0.0
Date: 2025-01-28
"""

__version__ = "2.0.0"
__author__ = "Claude Code Assistant"
__email__ = "noreply@anthropic.com"
__description__ = "Lightweight house price prediction ML pipeline"

from .house_price_predictor import HousePricePredictor
from .enhanced_predictor import EnhancedHousePricePredictor

__all__ = ["HousePricePredictor", "EnhancedHousePricePredictor"]