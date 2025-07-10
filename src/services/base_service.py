"""
Base Service Class - Shared functionality for all services.
"""

import unicodedata
from typing import Any


class BaseService:
    """Base class for all services with common functionality."""
    
    def normalize_text(self, data: Any) -> Any:
        """
        Recursively normalize Unicode text to clean ASCII in any data structure.
        
        Robustly handles nested structures of arbitrary depth with proper type checking.
        """

        if data is None or isinstance(data, (bool, int, float, complex)):
            return data
        

        if isinstance(data, str):
            try:

                normalized = unicodedata.normalize("NFKD", data)
                ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
                return ascii_text
            except (UnicodeError, TypeError):

                return str(data)
        

        if isinstance(data, bytes):
            try:
                decoded = data.decode('utf-8', errors='ignore')
                return self.normalize_text(decoded)
            except UnicodeError:
                return data.decode('ascii', errors='ignore')
        

        if isinstance(data, list):
            return [self.normalize_text(item) for item in data]
        

        if isinstance(data, tuple):
            return tuple(self.normalize_text(item) for item in data)
        

        if isinstance(data, set):
            return {self.normalize_text(item) for item in data}
        

        if isinstance(data, dict):
            normalized_dict = {}
            for key, value in data.items():

                normalized_key = self.normalize_text(key)
                normalized_value = self.normalize_text(value)
                normalized_dict[normalized_key] = normalized_value
            return normalized_dict
        

        if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            try:

                return type(data)(self.normalize_text(item) for item in data)
            except (TypeError, ValueError):

                return [self.normalize_text(item) for item in data]
        

        return data