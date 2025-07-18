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
        # Handle None and basic immutable types that don't need processing
        if data is None or isinstance(data, (bool, int, float, complex)):
            return data
        
        # Handle strings - the core normalization
        if isinstance(data, str):
            try:
                # Normalize Unicode characters to ASCII
                normalized = unicodedata.normalize("NFKD", data)
                ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
                return ascii_text
            except (UnicodeError, TypeError):
                # Fallback for problematic strings
                return str(data)
        
        # Handle bytes - convert to string then normalize
        if isinstance(data, bytes):
            try:
                decoded = data.decode('utf-8', errors='ignore')
                return self.normalize_text(decoded)
            except UnicodeError:
                return data.decode('ascii', errors='ignore')
        
        # Handle lists - recursively process all elements
        if isinstance(data, list):
            return [self.normalize_text(item) for item in data]
        
        # Handle tuples - recursively process all elements, preserve immutability
        if isinstance(data, tuple):
            return tuple(self.normalize_text(item) for item in data)
        
        # Handle sets - recursively process all elements
        if isinstance(data, set):
            return {self.normalize_text(item) for item in data}
        
        # Handle dictionaries - recursively process both keys and values
        if isinstance(data, dict):
            normalized_dict = {}
            for key, value in data.items():
                # Normalize both keys and values
                normalized_key = self.normalize_text(key)
                normalized_value = self.normalize_text(value)
                normalized_dict[normalized_key] = normalized_value
            return normalized_dict
        
        # Handle other iterables (but not strings/bytes which are handled above)
        if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            try:
                # Try to reconstruct the same type
                return type(data)(self.normalize_text(item) for item in data)
            except (TypeError, ValueError):
                # If reconstruction fails, return as list
                return [self.normalize_text(item) for item in data]
        
        # For any other type, return as-is (custom objects, functions, etc.)
        return data