"""
Models Service Module

Gestisce il recupero dei modelli dall'API del frontend invece che dai file di configurazione.
Fornisce un'interfaccia centralizzata per accedere ai modelli disponibili.
"""

import httpx
from typing import Dict, Any, Optional
from logging_config import get_context_logger
from config_manager import get_config

logger = get_context_logger(__name__)


class ModelsService:
    """Service for retrieving models from frontend API"""
    
    def __init__(self):
        self.config = get_config()
        self._models_cache: Optional[Dict[str, str]] = None
        self._cache_valid = False
    
    async def get_models(self) -> Dict[str, str]:
        """
        Get all available models from frontend API.
        Returns a dict mapping model keys to OpenRouter model IDs.
        """
        if self._cache_valid and self._models_cache:
            return self._models_cache
            
        try:
            frontend_url = self.config.get_frontend_url()
            timeout = self.config.get_frontend_timeout()
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{frontend_url}/api/models")
                
                if response.status_code == 200:
                    models_data = response.json()
                    
                    # Convert frontend API response to backend format
                    # Frontend returns: {"models": {id: {name, provider, ...}}}
                    # Backend needs: {key: openrouter_id}
                    models = {}
                    
                    if "models" in models_data:
                        for model_id, model_info in models_data["models"].items():
                            # Use model_id as both key and OpenRouter model ID
                            models[model_id] = model_id
                    
                    self._models_cache = models
                    self._cache_valid = True
                    
                    logger.info("Models retrieved from frontend API", 
                               model_count=len(models),
                               models=list(models.keys()))
                    
                    return models
                else:
                    logger.error("Failed to fetch models from frontend API",
                               status_code=response.status_code,
                               response_text=response.text)
                    return {}
                    
        except Exception as e:
            logger.error("Error fetching models from frontend API",
                        error=str(e),
                        error_type=type(e).__name__)
            return {}
    
    def get_model(self, model_key: str) -> Optional[str]:
        """
        Get specific model OpenRouter ID by key.
        Returns None if model not found.
        """
        if not self._cache_valid or not self._models_cache:
            # If cache is not valid, we can't do async call here
            # Return the model_key as-is (assuming it's already the OpenRouter ID)
            logger.warning("Models cache not valid, returning model key as-is",
                          model_key=model_key)
            return model_key
            
        return self._models_cache.get(model_key)
    
    def invalidate_cache(self):
        """Invalidate the models cache to force refresh on next request"""
        self._cache_valid = False
        self._models_cache = None
        logger.debug("Models cache invalidated")
    
    def is_model_available(self, model_key: str) -> bool:
        """Check if a model is available"""
        if not self._cache_valid or not self._models_cache:
            # If cache is not valid, assume model might be available
            return True
            
        return model_key in self._models_cache


# Global models service instance
models_service = ModelsService()