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
        self._models_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._cache_valid = False
        
        # No fallback mappings - use only database-configured models
    
    async def get_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available models from frontend API.
        Returns a dict mapping model keys to complete model information.
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
                    
                    # Store complete model information without conversion
                    # Frontend returns: {id: {name, provider, openrouterId, ...}}
                    models = {}
                    
                    # Handle both formats
                    models_dict = models_data.get("models", models_data) if isinstance(models_data, dict) else {}
                    
                    if models_dict:
                        for model_id, model_info in models_dict.items():
                            # Store complete model information
                            openrouter_id = model_info.get('openrouterId') or model_info.get('openrouter_id')
                            
                            if openrouter_id:
                                models[model_id] = {
                                    'id': model_id,
                                    'name': model_info.get('name', model_id),
                                    'provider': model_info.get('provider', 'Unknown'),
                                    'openrouterId': openrouter_id,
                                    'description': model_info.get('description', ''),
                                    'color': model_info.get('color', 'bg-gray-600'),
                                    'avatar': model_info.get('avatar', '🤖'),
                                    'capabilities': model_info.get('capabilities', {}),
                                    'type': model_info.get('type', 'free')
                                }
                                logger.debug("Loaded complete model info", 
                                           model_id=model_id, 
                                           model_name=model_info.get('name'),
                                           openrouter_id=openrouter_id)
                            else:
                                logger.warning("No OpenRouter ID found for model - model will be skipped", 
                                             model_id=model_id,
                                             model_info=model_info)
                    
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
    
    async def get_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """
        Get specific model information by key.
        Returns None if model not found.
        """
        # Ensure cache is populated
        if not self._cache_valid or not self._models_cache:
            logger.info("Models cache not valid, refreshing from API", model_key=model_key)
            await self.get_models()
        
        # Check the cache (from database/API)
        if self._models_cache and model_key in self._models_cache:
            return self._models_cache[model_key]
        
        logger.warning("Model not found in database", 
                      model_key=model_key,
                      available_models=list(self._models_cache.keys()) if self._models_cache else [])
        return None
    
    async def get_openrouter_id(self, model_key: str) -> Optional[str]:
        """
        Get OpenRouter ID for a specific model.
        Returns None if model not found.
        """
        model_info = await self.get_model(model_key)
        return model_info.get('openrouterId') if model_info else None
    
    def invalidate_cache(self):
        """Invalidate the models cache to force refresh on next request"""
        self._cache_valid = False
        self._models_cache = None
        logger.debug("Models cache invalidated")
    
    async def is_model_available(self, model_key: str) -> bool:
        """Check if a model is available"""
        if not self._cache_valid or not self._models_cache:
            await self.get_models()
            
        return self._models_cache is not None and model_key in self._models_cache


# Global models service instance
models_service = ModelsService()