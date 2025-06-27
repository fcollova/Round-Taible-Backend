"""
OpenRouter Client Module

Gestisce tutte le interazioni con l'API OpenRouter:
- Chat completions
- Rate limiting
- Error handling
- Model mappings
- Circuit breaker pattern
"""

import httpx
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from logging_config import get_context_logger, performance_metrics

logger = get_context_logger(__name__)


class ModelStatus(Enum):
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited" 
    BANNED = "banned"
    ERROR = "error"


@dataclass
class ModelState:
    status: ModelStatus = ModelStatus.AVAILABLE
    last_request: Optional[datetime] = None
    error_count: int = 0
    ban_until: Optional[datetime] = None
    last_error: Optional[str] = None
    cooldown_until: Optional[datetime] = None


class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", timeout: int = 30):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Model state tracking
        self.model_states: Dict[str, ModelState] = {}
        
        # Rate limiting configuration
        self.rate_limits = {
            'free_models': timedelta(seconds=10),  # More conservative for free models
            'paid_models': timedelta(seconds=1)
        }
        
        # Free models that need special handling
        self.free_models = {
            'meta-llama/llama-3.1-8b-instruct:free',
            'mistralai/mistral-7b-instruct:free',
            'google/gemma-7b-it:free'
        }

    async def chat_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Create a chat completion using OpenRouter API with comprehensive error handling
        """
        # Check if model is available
        if not await self._is_model_available(model):
            model_state = self.model_states.get(model, ModelState())
            raise Exception(f"Model {model} is not available: {model_state.status.value}")
        
        # Apply rate limiting
        await self._apply_rate_limit(model)
        
        # Prepare request
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://round-taible.vercel.app",
            "X-Title": "Round TAIble Debate Platform"
        }
        
        start_time = datetime.now()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.debug("Sending OpenRouter request",
                           model=model,
                           message_count=len(messages),
                           max_tokens=kwargs.get('max_tokens', 'default'),
                           temperature=kwargs.get('temperature', 'default'))
                
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                )
                
                response_time = (datetime.now() - start_time).total_seconds()
                
                # Log performance metrics
                performance_metrics.log_openrouter_call(
                    model=model,
                    response_time=response_time,
                    status_code=response.status_code,
                    tokens_used=response.json().get('usage', {}).get('total_tokens', 0) if response.status_code == 200 else 0
                )
                
                # Handle response
                await self._handle_response(model, response, response_time)
                
                if response.status_code == 200:
                    self._mark_model_success(model)
                    result = response.json()
                    
                    logger.info("OpenRouter request successful",
                              model=model,
                              response_time=response_time,
                              tokens_used=result.get('usage', {}).get('total_tokens', 0),
                              finish_reason=result.get('choices', [{}])[0].get('finish_reason', 'unknown'))
                    
                    return result
                else:
                    error_msg = await self._handle_error_response(model, response)
                    raise Exception(error_msg)
                    
        except httpx.TimeoutException:
            self._mark_model_error(model, "Request timeout")
            logger.error("OpenRouter request timeout", model=model, timeout=self.timeout)
            raise Exception(f"Request timeout for model {model}")
            
        except httpx.ConnectError:
            self._mark_model_error(model, "Connection error")
            logger.error("OpenRouter connection error", model=model)
            raise Exception(f"Connection error for model {model}")
            
        except Exception as e:
            self._mark_model_error(model, str(e))
            logger.error("OpenRouter request failed", model=model, error=str(e))
            raise

    async def _is_model_available(self, model: str) -> bool:
        """Check if a model is currently available for use"""
        model_state = self.model_states.get(model, ModelState())
        current_time = datetime.now()
        
        # Check if model is banned
        if model_state.ban_until and current_time < model_state.ban_until:
            logger.debug("Model is banned", model=model, ban_until=model_state.ban_until)
            return False
        
        # Check if model is in cooldown
        if model_state.cooldown_until and current_time < model_state.cooldown_until:
            logger.debug("Model is in cooldown", model=model, cooldown_until=model_state.cooldown_until)
            return False
        
        # Check error count for free models with auto-recovery
        if model in self.free_models and model_state.error_count >= 3:
            # Auto-recovery: reset error count after 5 minutes
            if model_state.last_request and (current_time - model_state.last_request) > timedelta(minutes=5):
                logger.info("Auto-recovering free model after cooldown", model=model, error_count=model_state.error_count)
                model_state.error_count = 0
                model_state.status = ModelStatus.AVAILABLE
            else:
                logger.debug("Free model has too many errors", model=model, error_count=model_state.error_count)
                return False
        
        return True
    
    def reset_model_state(self, model: str):
        """Reset the state of a model (useful when model becomes available again)"""
        if model in self.model_states:
            logger.info("Resetting model state", model=model)
            del self.model_states[model]

    async def _apply_rate_limit(self, model: str):
        """Apply rate limiting based on model type"""
        model_state = self.model_states.get(model, ModelState())
        
        if model_state.last_request:
            time_limit = self.rate_limits['free_models'] if model in self.free_models else self.rate_limits['paid_models']
            time_since_last = datetime.now() - model_state.last_request
            
            if time_since_last < time_limit:
                wait_time = (time_limit - time_since_last).total_seconds()
                logger.debug("Applying rate limit", model=model, wait_time=wait_time)
                await asyncio.sleep(wait_time)
        
        # Update last request time
        if model not in self.model_states:
            self.model_states[model] = ModelState()
        self.model_states[model].last_request = datetime.now()

    async def _handle_response(self, model: str, response: httpx.Response, response_time: float):
        """Handle response and update model state"""
        if response.status_code == 200:
            return
        
        # Handle specific error codes
        if response.status_code == 429:  # Rate limit
            await self._handle_rate_limit(model, response)
        elif response.status_code == 401:  # Unauthorized
            logger.error("OpenRouter authentication failed", model=model)
            self._mark_model_error(model, "Authentication failed")
        elif response.status_code == 403:  # Forbidden
            logger.error("OpenRouter access forbidden", model=model)
            self._mark_model_banned(model, "Access forbidden")
        elif response.status_code >= 500:  # Server errors
            logger.error("OpenRouter server error", model=model, status_code=response.status_code)
            self._mark_model_error(model, f"Server error: {response.status_code}")
        else:
            logger.warning("OpenRouter unexpected status", model=model, status_code=response.status_code)
            self._mark_model_error(model, f"Unexpected status: {response.status_code}")

    async def _handle_rate_limit(self, model: str, response: httpx.Response):
        """Handle rate limit response"""
        try:
            error_data = response.json()
            error_message = error_data.get('error', {}).get('message', 'Rate limit exceeded')
            
            # Extract retry-after if available
            retry_after = response.headers.get('retry-after')
            if retry_after:
                cooldown_seconds = int(retry_after)
            else:
                # Default cooldown based on model type
                cooldown_seconds = 60 if model in self.free_models else 10
            
            logger.warning("Rate limit exceeded", 
                          model=model, 
                          error_message=error_message,
                          cooldown_seconds=cooldown_seconds)
            
            # Set cooldown
            if model not in self.model_states:
                self.model_states[model] = ModelState()
            
            self.model_states[model].status = ModelStatus.RATE_LIMITED
            self.model_states[model].cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)
            self.model_states[model].last_error = error_message
            
        except Exception as e:
            logger.error("Failed to parse rate limit response", model=model, error=str(e))
            self._mark_model_error(model, "Rate limit - failed to parse response")

    async def _handle_error_response(self, model: str, response: httpx.Response) -> str:
        """Handle error response and return error message"""
        try:
            error_data = response.json()
            error_message = error_data.get('error', {}).get('message', f'HTTP {response.status_code}')
            
            logger.error("OpenRouter API error",
                        model=model,
                        status_code=response.status_code,
                        error_message=error_message,
                        response_text=response.text[:200])
            
            return f"OpenRouter error: {error_message}"
            
        except Exception:
            # If we can't parse JSON, use status code
            error_msg = f"OpenRouter HTTP {response.status_code}: {response.text[:100]}"
            logger.error("OpenRouter error (unparseable)", model=model, error=error_msg)
            return error_msg

    def _mark_model_success(self, model: str):
        """Mark model as successful and reset error count"""
        if model not in self.model_states:
            self.model_states[model] = ModelState()
        
        self.model_states[model].status = ModelStatus.AVAILABLE
        self.model_states[model].error_count = 0
        self.model_states[model].last_error = None
        self.model_states[model].cooldown_until = None

    def _mark_model_error(self, model: str, error: str):
        """Mark model as having an error"""
        if model not in self.model_states:
            self.model_states[model] = ModelState()
        
        self.model_states[model].status = ModelStatus.ERROR
        self.model_states[model].error_count += 1
        self.model_states[model].last_error = error
        
        # Implement exponential backoff for repeated errors
        if self.model_states[model].error_count >= 3:
            cooldown_minutes = min(self.model_states[model].error_count * 2, 30)
            self.model_states[model].cooldown_until = datetime.now() + timedelta(minutes=cooldown_minutes)

    def _mark_model_banned(self, model: str, reason: str):
        """Mark model as banned"""
        if model not in self.model_states:
            self.model_states[model] = ModelState()
        
        self.model_states[model].status = ModelStatus.BANNED
        self.model_states[model].last_error = reason
        self.model_states[model].ban_until = datetime.now() + timedelta(hours=1)  # 1 hour ban

    def get_model_status(self, model: str) -> Dict[str, Any]:
        """Get current status of a model"""
        model_state = self.model_states.get(model, ModelState())
        
        return {
            "model": model,
            "status": model_state.status.value,
            "error_count": model_state.error_count,
            "last_error": model_state.last_error,
            "last_request": model_state.last_request.isoformat() if model_state.last_request else None,
            "ban_until": model_state.ban_until.isoformat() if model_state.ban_until else None,
            "cooldown_until": model_state.cooldown_until.isoformat() if model_state.cooldown_until else None,
            "is_available": asyncio.run(self._is_model_available(model))
        }

    def get_all_model_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tracked models"""
        return {model: self.get_model_status(model) for model in self.model_states.keys()}


# Global OpenRouter client instance
openrouter_client = None

def get_openrouter_client(api_key: str, base_url: str = "https://openrouter.ai/api/v1", timeout: int = 30) -> OpenRouterClient:
    """Get or create the global OpenRouter client instance"""
    global openrouter_client
    if openrouter_client is None:
        openrouter_client = OpenRouterClient(api_key, base_url, timeout)
    return openrouter_client