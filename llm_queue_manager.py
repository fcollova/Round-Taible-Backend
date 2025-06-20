import asyncio
from asyncio import Queue, Lock
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
from enum import Enum
import time
import requests
import httpx
import os
import configparser
from logging_config import get_context_logger, performance_metrics

logger = get_context_logger(__name__)

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class MessageRequest:
    debate_id: str
    model_id: str
    context: str
    topic: str
    personality: str
    message_count: int
    priority: MessagePriority = MessagePriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    max_retries: int = 3
    retry_count: int = 0
    callback: Optional[Callable] = None

class LLMQueueManager:
    def __init__(self, max_concurrent_requests: int = 3, frontend_url: str = "http://localhost:3000", frontend_timeout: int = 10):
        # Code per priorità
        self.priority_queues: Dict[MessagePriority, Queue] = {
            priority: Queue() for priority in MessagePriority
        }
        
        # Semaforo per limitare richieste concorrenti
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Tracking delle richieste attive
        self.active_requests: Dict[str, MessageRequest] = {}
        self.request_lock = Lock()
        
        # Rate limiting per modello
        self.model_last_request: Dict[str, datetime] = {}
        self.model_rate_limits: Dict[str, timedelta] = {
            'gpt4': timedelta(seconds=1),
            'claude': timedelta(seconds=0.5),
            'gemini': timedelta(seconds=0.8),
            'llama': timedelta(seconds=0.3)
        }
        
        # Statistiche
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'queue_wait_times': []
        }
        
        # Frontend configuration
        self.frontend_url = frontend_url.rstrip('/')  # Remove trailing slash
        self.frontend_timeout = frontend_timeout
        
        # Workers
        self.workers = []
        self.running = False
    
    async def start(self, num_workers: int = 2):
        """Avvia i worker per processare le code"""
        self.running = True
        
        logger.info("Starting LLM Queue Manager", 
                   worker_count=num_workers,
                   max_concurrent=self.semaphore._value)
        
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Worker per cleanup richieste scadute
        cleanup_worker = asyncio.create_task(self._cleanup_expired_requests())
        self.workers.append(cleanup_worker)
        
        logger.info("LLM Queue Manager started successfully", 
                   total_workers=len(self.workers),
                   active_workers=num_workers,
                   cleanup_worker=True)
    
    async def stop(self):
        """Ferma tutti i workers"""
        logger.info("Stopping LLM Queue Manager", 
                   active_workers=len(self.workers),
                   pending_requests=len(self.active_requests))
        
        self.running = False
        cancelled_count = 0
        
        for worker in self.workers:
            worker.cancel()
            cancelled_count += 1
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("LLM Queue Manager stopped", 
                   cancelled_workers=cancelled_count,
                   final_stats=self.stats)
    
    async def enqueue_message(self, request: MessageRequest) -> str:
        """Aggiungi richiesta alla coda appropriata"""
        request_id = f"{request.debate_id}_{request.model_id}_{int(time.time() * 1000)}"
        
        logger.info("Enqueuing LLM message request",
                   request_id=request_id,
                   debate_id=request.debate_id,
                   model_id=request.model_id,
                   priority=request.priority.name,
                   message_count=request.message_count)
        
        async with self.request_lock:
            self.active_requests[request_id] = request
        
        await self.priority_queues[request.priority].put((request_id, request))
        
        # Log stato delle code
        queue_stats = self.get_queue_stats()
        performance_metrics.log_queue_metrics(
            queue_sizes=queue_stats['queue_sizes'],
            active_requests=queue_stats['active_requests'],
            avg_wait_time=queue_stats['stats']['average_queue_wait_time']
        )
        
        logger.debug("Request successfully enqueued",
                    request_id=request_id,
                    queue_size=queue_stats['queue_sizes'][request.priority.name],
                    total_active=queue_stats['active_requests'])
        
        return request_id
    
    async def _worker(self, worker_name: str):
        """Worker che processa le richieste dalle code"""
        while self.running:
            try:
                # Controlla code in ordine di priorità
                for priority in reversed(list(MessagePriority)):
                    queue = self.priority_queues[priority]
                    
                    if not queue.empty():
                        try:
                            # Timeout per evitare blocking indefinito
                            request_id, request = await asyncio.wait_for(
                                queue.get(), timeout=0.1
                            )
                            
                            logger.debug("Worker picked up request",
                                       worker_name=worker_name,
                                       request_id=request_id,
                                       priority=priority.name,
                                       debate_id=request.debate_id)
                            
                            # Processa la richiesta
                            await self._process_request(worker_name, request_id, request)
                            break
                            
                        except asyncio.TimeoutError:
                            continue
                
                # Pausa breve se nessuna richiesta da processare
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("Critical error in worker",
                           worker_name=worker_name,
                           error=str(e),
                           error_type=type(e).__name__)
                await asyncio.sleep(1)
    
    async def _process_request(self, worker_name: str, request_id: str, request: MessageRequest):
        """Processa una singola richiesta"""
        async with self.semaphore:  # Limita concorrenza
            start_time = datetime.now()
            queue_wait_time = (start_time - request.created_at).total_seconds()
            
            try:
                # Rate limiting per modello
                await self._apply_rate_limit(request.model_id)
                
                logger.info("Worker processing LLM request",
                           worker_name=worker_name,
                           request_id=request_id,
                           debate_id=request.debate_id,
                           model_id=request.model_id,
                           queue_wait_time=queue_wait_time,
                           retry_count=request.retry_count)
                
                # Genera risposta LLM
                response = await self._generate_llm_response(request)
                
                if response:  # Se la risposta è valida
                    # Calcola tempo di risposta
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    # Invia risposta via WebSocket
                    message_data = {
                        'modelId': request.model_id,
                        'content': response['content'],
                        'timestamp': datetime.now().isoformat(),
                        'type': 'ai',
                        'id': f"msg_{int(time.time() * 1000)}",
                        'turnNumber': request.message_count + 1
                    }
                    
                    # Salva il messaggio nel database
                    await self._save_message_to_database(request.debate_id, message_data)
                    
                    # Invia il messaggio via WebSocket
                    await self._send_response_to_debate(request.debate_id, {
                        'type': 'new_message',
                        'data': message_data
                    })
                    
                    # Callback se presente
                    if request.callback:
                        await request.callback(request_id, response, None)
                    
                    # Aggiorna statistiche
                    self.stats['successful_requests'] += 1
                    self._update_average_response_time(response_time)
                    self.stats['queue_wait_times'].append(queue_wait_time)
                    
                    # Mantieni solo ultime 100 metriche per memoria
                    if len(self.stats['queue_wait_times']) > 100:
                        self.stats['queue_wait_times'] = self.stats['queue_wait_times'][-100:]
                    
                    logger.info("Worker completed LLM request successfully",
                               worker_name=worker_name,
                               request_id=request_id,
                               debate_id=request.debate_id,
                               model_id=request.model_id,
                               response_time=response_time,
                               content_length=len(response.get('content', '')),
                               queue_wait_time=queue_wait_time)
                
            except Exception as e:
                await self._handle_request_error(request_id, request, e)
                
            finally:
                # Rimuovi richiesta dalle attive
                async with self.request_lock:
                    self.active_requests.pop(request_id, None)
                
                self.stats['total_requests'] += 1
    
    async def _apply_rate_limit(self, model_id: str):
        """Applica rate limiting per modello"""
        if model_id in self.model_last_request:
            time_limit = self.model_rate_limits.get(model_id, timedelta(seconds=1))
            time_since_last = datetime.now() - self.model_last_request[model_id]
            
            if time_since_last < time_limit:
                wait_time = (time_limit - time_since_last).total_seconds()
                await asyncio.sleep(wait_time)
        
        self.model_last_request[model_id] = datetime.now()
    
    async def _generate_llm_response(self, request: MessageRequest) -> dict:
        """Genera risposta dall'LLM"""
        try:
            # Import dinamico per evitare circular imports
            from main import call_openrouter, MODELS
            
            # Create system prompt based on personality and debate stage
            if request.message_count == 0:
                # First message - opening statement
                system_prompt = f"""You are starting a live debate about: {request.topic}

                Your personality and expertise: {request.personality}
                
                As the first speaker, provide an opening statement that:
                1. Clearly states your position on the topic
                2. Presents 2-3 key arguments
                3. Sets the tone for an engaging debate
                
                Keep it concise (2-3 paragraphs max) and compelling. This is your chance to make a strong first impression."""
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Start the debate on: {request.topic}"}
                ]
            else:
                # Continuing debate - response to previous messages
                system_prompt = f"""You are participating in an ongoing live debate about: {request.topic}

                Your personality and debate style: {request.personality}
                
                Previous discussion context:
                {request.context}
                
                Generate a thoughtful response that:
                1. Directly addresses specific points made by other participants
                2. Builds upon or challenges their arguments with evidence
                3. Maintains your unique perspective while engaging constructively
                4. Advances the debate with new insights or counterpoints
                5. If there is moderator guidance in the context, you MUST acknowledge and address it in your response
                
                Guidelines:
                - Keep it concise (1-2 paragraphs max) and engaging
                - Reference specific points from the previous discussion
                - Stay true to your personality and expertise
                - Be respectful but intellectually rigorous
                - Avoid repeating what others have already said
                - IMPORTANT: If the moderator has provided guidance, incorporate their direction into your response while maintaining your unique perspective"""
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Based on the discussion above, provide your next contribution to the debate on: {request.topic}"}
                ]
            
            response = call_openrouter(
                model=MODELS[request.model_id],
                messages=messages,
                max_tokens=350 if request.message_count == 0 else 300,
                temperature=0.8
            )
            
            content = response['choices'][0]['message']['content']
            return {
                "model": request.model_id,
                "content": content,
                "topic": request.topic,
                "message_count": request.message_count
            }
            
        except Exception as e:
            # Incrementa retry count
            request.retry_count += 1
            
            if request.retry_count < request.max_retries:
                # Re-enqueue con priorità più alta se possibile
                new_priority = MessagePriority.HIGH if request.priority == MessagePriority.NORMAL else request.priority
                request.priority = new_priority
                
                await asyncio.sleep(2 ** request.retry_count)  # Exponential backoff
                await self.enqueue_message(request)
                
                logger.warning("Retrying LLM request",
                              request_id=request_id,
                              debate_id=request.debate_id,
                              model_id=request.model_id,
                              retry_count=request.retry_count,
                              max_retries=request.max_retries,
                              new_priority=new_priority.name,
                              backoff_delay=2 ** request.retry_count)
                return None
            else:
                raise e
    
    async def _handle_request_error(self, request_id: str, request: MessageRequest, error: Exception):
        """Gestisce errori nelle richieste"""
        logger.error("Error processing LLM request",
                    request_id=request_id,
                    debate_id=request.debate_id,
                    model_id=request.model_id,
                    error=str(error),
                    error_type=type(error).__name__,
                    retry_count=request.retry_count,
                    max_retries=request.max_retries)
        
        # Invia messaggio di fallback via WebSocket
        fallback_messages = [
            "Mi dispiace, sto avendo difficoltà tecniche. Riproverò tra poco.",
            "Sto elaborando la risposta, un momento di pazienza...",
            "Sistema temporaneamente sovraccarico, riprovo immediatamente."
        ]
        
        import random
        fallback_content = random.choice(fallback_messages)
        
        await self._send_response_to_debate(request.debate_id, {
            'type': 'new_message',
            'data': {
                'id': f"{request_id}_fallback",
                'ai': request.model_id,
                'content': fallback_content,
                'timestamp': datetime.now().isoformat(),
                'is_fallback': True,
                'error_type': type(error).__name__
            }
        })
        
        # Callback con errore
        if request.callback:
            await request.callback(request_id, None, error)
        
        self.stats['failed_requests'] += 1
    
    async def _save_message_to_database(self, debate_id: str, message_data: dict):
        """Salva messaggio nel database tramite API Next.js"""
        try:
            # Costruisci URL API Next.js
            api_url = f"{self.frontend_url}/api/debates/{debate_id}/messages"
            
            # Prepara payload per API
            payload = {
                'modelId': message_data.get('modelId'),
                'content': message_data.get('content'),
                'timestamp': message_data.get('timestamp'),
                'type': message_data.get('type', 'ai')
            }
            
            logger.info("Saving message to database",
                       debate_id=debate_id,
                       model_id=payload['modelId'],
                       content_length=len(payload['content']) if payload['content'] else 0)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(api_url, json=payload, timeout=self.frontend_timeout)
                
                if response.status_code == 200:
                    saved_message = response.json()
                    logger.info("Message saved successfully",
                               debate_id=debate_id,
                               message_id=saved_message.get('id'),
                               turn_number=saved_message.get('turnNumber'))
                else:
                    error_text = response.text
                    logger.error("Failed to save message to database",
                                debate_id=debate_id,
                                status_code=response.status_code,
                                error_response=error_text[:200])
                                
        except Exception as e:
            logger.error("Exception saving message to database",
                        debate_id=debate_id,
                        error=str(e),
                        error_type=type(e).__name__)
    
    async def _send_response_to_debate(self, debate_id: str, message: dict):
        """Invia risposta al dibattito via WebSocket"""
        try:
            from websocket_manager import debate_manager
            await debate_manager.broadcast_to_debate(debate_id, message)
        except Exception as e:
            logger.error("Failed to send WebSocket response",
                    debate_id=debate_id,
                    error=str(e),
                    error_type=type(e).__name__)
    
    async def _cleanup_expired_requests(self):
        """Rimuove richieste scadute"""
        while self.running:
            try:
                current_time = datetime.now()
                expired_timeout = timedelta(minutes=5)
                
                async with self.request_lock:
                    expired_requests = [
                        req_id for req_id, req in self.active_requests.items()
                        if current_time - req.created_at > expired_timeout
                    ]
                    
                    for req_id in expired_requests:
                        expired_req = self.active_requests[req_id]
                        del self.active_requests[req_id]
                        logger.warning("Expired request removed",
                                     request_id=req_id,
                                     debate_id=expired_req.debate_id,
                                     model_id=expired_req.model_id,
                                     age_minutes=(current_time - expired_req.created_at).total_seconds() / 60)
                
                await asyncio.sleep(60)  # Cleanup ogni minuto
                
            except Exception as e:
                logger.error("Error in cleanup worker",
                           error=str(e),
                           error_type=type(e).__name__)
                await asyncio.sleep(60)
    
    def _update_average_response_time(self, response_time: float):
        """Aggiorna tempo di risposta medio"""
        current_avg = self.stats['average_response_time']
        total_successful = self.stats['successful_requests']
        
        if total_successful == 1:
            self.stats['average_response_time'] = response_time
        else:
            # Media pesata
            self.stats['average_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
    
    def get_queue_stats(self) -> dict:
        """Restituisce statistiche delle code"""
        queue_sizes = {
            priority.name: queue.qsize() 
            for priority, queue in self.priority_queues.items()
        }
        
        avg_queue_wait = (
            sum(self.stats['queue_wait_times']) / len(self.stats['queue_wait_times'])
            if self.stats['queue_wait_times'] else 0
        )
        
        return {
            'queue_sizes': queue_sizes,
            'active_requests': len(self.active_requests),
            'stats': {
                **self.stats,
                'average_queue_wait_time': avg_queue_wait
            },
            'model_rates': {
                model: last_req.isoformat() if last_req else None
                for model, last_req in self.model_last_request.items()
            }
        }

# Load configuration for frontend URL
config = configparser.ConfigParser()
config_paths = ['config.conf', './config.conf', '../config.conf']
config_loaded = False

for path in config_paths:
    try:
        if config.read(path):
            config_loaded = True
            logger.info("Configuration loaded for LLM Queue", config_path=path)
            break
    except Exception as e:
        logger.debug("Failed to read config file for LLM Queue", config_path=path, error=str(e))

# Get frontend configuration
frontend_url = os.getenv('FRONTEND_URL') or (config.get('frontend', 'url', fallback='http://localhost:3000') if config_loaded else 'http://localhost:3000')
frontend_timeout = int(os.getenv('FRONTEND_TIMEOUT', '10')) or (int(config.get('frontend', 'timeout', fallback='10')) if config_loaded else 10)

logger.info("Frontend configuration loaded", 
           url=frontend_url, 
           timeout=frontend_timeout,
           source="config.conf" if config_loaded else "defaults")

# Istanza globale
llm_queue = LLMQueueManager(
    max_concurrent_requests=4,
    frontend_url=frontend_url,
    frontend_timeout=frontend_timeout
)