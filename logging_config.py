"""
Configurazione logging strutturato per Round TAIble Backend
"""
import logging
import logging.handlers
import json
import sys
from datetime import datetime
from typing import Dict, Any
import os


class JSONFormatter(logging.Formatter):
    """Formatter per output JSON strutturato"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Aggiungi info di contesto se presenti
        if hasattr(record, 'debate_id'):
            log_entry['debate_id'] = record.debate_id
        if hasattr(record, 'model_id'):
            log_entry['model_id'] = record.model_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_action'):
            log_entry['user_action'] = record.user_action
        if hasattr(record, 'response_time'):
            log_entry['response_time'] = record.response_time
        if hasattr(record, 'queue_size'):
            log_entry['queue_size'] = record.queue_size
        if hasattr(record, 'websocket_connections'):
            log_entry['websocket_connections'] = record.websocket_connections
        
        # Aggiungi exception info se presente
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Aggiungi stack trace se presente
        if record.stack_info:
            log_entry['stack_info'] = record.stack_info
            
        return json.dumps(log_entry, ensure_ascii=False)


class ContextLogger:
    """Logger con supporto per contesto specifico del dibattito"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
    def _add_context(self, extra: Dict[str, Any], **context):
        """Aggiunge contesto al log"""
        if extra is None:
            extra = {}
        extra.update(context)
        return extra
    
    def debug(self, msg: str, extra: Dict[str, Any] = None, **context):
        self.logger.debug(msg, extra=self._add_context(extra, **context))
        
    def info(self, msg: str, extra: Dict[str, Any] = None, **context):
        self.logger.info(msg, extra=self._add_context(extra, **context))
        
    def warning(self, msg: str, extra: Dict[str, Any] = None, **context):
        self.logger.warning(msg, extra=self._add_context(extra, **context))
        
    def error(self, msg: str, extra: Dict[str, Any] = None, **context):
        self.logger.error(msg, extra=self._add_context(extra, **context))
        
    def critical(self, msg: str, extra: Dict[str, Any] = None, **context):
        self.logger.critical(msg, extra=self._add_context(extra, **context))


def setup_logging(debug: bool = False, log_file: str = None) -> None:
    """
    Configura il sistema di logging
    
    Args:
        debug: Se True, imposta livello DEBUG
        log_file: Path del file di log (opzionale)
    """
    # Livello di logging
    level = logging.DEBUG if debug else logging.INFO
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Rimuovi handlers esistenti
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler con formato umano per debug
    if debug:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler con formato JSON per produzione
    if log_file:
        # Crea directory se non esiste
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Rotating file handler (max 10MB, 5 file di backup)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024, 
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    
    # Handler JSON per stdout in produzione (se non debug)
    if not debug:
        json_handler = logging.StreamHandler(sys.stdout)
        json_handler.setLevel(level)
        json_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(json_handler)
    
    # Configura logger specifici
    
    # Riduci verbosità di librerie esterne
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('watchfiles').setLevel(logging.WARNING)
    
    # Logger specifici per l'applicazione
    app_loggers = [
        'main',
        'websocket_manager', 
        'llm_queue_manager',
        'debate_manager',
        'openrouter_client'
    ]
    
    for logger_name in app_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)


def get_context_logger(name: str) -> ContextLogger:
    """
    Ottieni un logger con supporto per contesto
    
    Args:
        name: Nome del logger
        
    Returns:
        ContextLogger configurato
    """
    return ContextLogger(name)


# Metriche di performance
class PerformanceMetrics:
    """Classe per tracciare metriche di performance"""
    
    def __init__(self):
        self.logger = get_context_logger('performance')
        
    def log_api_call(self, endpoint: str, method: str, response_time: float, 
                    status_code: int, **context):
        """Log chiamata API"""
        self.logger.info(
            f"API {method} {endpoint} - {status_code} in {response_time:.3f}s",
            endpoint=endpoint,
            method=method,
            response_time=response_time,
            status_code=status_code,
            **context
        )
    
    def log_llm_request(self, model: str, response_time: float, 
                       tokens_used: int = None, **context):
        """Log richiesta LLM"""
        self.logger.info(
            f"LLM {model} response in {response_time:.3f}s",
            model_id=model,
            response_time=response_time,
            tokens_used=tokens_used,
            **context
        )
    
    def log_queue_metrics(self, queue_sizes: Dict[str, int], 
                         active_requests: int, avg_wait_time: float):
        """Log metriche delle code"""
        self.logger.info(
            f"Queue metrics: {active_requests} active, avg wait {avg_wait_time:.2f}s",
            queue_sizes=queue_sizes,
            active_requests=active_requests,
            avg_wait_time=avg_wait_time
        )
    
    def log_websocket_metrics(self, total_connections: int, 
                             active_debates: int, **context):
        """Log metriche WebSocket"""
        self.logger.info(
            f"WebSocket: {total_connections} connections, {active_debates} debates",
            websocket_connections=total_connections,
            active_debates=active_debates,
            **context
        )


# Istanza globale per metriche
performance_metrics = PerformanceMetrics()