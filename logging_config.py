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


class DetailedFormatter(logging.Formatter):
    """Formatter umano che include i parametri extra per debug dettagliato"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Base message format
        base_msg = super().format(record)
        
        # Collect extra fields that are not standard log record attributes
        standard_attrs = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'processName', 'process', 'getMessage',
            'exc_info', 'exc_text', 'stack_info', 'message'
        }
        
        extra_fields = []
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith('_') and value is not None:
                # Format complex objects nicely
                if isinstance(value, (list, dict)):
                    if key == 'messages' and isinstance(value, list):
                        # Special formatting for conversation messages
                        msg_preview = []
                        for msg in value[:2]:  # Show first 2 messages
                            if isinstance(msg, dict):
                                role = msg.get('role', 'unknown')
                                content = msg.get('content', '')
                                preview = content[:100] + '...' if len(content) > 100 else content
                                msg_preview.append(f"{role}: {preview}")
                        extra_fields.append(f"{key}=[{'; '.join(msg_preview)}]")
                    else:
                        extra_fields.append(f"{key}={str(value)[:200]}")
                else:
                    extra_fields.append(f"{key}={value}")
        
        if extra_fields:
            return f"{base_msg} | {' | '.join(extra_fields)}"
        else:
            return base_msg


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


def setup_logging(debug: bool = False, level: str = "INFO", 
                 console_output: bool = True, file_output: bool = True, 
                 log_file: str = None, module_levels: Dict[str, str] = None) -> None:
    """
    Configura il sistema di logging
    
    Args:
        debug: Se True, attiva modalità debug (sovrascrive level)
        level: Livello di logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Se True, attiva output su console
        file_output: Se True, attiva output su file
        log_file: Path del file di log (opzionale)
        module_levels: Dict con livelli specifici per modulo
    """
    # Determina il livello di logging
    if debug:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Rimuovi handlers esistenti
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        if debug:
            # Formato umano dettagliato per debug che include parametri extra
            console_formatter = DetailedFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            # Formato umano anche in produzione per console Render leggibile
            console_formatter = DetailedFormatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if file_output and log_file:
        # Crea directory se non esiste
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Rotating file handler (max 10MB, 5 file di backup)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024, 
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    
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
    
    # Applica livelli di default o specifici per modulo
    if module_levels is None:
        module_levels = {}
        
    for logger_name in app_loggers:
        logger = logging.getLogger(logger_name)
        # Usa livello specifico del modulo se presente, altrimenti livello generale
        module_level = module_levels.get(logger_name, level)
        logger_level = getattr(logging, module_level.upper(), log_level)
        logger.setLevel(logger_level)


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
    
    def log_openrouter_call(self, model: str, response_time: float, 
                           status_code: int = None, **context):
        """Log chiamata OpenRouter"""
        self.logger.info(
            f"OpenRouter {model} - {status_code or 'N/A'} in {response_time:.3f}s",
            model_id=model,
            response_time=response_time,
            status_code=status_code,
            **context
        )


# Istanza globale per metriche
performance_metrics = PerformanceMetrics()