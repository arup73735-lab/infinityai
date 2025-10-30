"""
Async batch worker for efficient inference.

Design rationale:
- Batching: Groups multiple requests to maximize GPU utilization
- Async processing: Non-blocking I/O for high concurrency
- Dynamic batching: Balances latency vs throughput with configurable timeout
- Queue management: Fair scheduling with priority support

Performance tradeoffs:
- Batch size: Larger batches = higher throughput but higher latency
- Timeout: Shorter timeout = lower latency but lower throughput
- Memory: Larger batches require more GPU memory
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from queue import Queue
import torch

from model_loader import model_loader
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Represents a single inference request."""
    request_id: str
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = False
    future: asyncio.Future = field(default_factory=asyncio.Future)
    created_at: float = field(default_factory=time.time)


@dataclass
class InferenceResponse:
    """Represents an inference response."""
    request_id: str
    text: str
    tokens: int
    latency: float
    error: Optional[str] = None


class BatchWorker:
    """Manages batched inference requests."""
    
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self._worker_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the batch worker."""
        if self.running:
            logger.warning("Worker already running")
            return
            
        self.running = True
        self._worker_task = asyncio.create_task(self._process_loop())
        logger.info("Batch worker started")
        
    async def stop(self):
        """Stop the batch worker."""
        self.running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Batch worker stopped")
        
    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        """Submit a request for processing."""
        await self.queue.put(request)
        return await request.future
        
    async def _process_loop(self):
        """Main processing loop."""
        while self.running:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)
                
    async def _collect_batch(self) -> List[InferenceRequest]:
        """Collect requests into a batch."""
        batch = []
        deadline = time.time() + settings.batch_timeout
        
        try:
            # Wait for first request
            request = await asyncio.wait_for(
                self.queue.get(),
                timeout=settings.batch_timeout
            )
            batch.append(request)
            
            # Collect more requests until batch is full or timeout
            while len(batch) < settings.batch_size and time.time() < deadline:
                try:
                    request = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=deadline - time.time()
                    )
                    batch.append(request)
                except asyncio.TimeoutError:
                    break
                    
        except asyncio.TimeoutError:
            pass
            
        return batch
        
    async def _process_batch(self, batch: List[InferenceRequest]):
        """Process a batch of requests."""
        start_time = time.time()
        
        try:
            # Prepare inputs
            prompts = [req.prompt for req in batch]
            
            # Tokenize
            inputs = model_loader.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=settings.max_length
            ).to(model_loader.device)
            
            # Generate
            with torch.no_grad():
                outputs = model_loader.model.generate(
                    **inputs,
                    max_new_tokens=batch[0].max_new_tokens,
                    temperature=batch[0].temperature,
                    top_p=batch[0].top_p,
                    top_k=batch[0].top_k,
                    do_sample=True,
                    pad_token_id=model_loader.tokenizer.pad_token_id,
                    eos_token_id=model_loader.tokenizer.eos_token_id,
                )
            
            # Decode outputs
            generated_texts = model_loader.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            # Create responses
            latency = time.time() - start_time
            
            for req, text, output in zip(batch, generated_texts, outputs):
                # Remove prompt from output
                prompt_length = len(req.prompt)
                generated_text = text[prompt_length:].strip()
                
                response = InferenceResponse(
                    request_id=req.request_id,
                    text=generated_text,
                    tokens=len(output),
                    latency=latency
                )
                
                req.future.set_result(response)
                
            logger.info(
                f"Processed batch of {len(batch)} requests in {latency:.3f}s "
                f"({len(batch)/latency:.1f} req/s)"
            )
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            
            # Set error for all requests in batch
            for req in batch:
                if not req.future.done():
                    response = InferenceResponse(
                        request_id=req.request_id,
                        text="",
                        tokens=0,
                        latency=time.time() - start_time,
                        error=str(e)
                    )
                    req.future.set_result(response)


class StreamingWorker:
    """Handles streaming inference requests."""
    
    @staticmethod
    async def generate_stream(
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> AsyncIterator[str]:
        """Generate tokens in streaming mode."""
        try:
            # Tokenize input
            inputs = model_loader.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=settings.max_length
            ).to(model_loader.device)
            
            input_length = inputs.input_ids.shape[1]
            
            # Generate tokens one at a time
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = model_loader.model(
                        **inputs,
                        use_cache=True
                    )
                    
                    # Get next token logits
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Apply temperature
                    next_token_logits = next_token_logits / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(
                            next_token_logits, top_k
                        )[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(
                            next_token_logits, descending=True
                        )
                        cumulative_probs = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Check for EOS
                    if next_token.item() == model_loader.tokenizer.eos_token_id:
                        break
                    
                    # Decode and yield token
                    token_text = model_loader.tokenizer.decode(
                        next_token[0],
                        skip_special_tokens=True
                    )
                    yield token_text
                    
                    # Append to inputs for next iteration
                    inputs.input_ids = torch.cat([inputs.input_ids, next_token], dim=-1)
                    
                    # Yield control to event loop
                    await asyncio.sleep(0)
                    
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}", exc_info=True)
            yield f"[Error: {str(e)}]"


# Global worker instance
batch_worker = BatchWorker()
streaming_worker = StreamingWorker()
