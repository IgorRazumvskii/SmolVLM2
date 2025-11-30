import os
import re
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from config import Config


class ModelHandler:
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        if self.config.hf_model_id:
            model_id = self.config.hf_model_id
        else:
            MODEL_SIZES = {
                '256M': 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct',
                '500M': 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
                '2.2B': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
                '1b': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
                '2b': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
            }
            
            model_size_key = self.config.model_size.upper()
            if model_size_key not in MODEL_SIZES:
                for key in MODEL_SIZES.keys():
                    if key in model_size_key or model_size_key in key:
                        model_size_key = key
                        break
                else:
                    model_size_key = '256M'
            
            model_id = MODEL_SIZES.get(model_size_key, MODEL_SIZES['256M'])
        
        local_model_path = self.config.models_dir / self.config.model_name
        
        if local_model_path.exists() and any(local_model_path.iterdir()):
            model_path = str(local_model_path)
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            if "TRANSFORMERS_OFFLINE" in os.environ:
                del os.environ["TRANSFORMERS_OFFLINE"]
            model_path = model_id
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            cache_dir=str(self.config.hf_cache_dir),
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            model_dtype = torch.float32
            attn_implementation = "eager"
        else:
            model_dtype = torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16
            attn_implementation = "flash_attention_2" if self.device == "cuda" else "eager"
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            cache_dir=str(self.config.hf_cache_dir),
            trust_remote_code=True,
            torch_dtype=model_dtype,
            _attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            device_map="auto" if self.device != "cpu" else None
        )
        
        if not hasattr(self.model, 'hf_device_map') or self.model.hf_device_map is None:
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            elif self.device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.model = self.model.to("mps")
            else:
                self.model = self.model.to("cpu")
        
        self.model.eval()
        
        if not local_model_path.exists() or not any(local_model_path.iterdir()):
            local_model_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(local_model_path))
            self.processor.save_pretrained(str(local_model_path))
    
    def vqa(self, image: Image.Image, question: str) -> str:
        """Visual Question Answering"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": 512,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True
            }
            if hasattr(self.processor, 'tokenizer'):
                if self.processor.tokenizer.pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = self.processor.tokenizer.pad_token_id
                if hasattr(self.processor.tokenizer, 'eos_token_id') and self.processor.tokenizer.eos_token_id is not None:
                    generate_kwargs["eos_token_id"] = self.processor.tokenizer.eos_token_id
            
            outputs = self.model.generate(**inputs, **generate_kwargs)
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        match = re.search('Assistant: ', response)
        if match:
            return response[match.end():].strip()
        return response.strip()
    
    def caption(self, image: Image.Image) -> str:
        """Image Captioning"""
        question = "Describe this image."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": 80,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True,
                "early_stopping": True
            }
            if hasattr(self.processor, 'tokenizer'):
                if self.processor.tokenizer.pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = self.processor.tokenizer.pad_token_id
                if hasattr(self.processor.tokenizer, 'eos_token_id') and self.processor.tokenizer.eos_token_id is not None:
                    generate_kwargs["eos_token_id"] = self.processor.tokenizer.eos_token_id
            
            outputs = self.model.generate(**inputs, **generate_kwargs)
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        match = re.search('Assistant: ', response)
        if match:
            return response[match.end():].strip()
        return response.strip()
    
    def ocr(self, image: Image.Image) -> str:
        """Optical Character Recognition"""
        question = 'Perform OCR (optical character recognition). Extract text from this image'
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                min_new_tokens=1,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer.pad_token_id is not None else None,
                eos_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None,
                use_cache=True
            )
        
        extracted_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        match = re.search('Assistant: ', extracted_text)
        if match:
            return extracted_text[match.end():].strip()
        return extracted_text.strip()

