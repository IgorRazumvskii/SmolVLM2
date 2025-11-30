import os
import logging
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import gradio as gr

from model_handler import ModelHandler
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_handler: Optional[ModelHandler] = None


def validate_image(image) -> Tuple[bool, str, Optional[Image.Image]]:
    if image is None:
        return False, "Пожалуйста, загрузите изображение.", None
    
    try:
        if not isinstance(image, Image.Image):
            return False, "Некорректный формат изображения.", None
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return True, "Изображение успешно загружено.", image
    
    except Exception as e:
        return False, f"Не удалось обработать изображение. Ошибка: {str(e)}", None


def vqa_function(image, question: str) -> str:
    if model_handler is None:
        return "Ошибка: Модель не загружена. Пожалуйста, перезапустите приложение."
    
    if not question or not question.strip():
        return "Ошибка: Вопрос не может быть пустым. Пожалуйста, введите вопрос об изображении."
    
    is_valid, message, processed_image = validate_image(image)
    if not is_valid:
        return f"Ошибка: {message}"
    
    try:
        answer = model_handler.vqa(processed_image, question)
        return answer
    except Exception as e:
        logger.error(f"Ошибка при обработке VQA: {e}")
        return f"Ошибка при обработке запроса: {str(e)}"


def caption_function(image) -> str:
    """Image Captioning"""
    if model_handler is None:
        return "Ошибка: Модель не загружена. Пожалуйста, перезапустите приложение."
    
    is_valid, message, processed_image = validate_image(image)
    if not is_valid:
        return f"Ошибка: {message}"
    
    try:
        caption = model_handler.caption(processed_image)
        return caption
    except Exception as e:
        logger.error(f"Ошибка при генерации описания: {e}")
        return f"Ошибка при обработке запроса: {str(e)}"


def ocr_function(image) -> Tuple[str, Optional[str]]:
    if model_handler is None:
        return "Ошибка: Модель не загружена. Пожалуйста, перезапустите приложение.", None
    
    is_valid, message, processed_image = validate_image(image)
    if not is_valid:
        return f"Ошибка: {message}", None
    
    try:
        text = model_handler.ocr(processed_image)
        
        import uuid
        temp_dir = Path("/app/temp_results")
        temp_dir.mkdir(exist_ok=True)
        
        result_id = str(uuid.uuid4())
        result_file = temp_dir / f"ocr_result_{result_id}.txt"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return text, str(result_file)
    
    except Exception as e:
        logger.error(f"Ошибка при OCR обработке: {e}")
        return f"Ошибка при обработке запроса: {str(e)}", None


def create_interface():
    config = Config()
    
    global model_handler
    try:
        model_handler = ModelHandler(config)
        logger.info(f"Модель загружена. Режим: {config.device}, Размер: {config.model_size}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        raise
    
    with gr.Blocks(title="Multimodal Model Demo") as demo:
        
        with gr.Tabs():
            with gr.Tab("Вопросы и Описание"):
                gr.Markdown("### Visual Question Answering и Image Captioning")
                
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            type="pil",
                            label="Загрузите изображение",
                            sources=["upload", "clipboard"]
                        )
                        question_input = gr.Textbox(
                            label="Вопрос об изображении",
                            placeholder="Например: Что находится на этом изображении?",
                            lines=2
                        )
                        vqa_button = gr.Button("Задать вопрос", variant="primary")
                        caption_button = gr.Button("Сгенерировать описание", variant="secondary")
                    
                    with gr.Column():
                        vqa_output = gr.Textbox(
                            label="Ответ",
                            lines=10,
                            interactive=False
                        )
                        caption_output = gr.Textbox(
                            label="Описание изображения",
                            lines=10,
                            interactive=False
                        )
                
                vqa_button.click(
                    fn=vqa_function,
                    inputs=[image_input, question_input],
                    outputs=vqa_output
                )
                
                caption_button.click(
                    fn=caption_function,
                    inputs=image_input,
                    outputs=caption_output
                )
            
            with gr.Tab("OCR"):
                gr.Markdown("### Optical Character Recognition")
                gr.Markdown("Загрузите изображение с текстом для его распознавания.")
                
                with gr.Row():
                    with gr.Column():
                        ocr_image_input = gr.Image(
                            type="pil",
                            label="Загрузите изображение с текстом",
                            sources=["upload", "clipboard"]
                        )
                        ocr_button = gr.Button("Распознать текст", variant="primary")
                    
                    with gr.Column():
                        ocr_output = gr.Textbox(
                            label="Распознанный текст",
                            lines=15,
                            interactive=False
                        )
                        ocr_download = gr.File(
                            label="Скачать результат",
                            visible=True
                        )
                
                ocr_button.click(
                    fn=ocr_function,
                    inputs=ocr_image_input,
                    outputs=[ocr_output, ocr_download]
                )
        
        gr.Markdown(
            f"""
            ---
            **Конфигурация:** Режим: {config.device.upper()}, Размер модели: {config.model_size}
            """
        )
    
    return demo


if __name__ == "__main__":
    config = Config()
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=config.port,
        share=False
    )

