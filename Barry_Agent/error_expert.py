import os
import logging
from typing import List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

logger = logging.getLogger("error_expert")

class ErrorExpert:
    def __init__(self, model_id: str = "gemini-2.0-flash"):
        """
        Inicializa el experto en errores.
        """
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY no encontrada en el archivo .env para ActionExpert")
            raise ValueError("GEMINI_API_KEY no encontrada en el archivo .env")
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_id)
        

    
    
    def predict(self, info_for_error_expert: str, SOM):
        try:
            prompt = f"You are an error expert. Help me solve this problem: {info_for_error_expert}"
            response = self.model.generate_content([prompt, SOM])
            return response.text
        
        except Exception as e:
            logger.error(f"Error en la función predict() del error_expert: {e}")
            raise
    
        

if __name__ == "__main__":
    pass