import os
import logging
from typing import List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

logger = logging.getLogger("reflection_expert")

class ReflectionExpert:
    def __init__(self, model_id: str = "gemini-2.0-flash"):
        """
        Inicializa el experto en reflection.
        """
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY no encontrada en el archivo .env para ActionExpert")
            raise ValueError("GEMINI_API_KEY no encontrada en el archivo .env")
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_id)

        self.chat = self.model.start_chat(history=[])
        
        self.system_instruction = "You are a reflection expert"

    
    def save_instruction_list(self, subtask, instruction_list):
        try:
            initial_history = [
                {"role": "user", "parts": [{"text": self.system_instruction}]}, 
                {"role": "user", "parts": [{"text": f"This is the task: {subtask}"}]}, 
                {"role": "user", "parts": [{"text": f"This is what I was told to do: {instruction_list}"}]}
            ]
            self.chat = self.model.start_chat(history = initial_history)
        
        except Exception as e:
            logger.error(f"Error en la función save_instruction_list() del reflection_expert: {e}")
            raise

    def predict_with_error_expert_feedback(self, error_exepert_feedback, SOM):
        try:
            prompt = f"""This is what my error expert friend told me {error_exepert_feedback}. 
            With this give me feedback on how to solve the errors. I also pass you a screenshot of my screen right now"""
            response = self.chat.send_message([prompt, SOM])
            return response.text
        
        except Exception as e:
            logger.error(f"Error en la función predict_with_error_expert_feedback() del reflection_expert: {e}")
            raise
        
    
    def predict(self, execution_error, action_expert_feedback: str, SOM):
        try:
            if execution_error:
                prompt = f"""I had a problem executing the task, this is what i did: {action_expert_feedback} 
                and this is why i could't do it {execution_error}. Give me feedback on how to solve the errors.
                I have a friend who is an expert solving errors.
                If you want me to call him start your response with 'call_error_expert' and then explain what you need help with.
                I also pass you a set-of-mark of my screen"""
                response = self.chat.send_message([prompt, SOM])

                if response.text.startsWith("call_error_expert"):
                    return response.text, None
                return None, response.text
                # error_expert, reflection_expert_feedback

            # esto es cuando ha acabado su lista correctamente
            # hay que evaluar si ha acabado correctamente

            prompt = f"I finished the task, can you tell me if I finished it properly? this is what i did {action_expert_feedback}"
            response = self.chat.send_message(prompt)
            return None, response.text
            # error_expert, reflection_expert_feedback
        
        except Exception as e:
            logger.error(f"Error en la función predict() del reflection_expert: {e}")
            raise
    
        

if __name__ == "__main__":
    pass