import os
import logging
from typing import List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

logger = logging.getLogger("action_expert")

# Palabras clave de estado para el ActionExpert
ACTION_FINISH = "ACTION_FINISH"
ACTION_WAIT = "ACTION_WAIT"
ACTION_ERROR = "ACTION_ERROR"
ACTION_CALL_USER = "ACTION_CALL_USER"

class ActionExpert:
    def __init__(self, model_id: str = "gemini-2.0-flash"):
        """
        Inicializa el experto en acciones con la API de Gemini.
        """
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY no encontrada en el archivo .env para ActionExpert")
            raise ValueError("GEMINI_API_KEY no encontrada en el archivo .env")
        
        genai.configure(api_key=gemini_api_key)

        self.model = genai.GenerativeModel(model_id)
        self.chat = self.model.start_chat(history=[])
        
        self.system_instruction = f"""
            You are an expert in translating action descriptions into PyAutoGUI commands.
            You will be given a series of steps. Some of these steps are open so you have the freedom to decide what to do and
            some others are more closed and concise which you will have to execute as it is.

            After the steps are given to you, you will have to respond with the exact next step in PyAutoGUI that it has to be executed.
            Then i will pass you a screenshot with set-of-mark of the result. With the screenshot then you will have to give me the next step
            in PyAutoGUI and so on.

            To know the coordinates of the interactive elements of the screen you are provided with a set-of-mark

            Do not include any explanations, additional text, imports, or quotes around the response.
            Only the PyAutoGUI code. 
            
            If the series of steps are done just return 'finish'.

            If there is any step you can't complete for example you could not find what the description says (for example, you can't find the spotify button or you don't find 
            the specific text you are looking for) respond with this format:
            'error: <explanation why you can't perform the action>'

            Examples of input and their expected output:
            - Open the terminal: pyautogui.hotkey('ctrl', 'alt', 't')
            - Type 'hello': pyautogui.typewrite('hello')
            - Click at the spotify icon: pyautogui.click(100, 200)
            - Press the 'enter' key: pyautogui.press('enter')
            - Double click at the test files folder: pyautogui.doubleClick(500, 300)

            Supported PyAutoGUI actions:
            - pyautogui.click(x, y) - Click at coordinates
            - pyautogui.doubleClick(x, y) - Double click
            - pyautogui.rightClick(x, y) - Right click
            - pyautogui.drag(x1, y1, x2, y2) - Drag
            - pyautogui.typewrite('text') - Type text
            - pyautogui.press('key') - Press key
            - pyautogui.hotkey('ctrl', 'c') - Key combination
            - pyautogui.scroll(clicks) - Scroll
            - time.sleep(seconds) - Wait
            
            """


    def add_new_instructions(self, subtask, new_instructions):
        try:
            initial_history = [
                {"role": "user", "parts": [{"text": self.system_instruction}]}, 
                {"role": "user", "parts": [{"text": f"This is the task: {subtask}"}]},
                {"role": "user", "parts": [{"text": f"These are the steps you have to do: {new_instructions}"}]} 
            ]
            self.chat = self.model.start_chat(history = initial_history)
        
        except Exception as e:
            logger.error(f"Error en la función add_new_instructions() del action_expert: {e}")
            raise
        

    def predict(self, SOM):       
        try:
            response = self.chat.send_message([SOM])
            return response.text
        
        except Exception as e:
            logger.error(f"Error en la función predict() del action_expert: {e}")
            raise

if __name__ == "__main__":
    pass