import os
import logging
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import re
import time

logger = logging.getLogger("desktopenv.agent")

# Palabras clave para estados del agente
FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

class BarryAgent:
    def __init__(self, vision_model: str = "gemini-2.0-flash", observation_type: str = "screenshot", action_space: str = "pyautogui"):
        """
        Inicializa el agente con la configuración para interactuar con OSWorld y la API de Gemini.
        """
        # Cargar variables de entorno desde .env
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY no encontrada en el archivo .env")
            raise ValueError("GEMINI_API_KEY no encontrada en el archivo .env")

        # Configurar la API de Gemini
        genai.configure(api_key=gemini_api_key)
        self.vision_model = genai.GenerativeModel(vision_model)

        # Configurar parámetros del entorno
        self.observation_type = observation_type
        self.action_space = action_space
        
        # Historial para el agente
        self.trajectory_length = 0
        self.max_trajectory_length = 50
        self.call_user_count = 0
        self.call_user_tolerance = 3

    def process_observation(self, obs: Dict) -> Image.Image:
        """
        Procesa la captura de pantalla del entorno para convertirla a una imagen PIL compatible con Gemini.
        """
        if self.observation_type not in ["screenshot", "screenshot_a11y_tree"]:
            logger.error(f"observation_type debe ser 'screenshot' o 'screenshot_a11y_tree', recibido: {self.observation_type}")
            raise ValueError(f"observation_type no soportado: {self.observation_type}")

        if "screenshot" not in obs:
            logger.error("No se encontró 'screenshot' en la observación")
            raise ValueError("No se encontró 'screenshot' en la observación")

        try:
            screenshot = obs["screenshot"]
            # Manejar si screenshot es str
            if isinstance(screenshot, str):
                screenshot = screenshot.encode('utf-8')
            # Convertir a imagen PIL
            image = Image.open(BytesIO(screenshot))
            logger.info("Captura de pantalla procesada correctamente")
            return image
        except Exception as e:
            logger.error(f"Error al procesar el screenshot: {e}")
            raise

    def parse_gemini_response(self, response_text: str) -> Tuple[str, List[str]]:
        """
        Parsea la respuesta de Gemini para extraer el estado y las acciones.
        """
        response_text = response_text.strip()
        
        # Verificar si es una respuesta de estado especial
        if "finished" in response_text.lower() or "done" in response_text.lower():
            return "Task completed", ["DONE"]
        
        if "wait" in response_text.lower():
            return "Waiting for system response", ["WAIT"]
        
        if "error" in response_text.lower() or "fail" in response_text.lower():
            return "Task failed", ["FAIL"]
        
        if "call_user" in response_text.lower():
            if self.call_user_count < self.call_user_tolerance:
                self.call_user_count += 1
                return "Calling user for help", ["WAIT"]
            else:
                return "Too many user calls, failing", ["FAIL"]
        
        # Extraer acciones PyAutoGUI
        actions = []
        if response_text.startswith('"') and response_text.endswith('"'):
            response_text = response_text[1:-1]
        
        # Dividir por punto y coma para obtener acciones individuales
        action_parts = [action.strip() for action in response_text.split(';') if action.strip()]
        
        if not action_parts:
            return "No valid actions found", ["FAIL"]
        
        actions = action_parts
        logger.info(f"Acciones parseadas: {actions}")
        return response_text, actions

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        """
        Envía la captura de pantalla y la instrucción a Gemini para generar acciones pyautogui.
        Retorna una tupla con la respuesta de Gemini y la lista de acciones con estados apropiados.
        """
        # Incrementar contador de trayectoria
        self.trajectory_length += 1
        
        # Verificar si excedemos la longitud máxima de trayectoria
        if self.trajectory_length > self.max_trajectory_length:
            logger.warning(f"Trayectoria excede el límite máximo de {self.max_trajectory_length} pasos")
            return "Maximum trajectory length exceeded", ["FAIL"]
        
        try:
            # Procesar la observación para obtener la imagen
            image = self.process_observation(obs)

            # Construir el prompt mejorado
            prompt = f"""
            Instrucción: {instruction}
            
            Basado en la captura de pantalla proporcionada, analiza la situación y determina qué hacer.
            
            Tienes las siguientes opciones:
            1. Si la tarea está completada, responde con "finished"
            2. Si necesitas esperar a que el sistema responda, responde con "wait"
            3. Si hay un error y no puedes continuar, responde con "error"
            4. Si necesitas ayuda del usuario, responde con "call_user"
            5. Si puedes continuar con la tarea, proporciona acciones PyAutoGUI
            
            Para las acciones PyAutoGUI, proporciona solo las acciones en una sola línea, separadas por punto y coma.
            Incluye time.sleep() apropiados entre acciones para dar tiempo al sistema.
            
            Ejemplos de acciones:
            - Abrir terminal: pyautogui.hotkey('ctrl', 'alt', 't');time.sleep(2)
            - Escribir comando: pyautogui.typewrite('ls');pyautogui.press('enter');time.sleep(1)
            - Hacer clic: pyautogui.click(x, y);time.sleep(0.5)
            - Copiar archivo: pyautogui.hotkey('ctrl', 'c');time.sleep(0.5);pyautogui.hotkey('ctrl', 'v')
            
            Recuerda:
            - Para rutas de directorio usa ./directorio
            - Siempre incluye time.sleep() entre acciones
            - No incluyas importaciones ni explicaciones
            - Si no estás seguro, es mejor responder "wait" que fallar
            - No pongas la respuesta entre comillas
            
            Respuesta:
            """

            # Enviar la solicitud a Gemini
            response = self.vision_model.generate_content([prompt, image])
            response_text = response.text.strip()
            
            # Parsear la respuesta
            parsed_response, actions = self.parse_gemini_response(response_text)
            
            logger.info(f"Respuesta de Gemini: {parsed_response}")
            logger.info(f"Acciones generadas: {actions}")
            
            return parsed_response, actions
            
        except Exception as e:
            logger.error(f"Error al procesar la solicitud a Gemini: {e}")
            return f"Error: {e}", ["FAIL"]

    def reset(self, runtime_logger):
        """
        Reinicia el estado del agente para una nueva tarea.
        """
        self.trajectory_length = 0
        self.call_user_count = 0
        logger.info("Agente reiniciado")

    def get_action_space(self) -> str:
        """
        Retorna el espacio de acciones soportado por el agente.
        """
        return """
        Acciones PyAutoGUI soportadas:
        - pyautogui.click(x, y) - Hacer clic en coordenadas
        - pyautogui.doubleClick(x, y) - Doble clic
        - pyautogui.rightClick(x, y) - Clic derecho
        - pyautogui.drag(x1, y1, x2, y2) - Arrastrar
        - pyautogui.typewrite('texto') - Escribir texto
        - pyautogui.press('tecla') - Presionar tecla
        - pyautogui.hotkey('ctrl', 'c') - Combinación de teclas
        - pyautogui.scroll(clicks) - Desplazarse
        - time.sleep(seconds) - Esperar
        
        Estados especiales:
        - finished: Tarea completada
        - wait: Esperar respuesta del sistema
        - error: Error, no se puede continuar
        - call_user: Solicitar ayuda del usuario
        """


if __name__ == "__main__":
    """
    Bloque de prueba local para el agente.
    """
    try:
        # Crear una instancia del agente
        agent = BarryAgent()
        
        # Reiniciar el agente
        agent.reset(None)
        
        # Instrucción de prueba
        test_instruction = "Listar los archivos en el directorio actual"
        
        # Simular observación de prueba (necesitarías un screenshot real)
        test_obs = {
            "screenshot": b"",  # Aquí iría los bytes del screenshot real
        }
        
        print("Agente inicializado correctamente")
        print(f"Espacio de acciones:\n{agent.get_action_space()}")
        
        # Para probar realmente, necesitarías cargar un screenshot real
        # response, actions = agent.predict(test_instruction, test_obs)
        # print(f"Respuesta: {response}")
        # print(f"Acciones: {actions}")
        
    except Exception as e:
        print(f"Error en la prueba: {e}")
        print("Asegúrate de tener configurado GEMINI_API_KEY en tu archivo .env")