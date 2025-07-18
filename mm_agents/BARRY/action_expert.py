import os
import logging
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

logger = logging.getLogger("action_expert")

# PROMPTS

PROMPT = """
You are an expert agent for the OSWorld environment. Your sole purpose is to translate a given instruction into a precise PyAutoGUI command.

### Task
You will be provided with an instruction, a screenshot, and a Semantic Object Map (SOM) description. Your task is to generate the single, exact PyAutoGUI command that accomplishes the instruction.

### Instruction
{instruction}

### Semantic Object Map (SOM)
{som_description}

### Rules
1.  **Analyze screen state**: Use the screenshot and SOM to understand the current GUI state.
2.  **Determine next action**: Identify the correct PyAutoGUI command for the instruction.
3.  **Output format**: Respond with **only** a single line of valid PyAutoGUI code.
4.  **Constraints**: Do not include any extra text, explanations, comments, or code imports. Do not enclose the response in backticks or any other formatting.

### Supported PyAutoGUI Actions
* `pyautogui.click(x, y)`
* `pyautogui.doubleClick(x, y)`
* `pyautogui.rightClick(x, y)`
* `pyautogui.drag(x1, y1, x2, y2)`
* `pyautogui.typewrite('text')`
* `pyautogui.press('key')`
* `pyautogui.hotkey('ctrl', 'c')`
* `pyautogui.scroll(clicks)`
* `time.sleep(seconds)`
"""

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

        self.current_instruction = ""
    

    # GLOBAL FUNCTIONS 
    
    def get_instruction_list(self):
        """
        Instruction list getter.

        Return:
            instruction_list(str): The instruction list to complete the current subtask.
        """
        return self.instruction_list
    
    def set_current_instruction(self, new_instruction):
        """
        Current instruction setter.

        Args:
            new_instruction(str): Sets the new current instruction that needs to be solved.
        """
        self.set_current_instruction(new_instruction)
    
    def process_instruction(self, new_screenshot, new_som_screenshot, new_som_description):
        """
        This functions provide the Pyautogui code generation needed to solve the current instruction.

        Args:
            new_screenshot(PIL Image): Vanilla image containing the current state of the OSWorld environment. 
            new_som_screenshot(PIL Image): SOM picture of the current state of the OSWorld environment.
            new_som_description(str): Description of the elements shown in the SOM screenshot.
        
        Return:
            action(str): pyautogui code that needs to be executed within osworld environment.
        """
        try:
            logger.info("Process Instruction dentro del Action Expert")
            prompt= PROMPT.format(
                instruction=self.current_instruction,
                SOM_Description=new_som_description
            )
            logger.info("Prompt enviado al LLM dentro del Action Expert: " + prompt)
            action = self.chat.send_message([prompt, new_screenshot, new_som_screenshot])
            logger.info("Respuesta recibida por el LLM en el Action Expert: " + action.text)
            return action.text

        except Exception as e:
            logger.error(f"Error in process instruction function of the Action Expert: {e}")
            raise

if __name__ == "__main__":
    logger.info("Probando el Action Expert")
    logger.info("Process instruction:")
    action_expert = ActionExpert()
    subtask = """
    Open the browser
    """ 
    instruction_list = """
    1. Click on the browser logo to open it
    """
    action_expert.set_subtask_and_instructions(subtask, instruction_list)

    new_som_screenshot_path = "./screenshot.png"
    new_som_screenshot = Image.open(new_som_screenshot_path)

    new_som_description=""

    action = action_expert.process_instruction(new_som_screenshot, new_som_description)
    feedback = action_expert.summarize_done_instructions()
    
