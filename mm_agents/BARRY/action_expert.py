import os
import logging
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

logger = logging.getLogger("action_expert")

# PROMPTS
FIRST_PROMPT="""
You are an image recognision expert.

Instruction:

{instruction}

Task:

Given the attached screenshot, reason what is needed to be done to accomplish the instruction inlcuded before.

Response:

Give a brief and detailed description of the actions that need to be done to do the instruction. The format needs to be a text. 
"""

SECOND_PROMPT="""
You are an image element recognision expert.

SOM Element Description:

{SOM_description}

Task:

Recognise the boxes that are related with the instruction and the actions needed to solve it obtained in the previous query. 
Take into account that the number of the boxes are random, related boxes don't need to have related numbers too. 
Use also the SOM description of each element to elaborate the response, you must know that the name of the icon is not always accurate. 
It tends to be wrong for the interactive element of the drag boxes.

Response:

Give me the related boxes, with a description of why are important to solve the instruction. Include the number of each box if visible.
If the number is not visible, but the correct box is present inside the picture, do not discard it.
"""

THIRD_PROMPT="""
You are an image element recognision expert.

Task:

Taking into account the previous query, tell me which of the boxes I need to interact with to solve the instruction.

Response:

The box of the SOM image which I need to use, with the justification of why did you chose this box among the other candidates. 
"""

FOURTH_PROMPT="""
Screen resolution:

{Screen_resolution}

Task:

Give me the coordinates of the center of the box you choose in the previous instruction. 
Take into account that the coordinates in the SOM description represent the upper left vertex of the box and the bottom right vertex of the box. 
All the coordinates are relative to the actual resolution of the screen.

Response:

The coordinates of the center of the box, the box chosen, in pixel format taking into acount the screen resolution and the coordinates of the vertexs in the SOM description given in the previous querys. 
"""

FIFTH_PROMPT="""
You are an expert in pyautogui code generation.

Task:

Taking into account the coordinates generated in the previous query. Give me the pyautogui code necessary to solve the instruction.

Supported pyautogui actions:

* `pyautogui.click(x, y)`
* `pyautogui.doubleClick(x, y)`
* `pyautogui.rightClick(x, y)`
* Dragging:
- You will need two instructions to drag:
* `pyautogui.moveTo(x1,y1)` (starting position)
* `pyautogui.dragTo(x2, y2, duration=duration, button=button)` (ending position)
* `pyautogui.typewrite('text')`
* `pyautogui.press('key')`
* `pyautogui.hotkey('ctrl', 'c')`
* `pyautogui.scroll(clicks)`
* `time.sleep(seconds)`

Response format:

Respond with one or a list of valid PyAutoGUI code.  All coordinates in the output must be in pixels (integers). 
Do not include comments or imports, **only** pyautogui instructions. If you return more than one instruction, write one instruction per line ('\n' after each pyautogui instruction)
"""

class ActionExpert:
    def __init__(self, model_id: str = "gemini-2.5-flash"):
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
    
    def set_current_instruction(self, new_instruction):
        """
        Current instruction setter.

        Args:
            new_instruction(str): Sets the new current instruction that needs to be solved.
        """
        self.current_instruction = new_instruction
    
    def process_instruction(self, new_screenshot, new_som_screenshot, new_som_description, reflection_feedback):
        """
        This functions provide the Pyautogui code generation needed to solve the current instruction.

        Args:
            new_screenshot(PIL Image): Vanilla image containing the current state of the OSWorld environment. 
            new_som_screenshot(PIL Image): SOM picture of the current state of the OSWorld environment.
            new_som_description(str): Description of the elements shown in the SOM screenshot.
            reflection_feedback(str): If the action expert has tried to solve the instruction and has failed by a minor error,
                                      it contains the description of the error and a solution. "" otherwise.
        
        Return:
            action(str): pyautogui code that needs to be executed within osworld environment.
        """
        try:
            logger.info("Process Instruction dentro del Action Expert")
            first_prompt = FIRST_PROMPT.format(instruction=self.current_instruction)
            logger.info("Primer prompt enviado en el Action Expert " + first_prompt)
            action = self.chat.send_message([new_screenshot, first_prompt])
            logger.info("Primera respuesta recibida por el LLM en el Action Expert: " + action.text)
            second_prompt = SECOND_PROMPT.format(SOM_description=new_som_description)
            logger.info("Segundo prompt enviado en el Action Expert " + second_prompt)
            action = self.chat.send_message([new_som_screenshot, second_prompt])
            logger.info("Segunda respuesta recibida por el LLM en el Action Expert " + action.text)
            logger.info("Tercer prompt enviado en el Action Expert " + THIRD_PROMPT)
            action = self.chat.send_message(THIRD_PROMPT)
            logger.info("Tercera respuesta recibida por el LLM en el Action Expert: " + action.text)
            screen_resolution = new_screenshot.size
            fourth_prompt = FOURTH_PROMPT.format(Screen_resolution=screen_resolution)
            logger.info("Cuarto prompt enviado en el Action Expert " + fourth_prompt)
            action = self.chat.send_message(fourth_prompt)
            logger.info("Cuarta respuesta recibida por e LLM en el Action Expert: " + action.text)
            logger.info("Quinto prompt enviado en el Action Expert " + FIFTH_PROMPT)
            action = self.chat.send_message(FIFTH_PROMPT)
            logger.info("Quinta respuesta recibida por el LLM en el Action Expert: " + action.text)
            return action.text

        except Exception as e:
            logger.error(f"Error in process instruction function of the Action Expert: {e}")
            raise

if __name__ == "__main__":
    logger.info("Probando el Action Expert")
    logger.info("Process instruction:")
    action_expert = ActionExpert()

    current_instruction="Open the browser"
    action_expert.set_current_instruction(current_instruction)

    new_screenshot_path = "./screenshot.png"
    new_screenshot = Image.open(new_screenshot_path)

    new_som_screenshot_path = "./annotated.png"
    new_som_screenshot = Image.open(new_som_screenshot_path)

    new_som_description="""
    """

    feedback = ""

    action = action_expert.process_instruction(new_screenshot, new_som_screenshot, new_som_description, feedback)
    print(action)
    
