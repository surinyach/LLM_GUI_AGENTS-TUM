import os
import logging
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

logger = logging.getLogger("action_expert")

# --- LOGGING CONFIGURATION WITH COLORS ---
# Define color codes for different log levels
class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add color to log messages based on level.
    """
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    def format(self, record):
        log_message = super().format(record)
        if record.levelno == logging.INFO:
            return f"{self.GREEN}{log_message}{self.RESET}"
        if record.levelno == logging.ERROR:
            return f"{self.RED}{log_message}{self.RESET}"
        return log_message

logger = logging.getLogger("action_expert")
logger.setLevel(logging.INFO)

# Create a handler for the logger
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# PROMPTS

PROMPT = """
You are an expert in translating action descriptions into PyAutoGUI commands. Your goal is to act as a precise, command-line interface for the OSWorld environment.

### Task
You will be given a series of user instructions, a SOM (Semantic Object Map) screenshot of the environment, and a description of each component. Your sole task is to determine the single, exact PyAutoGUI command that needs to be executed to complete the next step of the instruction.

### Input
- **SOM description**: 'som_description'
- **Goal**: 'instruction_list'

### Output
Your response must be one of the following:
1.  **A single line of valid PyAutoGUI code**
2.  **`'finish'`**: If you believe all instructions are accomplished.
3.  **`'error: <explanation why you can't perform the action>'`**: If a step is impossible to complete (e.g., you cannot find an element described in the instructions).

### Constraints
* Do not include any explanations, additional text, imports, or quotes around the response.
* Only the PyAutoGUI code, `'finish'`, or the specified error format should be returned.

### Supported PyAutoGUI Actions:
* `pyautogui.click(x, y)` - Click at coordinates
* `pyautogui.doubleClick(x, y)` - Double click
* `pyautogui.rightClick(x, y)` - Right click
* `pyautogui.drag(x1, y1, x2, y2)` - Drag
* `pyautogui.typewrite('text')` - Type text
* `pyautogui.press('key')` - Press key
* `pyautogui.hotkey('ctrl', 'c')` - Key combination
* `pyautogui.scroll(clicks)` - Scroll
* `time.sleep(seconds)` - Wait

### Examples of input and their expected output:
* Open the terminal: `pyautogui.hotkey('ctrl', 'alt', 't')`
* Type 'hello': `pyautogui.typewrite('hello')`
* Click at the spotify icon: `pyautogui.click(100, 200)`
* Press the 'enter' key: `pyautogui.press('enter')`
* Double click at the test files folder: `pyautogui.doubleClick(500, 300)`
"""

SUMMARY_PROMPT="""
You are an expert at summarizing a list of executed instructions.

Your task is to review the conversation and generate a brief summary of the completed steps. The summary should be a concise list of what was done, based on the actions you took in the previous chat turns.

Focus on describing the high-level actions, such as "Opened the terminal," "Typed 'hello' into the terminal," or "Clicked on the 'test files' folder." Do not include the raw PyAutoGUI code in the summary.

Provide only the summary, with each step on a new line. Do not include any other text, explanations, or quotes.
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
        
        self.instruction_list = ""
        self.subtask = ""

    # GLOBAL FUNCTIONS 
    
    def get_instruction_list(self):
        """
        Instruction list getter.

        Return:
            instruction_list(str): The instruction list to complete the current subtask.
        """
        return self.instruction_list
    
    def set_subtask_and_instructions(self, new_subtask, new_instruction_list):
        """
        Subtask and instruction list setter.

        Args:
            new_subtask(str): new subtask to solve.
            new_instruction_list(str): new instruction list to be processed.
        """
        self.subtask = new_subtask
        self.instruction_list = new_instruction_list

    def process_instruction(self, new_som_screenshot, new_som_description):
        """
        This functions has three possible outcomes.
            1. Pyautogui code generation needed to solve the current instruction.
            2. If the instruction is impossible to execute, notify it.
            3. If all of the instructions are finished, notify it.

        Args:
            new_som_screenshot(PIL Image): SOM picture of the current state of the OSWorld environment.
            new_som_description(str): Description of the elements shown in the SOM screenshot.
        
        Return:
            1. action(str): pyautogui code that needs to be executed within osworld environment.
            2. action(str): "error: description of the error"
            3. action(str): "finish"
        """
        try:
            logger.info("Process Instruction dentro del Action Expert")
            prompt= PROMPT.replace("'som_description'", new_som_description).replace("'instruction_list'", self.instruction_list)
            logger.info("Prompt enviado al LLM dentro del Action Expert: " + prompt)
            action = self.chat.send_message([prompt, new_som_screenshot])
            logger.info("Respuesta recibida por el LLM en el Action Expert: " + action.text)
            return action.text

        except Exception as e:
            logger.error(f"Error in process instruction function of the Action Expert: {e}")
            raise

    def summarize_done_instructions(self):
        """
        Generates a summary of the actions done by the action expert by now.

        Return:
            feedback(str): Summary of all the actions performed by the Action Expert LLM.
        """
        logger.info("Summarize done instructions dentro del Action Expert")
        feedback = self.chat.send_message(SUMMARY_PROMPT)
        logger.info("Summary generated by the Action Expert: " + feedback.text)
        return feedback.text

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

    new_som_description="""
    Parsed elements:
    {'type': 'text', 'bbox': [0.008872651495039463, 0.006499535869807005, 0.051670145243406296, 0.02135561779141426], 'interactivity': False, 'content': 'Actividades', 'source': 'box_ocr_content_ocr'}
    {'type': 'text', 'bbox': [0.5015657544136047, 0.002785515272989869, 0.524530291557312, 0.02135561779141426], 'interactivity': False, 'content': 'dejul', 'source': 'box_ocr_content_ocr'}
    {'type': 'text', 'bbox': [0.5266179442405701, 0.006499535869807005, 0.5464509129524231, 0.019498607143759727], 'interactivity': False, 'content': '1427', 'source': 'box_ocr_content_ocr'}
    {'type': 'icon', 'bbox': [0.03536141291260719, 0.030255816876888275, 0.10708547383546829, 0.14553005993366241], 'interactivity': True, 'content': 'Carpeta personal ', 'source': 'box_yolo_content_ocr'}
    {'type': 'icon', 'bbox': [0.0449901819229126, 0.13491632044315338, 0.09136451035737991, 0.22245870530605316], 'interactivity': True, 'content': 'FIBUPC ', 'source': 'box_yolo_content_ocr'}
    {'type': 'icon', 'bbox': [4.867613461101428e-05, 0.4103085398674011, 0.037095509469509125, 0.46590089797973633], 'interactivity': True, 'content': 'Apple', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.00013839901657775044, 0.34838080406188965, 0.03669852390885353, 0.4026133418083191], 'interactivity': True, 'content': 'Formatted', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.0, 0.4720260500907898, 0.03725579380989075, 0.5329386591911316], 'interactivity': True, 'content': 'Microsoft OneNote', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [5.794763637823053e-05, 0.15829430520534515, 0.03724779561161995, 0.21547317504882812], 'interactivity': True, 'content': 'File Manager', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [1.8614531654748134e-05, 0.28211989998817444, 0.03697828948497772, 0.34342771768569946], 'interactivity': True, 'content': 'YellowYellow', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.0, 0.5384803414344788, 0.036886684596538544, 0.5952269434928894], 'interactivity': True, 'content': 'Draw Functions', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.0, 0.22151590883731842, 0.037247903645038605, 0.27791860699653625], 'interactivity': True, 'content': 'Help', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.0, 0.6592819690704346, 0.03761951997876167, 0.7191570997238159], 'interactivity': True, 'content': 'Glasses', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.00012264847464393824, 0.09299878031015396, 0.03713191673159599, 0.15306594967842102], 'interactivity': True, 'content': 'Weather Weather VPN', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.00018414558144286275, 0.7900382280349731, 0.03638904541730881, 0.8515233993530273], 'interactivity': True, 'content': 'Ink Pro', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.0, 0.9140973687171936, 0.03850214555859566, 0.9951076507568359], 'interactivity': True, 'content': 'Calendar or date.', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.0, 0.02760356292128563, 0.03652464598417282, 0.08943263441324234], 'interactivity': True, 'content': 'Firefox', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.0003614485322032124, 0.7270521521568298, 0.03756094351410866, 0.7807366847991943], 'interactivity': True, 'content': 'WeChat', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [4.2194129491690546e-05, 0.5977453589439392, 0.03676231950521469, 0.654191255569458], 'interactivity': True, 'content': 'Toggle Terminal', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.00036143363104201853, 0.8581324815750122, 0.036128368228673935, 0.9108204245567322], 'interactivity': True, 'content': 'Spotify - Music and Podcasts', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.9820717573165894, 0.0, 0.9954875111579895, 0.02488916739821434], 'interactivity': True, 'content': 'Paste', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.9291857481002808, 0.0, 0.9440109133720398, 0.02626079134643078], 'interactivity': True, 'content': 'Horton Security', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.9523779153823853, 0.0, 0.9673864841461182, 0.02580481395125389], 'interactivity': True, 'content': 'Properties', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.9673172235488892, 0.0, 0.9816845655441284, 0.025056036189198494], 'interactivity': True, 'content': 'Toggleoggleoggle Screen Screen Screen Time', 'source': 'box_yolo_content_yolo'}
    {'type': 'icon', 'bbox': [0.11300404369831085, 0.24912036955356598, 0.8621399998664856, 0.9113406538963318], 'interactivity': True, 'content': 'a mountain or a mountain top.', 'source': 'box_yolo_content_yolo'}
    """

    action = action_expert.process_instruction(new_som_screenshot, new_som_description)
    feedback = action_expert.summarize_done_instructions()
    
