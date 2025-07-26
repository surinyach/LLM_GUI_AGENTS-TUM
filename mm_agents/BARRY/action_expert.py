import os
import logging
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from typing import List, Tuple

logger = logging.getLogger("action_expert")

# PROMPTS

PROMPT = """
You are an expert agent for the OSWorld environment. Your sole purpose is to translate a given instruction into a precise PyAutoGUI command.

### Task
You will be provided with an instruction, a screenshot and feedback if you failed previously solving the instruction. Your task is to generate the single, exact PyAutoGUI command that accomplishes the instruction.

### Instruction
{instruction}

### Feedback
{feedback}

### Rules
1.  **Analyze screen state**: Use the screenshot to understand the current GUI state.
2.  **Determine next action**: Identify the correct PyAutoGUI command for the instruction.
3.  **Output format**: Respond with **only** a single line of valid PyAutoGUI code.  All coordinates in the output must be in pixels (integers).
4.  **Constraints**: Do not include any extra text, explanations, comments, or code imports.
5.  **Format**: Important! Do not enclose the response in backticks or any other formatting.

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
PROMPT_STEP_1 = """
This is what I need to do: "{instruction}" 
and this is my screen right now. What should I do?
"""

PROMPT_STEP_2 = """
This is the set of mark of my screen. For each element related with the instruction tell me:
1. The number of the box they are encapsulated in
2. What is this element.
3. What is its function.


REMEMBER!
The numbers of the boxes are random!!! so it is very unlikely that the chosen boxes are in order.

the sliders are not encapusalated in a whole box, only its interactable element. This interactable element usually has a wrong description

Select at least 5 elements that are close and could be raltionated with the instruction topic so you can examine them and compare them later.
"""

PROMPT_STEP_3 = """
Now i'm going to pass you the description of the set of mark of the screen that i just passed you in the previous message. 
I want you to look at the numbers you have chosen and reflect about if you got them right, If you looked at the wrong element, and if not, which number could be the right ones. 
Probably if the numbers are in order they are plrobably wrong because the numbers are random.
You could think about if the numbers are right or you should have gotten other elements taking into account the description or the coordinates of its boxes. 
For example is there is an element with a box slightly to the left of the element you chose anad its description is brightness maybe you choose the brightness slider instead of the volume slider.

{SOM_description}

"""

PROMPT_STEP_4 = """
now, taking into account that this is the resolution of the screen: {width}x{height} 
Give me te coordenates of the center of the box I have to click.
"""

PROMPT_STEP_5 = """
Now give me the command (or the set of commands) in pyAutoGUI to perform the instruction with the coordinates of the last message.

Supported pyautogui actions:
* pyautogui.click(x, y)
* pyautogui.doubleClick(x, y)
* pyautogui.rightClick(x, y)

* Dragging:
- You will need two instructions to drag:
* pyautogui.moveTo(x1,y1) (starting position)
* pyautogui.dragTo(x2, y2, duration=duration, button=button) (ending position)

* pyautogui.typewrite('text')
* pyautogui.press('key')
* pyautogui.hotkey('ctrl', 'c')
* pyautogui.scroll(clicks)
* time.sleep(seconds)

The response MUST start with the exact phrase "RESPONSE:" on its own line, followed immediately by the response.
If there is more than one command it should be separated only by a ';'. 

Always add time.sleep(1) at the end of the commands

Example of how the response should appear:
RESPONSE: pyautogui.click(x, y);pyautogui.typewrite('text');time.sleep(1)
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

        self.last_printed_index = 0
        self.current_instruction = ""
    
    # GLOBAL FUNCTIONS 

    def _parse_subtask_response(self, response_text: str, marker) -> List[str]:
        """
        Parses the raw text response from the LLM, by splitting on marker
        and taking the latter part. Assumes subtasks are separated by ';'.

        Args:
            response_text (str): The full string response from the LLM.

        Returns:
            List[str]: A list containing each parsed subtask.
                       Returns an empty list if the marker is not found or no subtasks are present.
        """
        if not response_text:
            return []
        parts = response_text.split(marker, 1) # Use 1 to split only on the first occurrence

        if len(parts) < 2:
            print(f"Warning: Marker '{marker}' not found in the response.")
            return []

        # Grab the part after the marker and strip leading/trailing whitespace (including newlines)
        subtasks_raw_string = parts[1].strip()

        # Split by semicolon and clean each individual task.
        subtasks = [task.strip() for task in subtasks_raw_string.split(';') if task.strip()]
        return subtasks
    
    def set_current_instruction(self, new_instruction):
        """
        Current instruction setter.

        Args:
            new_instruction(str): Sets the new current instruction that needs to be solved.
        """
        self.current_instruction = new_instruction
    
    def _print_history(self):
        for i in range(self.last_printed_index, len(self.chat.history)):
            message = self.chat.history[i]
            text_content = message.parts[0].text 
            print(f"{message.role}: {{ \"{text_content}\" }}")
        
        self.last_printed_index = len(self.chat.history)
    
    def process_instruction(self, screenshot, SOM_screenshot, SOM_description):
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
            logger.info("Processing the next instruction:" + self.current_instruction)
            prompt= PROMPT_STEP_1.format(
                instruction=self.current_instruction
            )
            self.chat.send_message([prompt, screenshot])

            prompt= PROMPT_STEP_2
            self.chat.send_message([prompt, SOM_screenshot])

            prompt= PROMPT_STEP_3.format(
                SOM_description=SOM_description
            )
            self.chat.send_message(prompt)

            width, height = screenshot.size
            prompt= PROMPT_STEP_4.format(
                width=width,
                height=height
            )
            self.chat.send_message(prompt)

            prompt= PROMPT_STEP_5
            response = self.chat.send_message(prompt)
            actions = self._parse_subtask_response(response.text, "RESPONSE:")
            return actions

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
    Parsed elements:
    - Text: 'transcript_of_re;' (Bounding Box: [0.127, 0.361, 0.182, 0.376], non-interactive)
    - Text: 'CURRICULUMS' (Bounding Box: [0.066, 0.793, 0.12, 0.812], non-interactive)
    - Text: '21*€' (Bounding Box: [0.033, 0.958, 0.051, 0.973], non-interactive)
    - Text: '18.21' (Bounding Box: [0.968, 0.955, 0.989, 0.973], non-interactive)
    - Text: 'dx 0J' (Bounding Box: [0.909, 0.961, 0.94, 0.983], non-interactive)
    - Text: 'Mayorm: soleado' (Bounding Box: [0.033, 0.977, 0.095, 0.992], non-interactive)
    - Text: '18/07/2025' (Bounding Box: [0.946, 0.973, 0.99, 0.992], non-interactive)
    - Icon: 'Busqueda ' (Bounding Box: [0.256, 0.954, 0.402, 0.992], Interactive)
    - Icon: 'Papelera de reciclaje ' (Bounding Box: [0.005, 0.003, 0.059, 0.11], Interactive)
    - Icon: 'Adobe Acrobat ' (Bounding Box: [0.059, 0.0, 0.128, 0.114], Interactive)
    - Icon: 'Oracle VM VirtualBox ' (Bounding Box: [0.943, 0.006, 0.992, 0.11], Interactive)
    - Icon: 'TUM ' (Bounding Box: [0.131, 0.001, 0.178, 0.118], Interactive)
    - Icon: 'Engine exe ' (Bounding Box: [0.951, 0.448, 0.99, 0.525], Interactive)
    - Icon: 'Arduino IDE ' (Bounding Box: [0.949, 0.151, 0.992, 0.234], Interactive)
    - Icon: 'VMware Workstation Pro ' (Bounding Box: [0.881, 0.146, 0.936, 0.258], Interactive)
    - Icon: 'KeePass ' (Bounding Box: [0.072, 0.436, 0.117, 0.553], Interactive)
    - Icon: 'GIMP 2.10.38 ' (Bounding Box: [0.878, 0.004, 0.94, 0.121], Interactive)
    - Icon: 'INLAB ' (Bounding Box: [0.068, 0.573, 0.117, 0.694], Interactive)
    - Icon: 'FortiClient VPN ' (Bounding Box: [0.0, 0.135, 0.067, 0.265], Interactive)
    - Icon: 'PDF Expedient_ECT_ ' (Bounding Box: [0.117, 0.437, 0.183, 0.556], Interactive)
    - Icon: 'PALAU INSCRIPCIO_ ' (Bounding Box: [0.072, 0.29, 0.112, 0.393], Interactive)
    - Icon: 'Microsoft Edge ' (Bounding Box: [0.003, 0.573, 0.058, 0.693], Interactive)
    - Icon: 'Steam ' (Bounding Box: [0.953, 0.298, 0.989, 0.38], Interactive)
    - Icon: 'duet ' (Bounding Box: [0.014, 0.295, 0.048, 0.406], Interactive)
    - Icon: 'PDF Transcript_of ' (Bounding Box: [0.121, 0.573, 0.188, 0.696], Interactive)
    - Icon: 'Firefox_ ' (Bounding Box: [0.014, 0.442, 0.05, 0.528], Interactive)
    - Icon: 'Obsidian ' (Bounding Box: [0.011, 0.733, 0.056, 0.824], Interactive)
    - Icon: 'Cheat Engine ' (Bounding Box: [0.883, 0.307, 0.932, 0.379], Interactive)
    - Icon: 'PDF ' (Bounding Box: [0.142, 0.296, 0.17, 0.357], Interactive)
    - Icon: 'Shuffle 3' (Bounding Box: [0.689, 0.956, 0.716, 0.996], Interactive)
    - Icon: 'Settings' (Bounding Box: [0.748, 0.956, 0.771, 0.997], Interactive)
    - Icon: 'Windows' (Bounding Box: [0.228, 0.952, 0.254, 0.992], Interactive)
    - Icon: 'Microsoft 365' (Bounding Box: [0.719, 0.957, 0.744, 0.997], Interactive)
    - Icon: 'Internet Explorer - web browser' (Bounding Box: [0.577, 0.953, 0.602, 0.992], Interactive)
    - Icon: 'folder' (Bounding Box: [0.49, 0.955, 0.513, 0.99], Interactive)
    - Icon: 'Microsoft Edge browser.' (Bounding Box: [0.52, 0.955, 0.541, 0.99], Interactive)
    - Icon: 'Movie Maker' (Bounding Box: [0.605, 0.953, 0.628, 0.992], Interactive)
    - Icon: 'Copy' (Bounding Box: [0.405, 0.951, 0.428, 0.99], Interactive)
    - Icon: 'Firefox' (Bounding Box: [0.632, 0.954, 0.658, 0.994], Interactive)
    - Icon: 'Teams' (Bounding Box: [0.461, 0.954, 0.484, 0.989], Interactive)
    - Icon: 'Microsoft OneNote' (Bounding Box: [0.434, 0.954, 0.454, 0.988], Interactive)
    - Icon: 'Text Box' (Bounding Box: [0.664, 0.955, 0.685, 0.995], Interactive)
    - Icon: 'Toggle Lock' (Bounding Box: [0.924, 0.947, 0.94, 1.0], Interactive)
    - Icon: 'Microsoft Office.' (Bounding Box: [0.549, 0.954, 0.57, 0.989], Interactive)
    - Icon: 'Redo' (Bounding Box: [0.871, 0.949, 0.89, 0.996], Interactive)
    - Icon: 'M0,0L9,0 4.5,5z' (Bounding Box: [0.83, 0.953, 0.847, 0.993], Interactive)
    - Icon: 'Dismiss' (Bounding Box: [0.908, 0.947, 0.924, 0.999], Interactive)
    - Icon: 'WiFi' (Bounding Box: [0.892, 0.948, 0.909, 0.997], Interactive)
    - Icon: 'Cloud' (Bounding Box: [0.848, 0.949, 0.869, 0.994], Interactive)
    - Icon: 'Norton Weather' (Bounding Box: [0.008, 0.949, 0.033, 0.994], Interactive)
    - Icon: 'The File Manager - Home Manager' (Bounding Box: [0.071, 0.141, 0.114, 0.26], Interactive)
    - Icon: 'Initiative' (Bounding Box: [0.076, 0.738, 0.109, 0.786], Interactive)
    - Icon: 'a beach or ocean view.' (Bounding Box: [0.99, 0.002, 1.0, 0.104], Interactive)
    - Icon: 'a table or counter.' (Bounding Box: [0.991, 0.444, 1.0, 0.529], Interactive)
    - Icon: 'M0,0L9,0 4.5,5z' (Bounding Box: [0.0, 0.948, 0.007, 0.998], Interactive)
    """

    feedback = ""

    action = action_expert.process_instruction(new_screenshot, new_som_screenshot, new_som_description, feedback)
    print(action)
    
