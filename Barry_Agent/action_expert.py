import os
import logging
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

logger = logging.getLogger("action_expert")

# PROMPTS
FIRST_PROMPT="""
Instruction:
{instruction}

Feedback:
Consider any previous execution errors and solutions provided in this section to avoid repeating mistakes:
{Reflection_feedback}

Task:
Given the attached screenshot, describe the necessary actions to accomplish the instruction.

Response:
Provide a clear, concise, and detailed description of the actions to be performed.
"""

SECOND_PROMPT="""
SOM Element Description:
{SOM_description}

Task:
From the SOM description, identify boxes relevant to accomplishing the instruction and the actions from the previous query.
Note: Box numbers are random and not necessarily related. Element names in the SOM description may not always be accurate, especially for drag bar interactive elements.

Response:
List the relevant boxes, including their visible numbers. For each box, describe why it is important to solve the instruction. 
Do not discard relevant boxes even if their number is not visible in the image.
"""

THIRD_PROMPT="""
Task:
Taking into account the previous query, tell me which of the boxes I need to interact with to solve the instruction.
Also, consider that the precise interactive element might not always be represented as a distinct box in the SOM but could be in close proximity to a highly related box. 
Leverage your VLM capabilities to identify such unboxed but critical interaction areas by analyzing the screenshot.

Response:
The box of the SOM image which I need to use, with the justification of why did you chose this box among the other candidates.
"""

FOURTH_PROMPT="""
Screen resolution:
{Screen_resolution}

Task:
Calculate the center coordinates (x, y) of the box chosen in the previous instruction.
Note: SOM description coordinates are (upper_left_x, upper_left_y, bottom_right_x, bottom_right_y) and are relative to the screen resolution.

Response:
Provide the center coordinates (x, y) of the chosen box in pixel format, considering the screen resolution and the SOM vertex coordinates.
"""

FIFTH_PROMPT="""
Task:
Generate the necessary PyAutoGUI code to solve the instruction, using the coordinates from the previous query.

Supported pyautogui actions:
* `pyautogui.click(x, y)`
* `pyautogui.doubleClick(x, y)`
* `pyautogui.rightClick(x, y)`
* Dragging (requires two instructions):
    * `pyautogui.moveTo(x1, y1)` (starting position)
    * `pyautogui.dragTo(x2, y2, duration=duration, button=button)` (ending position)
* `pyautogui.typewrite('text')`
* `pyautogui.press('key')`
* `pyautogui.hotkey('ctrl', 'c')`
* `pyautogui.scroll(clicks)`
* `time.sleep(seconds)`

Response format:
Respond with one or a list of valid PyAutoGUI code. All coordinates in the output must be in pixels (integers).
Do not include comments or imports, **only** pyautogui instructions. If multiple instructions, write one instruction per line.
"""

class ActionExpert:
    def __init__(self, model_id: str = "gemini-2.5-flash"):
        """
        Action Expert in screen element recognition and PyAutoGUI code generation.
        It uses a Chain-of-Thought prompt mechanism to analyze the task, understand the SOM element description, 
        and finally generate the next action in PyAutoGUI code.
        """
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY not found in the .env file for ActionExpert")
            raise ValueError("GEMINI_API_KEY not found in the .env file")
        
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
            logger.info("Process Instruction inside the Action Expert")
            first_prompt = FIRST_PROMPT.format(instruction=self.current_instruction, Reflection_feedback=reflection_feedback)
            logger.info("CURRENT INSTRUCTION INSIDE THE ACTION: " + self.current_instruction)
            action = self.chat.send_message([new_screenshot, first_prompt])
            second_prompt = SECOND_PROMPT.format(SOM_description=new_som_description)
            action = self.chat.send_message([new_som_screenshot, second_prompt])
            action = self.chat.send_message(THIRD_PROMPT)
            screen_resolution = new_screenshot.size
            fourth_prompt = FOURTH_PROMPT.format(Screen_resolution=screen_resolution)
            action = self.chat.send_message(fourth_prompt)
            action = self.chat.send_message(FIFTH_PROMPT)
            return action.text

        except Exception as e:
            logger.error(f"Error in process instruction function of the Action Expert: {e}")
            raise

if __name__ == "__main__":
    pass