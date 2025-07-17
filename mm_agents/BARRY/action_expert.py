import os
import logging
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

logger = logging.getLogger("action_expert")

# PROMPTS

INITIAL_PROMPT = """
You are an expert agent for the OSWorld environment. Your sole purpose is to translate high-level instructions into precise PyAutoGUI commands.

### Instructions

1.  **Analyze the screen state**: You will be provided with a Semantic Object Map (SOM) description of the current screen, including a list of interactive and non-interactive elements, their content, and their bounding box coordinates.
2.  **Determine the next action**: Based on the provided SOM and a list of step-by-step instructions, you must identify the single, exact PyAutoGUI command required to complete the next instruction.
3.  **Return the command**: Your response must be the command itself, with no additional text or explanation.

### Input Format

- **SOM description**: A string containing a list of screen elements with their type, content, interactivity, and bounding box. Example: `- Icon: 'Firefox' (Bounding Box: [0.0, 0.028, 0.037, 0.089], Interactive)`
- **Task**: The overall goal of the session.
- **Instruction list**: A numbered list of subtasks to be completed. You must always address the very next incomplete instruction.

### Output Format

Your response must be one of the following, and nothing else:

1.  **A single line of valid PyAutoGUI code**.
    * Examples: `pyautogui.click(100, 200)`, `pyautogui.typewrite('hello')`
2.  **`finish`**: If the entire task is complete and all instructions have been accomplished.
3.  **`error: <explanation why action is impossible>`**: If the current step cannot be performed.

### Constraints

-   Your response must be a single line.
-   Do not include any explanations, comments, or extra text.
-   Do not include any imports (e.g., `import pyautogui`).
-   Do not enclose the response in backticks or any other formatting.

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

---

### Current Exection context

- **Task**: 'task'
- **Instruction list**:
  'instruction_list'
"""

ACTION_PROMPT="""
### Current Screen State

- **New SOM description**:
'SOM_description'

Proceed to the next required action based on the task and instructions already provided, using both the attached image and this description of its elements.
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
        Also resets and initializes the chat history with the LLM with the necessary information.

        Args:
            new_subtask(str): new subtask to solve.
            new_instruction_list(str): new instruction list to be processed.
        """
        self.subtask = new_subtask
        self.instruction_list = new_instruction_list

        # Reset the chat history providing the new initial context
        initial_prompt = INITIAL_PROMPT.replace("'task'", subtask).replace("'instructio_list'", instruction_list)
        initial_context = [
            {
                "role": "user",
                "parts": initial_prompt
            }
        ]
        logger.info("Initial context of the Action Expert:" + initial_context)
        self.chat = self.model.start_chat(history=initial_context)

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
            prompt= ACTION_PROMPT.replace("'SOM_description'", new_som_description)
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

    new_som_description=""

    action = action_expert.process_instruction(new_som_screenshot, new_som_description)
    feedback = action_expert.summarize_done_instructions()
    
