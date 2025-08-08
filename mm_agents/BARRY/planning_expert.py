import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from .utils import parse_llm_response
from datetime import datetime


logger = logging.getLogger("planning_expert")

DECOMPOSE_MAIN_TASK_PROMPT_TEMPLATE = """
This is the main task: "{main_task}"
Give me the first subtask to acoplish the main task The subtask must be goals and they should be used for guidance. 
Avoid subtasks that involve taking screenshots, locating elements, or recording coordinates, as the agent has screen markers for execution. 
Identify the active application or window in the screenshot to ensure subtasks align with the current context (e.g., browser, file explorer). 
Do not include a final subtask like 'Finish the task'; each subtask must be meaningful.

Here's how I want you to structure your response:
1.  **Reasoning Process:** Write down your thought process here.
2.  **Final Subtasks:** The subtask MUST start with the exact phrase "RESPONSE:" on its own line, followed immediately by the subtask.
"""

RETHINK_SUBTASK_PROMPT_TEMPLATE = """
Give me the next subtask to acomplish the main task.
Do not repeat approaches that failed, as indicated by this feedback (if any): {reflection_expert_feedback}.
If the current subtask "{current_subtask}" was completed successfully, determine the next subtask.
Analyze the screenshot to identify the active application, visible elements (e.g., browser tabs, search bars), and current state.
The subtask must be a goal and it should be used for guidance. It is not a instruction to perform the task. 
Avoid subtasks that involve taking screenshots, locating elements, or recording coordinates, as the agent has screen markers for execution. 
Identify the active application or window in the screenshot to ensure subtask aligns with the current context (e.g., browser, file explorer). 
If there is NOTHING to do because the main task is done make an instruction with an sleep of 1 second.

Reasoning Process:
1. Review the feedback (if any) to identify what went wrong or what subtask was completed.
2. Analyze the screenshot to determine the active application, window state, and visible elements.
3. If feedback indicates failure, devise an alternative approach (e.g., use a different application, a different path to get to the same point).
4. If the subtask was completed, identify the next goal to complete the main task.


Here's how I want you to structure your response:
1.  **Reasoning Process:**  Write down your thought process here.
2.  **Final Subtask:** The subtask MUST start with the exact phrase "RESPONSE:" on its own line, followed immediately by the subtask.
"""


DECOMPOSE_SUBTASK_PROMPT_TEMPLATE = """
Decompose the subtask "{current_subtask}" into detailed, actionable instructions. 
IMPORTANT THINGS TO TAKE INTO ACCOUNT:
0. There is no need to decompose a sequence of keys in different instructions. In pyAutoGUI you can press different keys at the same time and it does not need different instructions.
1. Analyze the screenshot to identify the active application, visible elements (e.g., address bar, search bar, buttons), and window state. 
3. Think about how to execute the instruction using combination of hotkeys. 
4. Combine related actions (e.g., click, select text with Ctrl+A, type and press enter) into a SINGLE instruction (NOT SEPARATED WITH ';') where appropriate. 
   If there is text were it should be clicked there is no need to use hotkeys as the llm is good clicking where there is text! in the SAME instruction, not in different instructions 
5. Avoid instructions for screenshots, locating elements, or recording coordinates, as the agent has screen markers. 
6. If an element is ambiguous (e.g., multiple search bars), specify which one (e.g., 'the browser's address bar').
7. Do not make any instruction of release button as in pyAutoGUI there is not such instruction.
8. Do not put 'if' in instructions, you are being passed a screenshot. You decide what to do.
9. Remember that if there is text on a text box you will have to do ctrl + A before typing the new text


If it is need it to click on a place where there is no icon or text describe its position referencing a place where there is text or a icon.

Here's how I want you to structure your response:
1.  **Reasoning Process:** Write down your thought process here and the first version of your answer.
2.  **Revised Instruction List:** After your reasoning, you MUST provide the revised list of the instruction list. This list MUST start with the exact phrase "RESPONSE:" on its own line, followed immediately by the instructions. 
All text from "RESPONSE:" until the end of your response will be considered the revised list of instructions. Each instruction should be separated by a semicolon ';'.

Example of how the revised subtask list should appear:
RESPONSE: Click on the browser icon; Click on the search bar and Type dogs.
"""

DECOMPOSE_SUB_TASK_PROMPT_TEMPLATE_REFLECT = """
Reflect Questions:
2. For actions like opening new tabs, saving, selecting all text, or maximizing/minimizing windows, have keyboard shortcuts (e.g., Ctrl+T, Ctrl+S, Ctrl+A) been used instead of mouse clicks where applicable? (if there is a text on the area where it should be clicked there is no need of using hotkeys)
3. Were hotkeys used for actions where visual recognition might be difficult (e.g., an empty bookmark bar, using Ctrl+Shift+B to make it visible)?
4. Are the chosen hotkeys standard and widely applicable for the identified active application?
5. Are there any instructions that instruct the agent to "take a screenshot," "locate the X icon," or "find the coordinates of the button"? (There should not been)
6. Some steps could be performed secuentally without needing to get another screenshot, put them in the same instruction (steps like click and type MUST be in the same instruction, not separated by a ;)
7. Do any of the instructions include instructions to "release" a key or mouse button?
8. Are there any 'if' in the instructions? (it should not be)

Here's how I want you to structure your response:
1.  **Reflect Process:** Review your answer answering the Reflect questions.
2.  **Revised Instruction List:** After your reasoning, you MUST provide the revised list of the instruction list. This list MUST start with the exact phrase "RESPONSE:" on its own line, followed immediately by the instructions. 
All text from "RESPONSE:" until the end of your response will be considered the revised list of instructions. Each instruction should be separated by a semicolon ';'

Example of how the revised subtask list should appear:
RESPONSE: Click on the browser icon; Click on the search bar and Type dogs.
"""

IS_LAST_TASK_PROMPT_TEMPLATE = """
The subtask "{current_subtask}" was completed successfully.
The main task is: "{main_task}".
Analize if there is still need to press a save or done button or click outside th text box
Analyze the screenshot to verify if the main task is complete (e.g., check for a downloaded file, specific UI state, or visible result).
Respond with 'yes' if the main task is fully accomplished, or 'no' if additional steps are needed.

Here's how I want you to structure your response:
1.  **Reasoning Process:** Write down your thought process here.
2.  **Final answer:** After your reasoning, you MUST respond with 'RESPONSE:' followed only with a 'yes' or 'no'

Example: 
RESPONSE:yes
"""


class PlanningExpert:
    def __init__(self, model_id: str = "gemini-2.0-flash"):
        """
        Initialitation of the Planning Expert
        """
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY not found in the file .env for Planning Expert")
            raise ValueError("GEMINI_API_KEY not found in the file .env")
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_id)

        self.chat = self.model.start_chat(history=[])
        self.last_printed_index = 0 # this is for printing the chat history for debugging

        self.main_task = ""
        self.current_subtask = ""

        self.first_iter = True    

        # Set up log file directory and path
        self.log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)  # Create logs directory if it doesn't exist
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f'chat_history_{timestamp}.log')
    
    def _save_chat_history_to_file(self):
        """
        Saves the chat history to a log file since the last printed index.
        Each message includes a timestamp, role, and content.
        """
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                for i in range(self.last_printed_index, len(self.chat.history)):
                    message = self.chat.history[i]
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    text_content = message.parts[0].text
                    log_entry = f"[{timestamp}] {message.role} PLANNING EXPERT: {text_content}\n"
                    f.write(log_entry)
                self.last_printed_index = len(self.chat.history)
        except Exception as e:
            logger.error(f"Error saving chat history to file: {e}")
            raise
    
    def decompose_main_task(self, main_task, screenshot):
        """
        Receives the main task and uses an LLM to generate the first subtask.

        This function initiates the planning phase by instructing a language model (LLM)
        to break down a high-level `main_task` into smaller, manageable subtasks.
        The LLM is specifically prompted to provide the first subtask.
        The `main_task` and the screenshot representing the current GUI
        state are sent to the LLM to provide necessary context for decomposition.
        The first generated subtask is stored as `self.current_subtask` and is also
        returned as the initial subtask for subsequent action or execution.

        Args:
            main_task (str): The primary objective or high-level goal that needs to be decomposed.
            screenshot: screenshot of the current GUI environment. This provides visual and 
            interactive context to the LLM during the task decomposition process.

        Returns:
            str: The first subtask from the main task.
        """

        try:
            self.main_task = main_task
            prompt = DECOMPOSE_MAIN_TASK_PROMPT_TEMPLATE.format(main_task=main_task)
            response = self.chat.send_message([prompt, screenshot])

            subtask = parse_llm_response(response.text)
                    
            logger.info(f"This is the subtask created by the planning expert: {subtask}")

            self.current_subtask = subtask

            self._save_chat_history_to_file()

            return self.current_subtask
    
        except Exception as e:
            logger.error(f"Error in decompose_main_task() of planning_expert: {e}")
            raise
    
    def is_main_task_done(self, screenshot) -> bool:
        """
        Determines if the main task is complete by querying a language model (LLM).

        This function queries the LLM to assess if the entire `main_task` has reached completion,
        following the execution of `self.current_subtask`. It constructs a prompt using the
        `IS_LAST_TASK_PROMPT_TEMPLATE`, providing context with the `current_subtask`, `main_task`,
        and a `screenshot` of the current GUI state. The LLM is expected to respond with a
        definitive 'yes' (the main task is completed) or 'no' (there is still more work to do).

        Args:
            self: The instance of the class, expected to have `current_subtask`, `main_task`,
                  and a `chat` object for LLM interaction.
            screenshot: The image of the current GUI environment, providing visual context to the LLM.

        Returns:
            bool: True if the LLM responds 'yes' (meaning this *is* the last task and no more work is needed),
                  False if the LLM responds 'no' (meaning there's *more* work to do)
        """
        try:
            prompt = IS_LAST_TASK_PROMPT_TEMPLATE.format(current_subtask=self.current_subtask, main_task = self.main_task)
            response = self.chat.send_message([prompt, screenshot])
            
            llm_response_text = parse_llm_response(response.text)

            self._save_chat_history_to_file()

            return llm_response_text.lower() == 'yes'
        
        except Exception as e:
            logger.error(f"Error in is_main_task_done() of planning_expert: {e}")
            raise

    def rethink_subtask(self, reflection_expert_feedback: str, screenshot) -> None:
        """
        Revises the current planning of subtasks based on feedback and the current GUI state.

        This function is invoked when a re-evaluation of the subtask plan is necessary,
        for instance, due to an execution error, or after the successful completion of a previous subtask. 
        It constructs a prompt for a language model (LLM), providing it with `reflection_expert_feedback`, 
        the `self.current_subtask` just processed, the overall `self.main_task`, and the current `screenshot`.
        The LLM is tasked with generating a new, updated subtask based on this comprehensive context.

        The  newly generated subtask is then set as `self.current_subtask`, effectively updating the active task for the agent.

        Args:
            self: The instance of the class, providing access to `chat` for LLM interaction,
                  and `main_task` and `current_subtask` for context.
            reflection_expert_feedback (str): Feedback from the reflection expert. This
                                              string can be empty if the previous instruction
                                              list was completed successfully, or it can
                                              contain details about issues encountered.
            screenshot: The current image of the GUI environment, providing visual context
                        to the LLM for rethinking the plan.

        Returns:
            str: The new subtask which becomes the new `self.current_subtask`.
        """
        try:
            prompt = RETHINK_SUBTASK_PROMPT_TEMPLATE.format(
                reflection_expert_feedback=reflection_expert_feedback,
                current_subtask=self.current_subtask,
                main_task=self.main_task
            )
            response = self.chat.send_message([prompt, screenshot])

            subtask = parse_llm_response(response.text)

            self.current_subtask = subtask

            self._save_chat_history_to_file()

            return subtask


        except Exception as e:
            logger.error(f"Error in rethink_subtask() of planning_expert: {e}")
            raise
    
    def decompose_subtask(self, screenshot) -> str:
        """
        Decomposes the current active subtask into a detailed list of instructions/steps.

        This function expands a higher-level subtask into a granular sequence of executable
        instructions. It communicates with a language model (LLM), providing it with the
        `self.current_subtask` and a `screenshot` of the current GUI state as context.
        The LLM is expected to generate a detailed plan of actions, formatted as a
        semicolon-separated string of instructions. The function then parses this raw
        response by splitting it into individual instructions, forming a list of actionable
        steps suitable for an action execution expert.

        Args:
            self: The instance of the class, providing access to the `chat` object for
                  LLM interaction and the `current_subtask` which needs decomposition.
            screenshot: The current image of the GUI environment, providing visual
                        and contextual information to the LLM for accurate subtask decomposition.

        Returns:
            list[str]: A list of strings, where each string is a distinct LLM-generated
                       instruction for carrying out the subtask.
        """
        try:
            

            prompt = DECOMPOSE_SUBTASK_PROMPT_TEMPLATE.format(
                current_subtask=self.current_subtask,
            )

            response = self.chat.send_message([prompt, screenshot])

            prompt = DECOMPOSE_SUB_TASK_PROMPT_TEMPLATE_REFLECT
            response = self.chat.send_message([prompt, screenshot])

            instruction_list_str = parse_llm_response(response.text)
            instruction_list = [task.strip() for task in instruction_list_str.split(';') if task.strip()]
            logger.info(f"These are the instructions for the task: {instruction_list}")

            self._save_chat_history_to_file()
            
            return instruction_list
        
        except Exception as e:
            logger.error(f"Error in decompose_subtask() of planning_expert: {e}")
            raise
    

if __name__ == "__main__":
    pass