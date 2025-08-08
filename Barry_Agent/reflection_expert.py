import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from .utils import parse_llm_response
from datetime import datetime


logger = logging.getLogger("reflection_expert")

FIRST_EVALUATE_EXECUTION_PROMPT = """
Instruction: {instruction}

Task:
Identify active application, window state, and visible UI elements (e.g., search bars, tabs, buttons) from the screenshot.
Determine critical UI elements relevant to the instruction's execution. Include elements even if indirectly related.
Describe the *expected state* of these elements after successful execution, reflecting the instruction's intended outcome.
For hotkey actions (e.g., Ctrl+Shift+B), assume success even if not visually distinct.

Output:
Critical elements to inspect and their expected state after successful instruction execution.
Additionally, if the screenshot suggests the screen is still loading (e.g., presence of a loading spinner, partial content, "Loading..." text, or a blank/white screen when content is expected), clearly state this.
"""

SECOND_EVALUATE_EXECUTION_PROMPT = """
Task:
From the previously identified elements, select the most important ones relevant to the instruction's goal (e.g., address bar for typing, new tab for Ctrl+T).
Analyze the screenshot to determine their current state.
Pay special attention to drag bars (sliders), verifying their exact position matches the expected state (e.g., if expected at maximum, ensure it's at the far right/top).
Trust visible elements in the screenshot (e.g., buttons, tabs). Assume hotkey actions (e.g., Ctrl+Shift+B) succeeded if not visually distinct.
If ambiguous (e.g., multiple search bars), specify (e.g., 'browser’s address bar').

Output:
Detailed description of the most important elements' current state and reasoning on whether it matches the expected state.

"""

THIRD_EVALUATE_EXECUTION_PROMPT = """
Task:
Determine if the instruction was successfully executed, based on the previous analysis, expected element states, and the screenshot.
Evaluate success by checking for changes or absence of elements the instruction aimed to alter, as the screenshot shows the *after* state.
Rely on screenshot for visible changes (e.g., new folder, closed pop-up). Assume hotkey actions (e.g., Ctrl+T) succeeded.

Output Structure:
1. **Reasoning Process:** Your thought process.
2. **Final Answer:** Respond 'yes' or 'no' on whether the instruction was successfully completed. Your response MUST start with 'RESPONSE:' on a new line.
"""

EVALUATE_ERROR_PROMPT = """
Task:
Classify the last error as 'Minor' or 'Major' based on the definitions below, considering the current instruction: {instruction}.
Additionally, if the screenshot suggests the screen is still loading (e.g., presence of a loading spinner, partial content, "Loading..." text, or a blank/white screen when content is expected), clearly state this.

Error Classification:

* *Minor:* The instruction can be completed from the current screen with only one additional instruction. If more than one instruction is needed, it's a Major error.
    * Examples:
        * Clicked the wrong button but the correct one is still visible (one instruction to click the right one).
        * Typed the wrong word but can delete and retype (one instruction to correct).
        * Couldn't click exact coordinates for a slider but can aim better from current cursor position.
        * Needed to scroll to find a link that wasn't immediately visible.
    * Note: You have 2 chances to resolve the same Minor error. If it persists, it becomes a Major error.

* *Major:* The instruction cannot be completed from the current screen.
    * Examples:
        * Clicked a button, and the wrong page appeared (cannot revert or continue in one step).
        * Tried to type on a search bar, but no search bar exists on the screen.
        * Tried to click a non-existent icon.

Output Structure:
1.  *Reasoning Process:* Step-by-step thoughts on error classification and potential solutions. Analyze the screenshot to inform your thoughts. Consider the main task and if scrolling is needed.
2.  *Revised Instruction:* What caused the error and a solution to resolve it. The response MUST start with 'RESPONSE:' on a new line, followed by 'Minor:' or 'Major:' and your solution.

Example of expected output:
RESPONSE: Minor: You should click slightly more to the right.
"""


class ReflectionExpert:
    def __init__(self, model_id: str = "gemini-2.0-flash"):
        """
        Initialitation of the Reflexion Expert
        """
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY not found in the file .env for Reflection Expert")
            raise ValueError("GEMINI_API_KEY not found in the file .env")
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_id)

        self.chat = self.model.start_chat(history=[])

        self.instruction_list = []
        self.instruction_index = 0
        
        self.last_printed_index = 0 # this is for printing the chat history for debugging

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
                    log_entry = f"[{timestamp}] {message.role} REFLECTION EXPERT: {text_content}\n"
                    f.write(log_entry)

                self.last_printed_index = len(self.chat.history)

        except Exception as e:
            logger.error(f"Error saving chat history to file: {e}")
            raise
    

    def set_subtask_and_instructions(self, instruction_list) -> None:
        """
        Saves the instruction_list and adds the current subtask to the chat history.

        This function serves to log the detailed plan (subtask and its specific
        instructions) within the ongoing conversation history. This ensures that
        the LLM (and anyone reviewing the chat history) has a clear record of
        the plan that was put into action. It also initializes the instruction
        index to zero, preparing for the sequential execution of the instructions.

        Args:
            subtask (str): The specific subtask that has been set for execution.
            instruction_list (list): The detailed list of instructions generated
                                     for carrying out the given subtask. This is
                                     expected to be a list of strings, where each
                                     string is a distinct instruction.

        Returns:
            None: This function directly modifies the `chat.history` by appending
                  a user message, and it updates `self.instruction_list` and
                  `self.instruction_index`.
        """

        self.instruction_list = instruction_list
        self.instruction_index = 0

    

    def evaluate_execution(self, screenshot):
        """
        Evaluates the success of an executed instruction by interacting with a language model
        (LLM) in a multi-step conversational process.

        Args:
            self: The instance of the class, expected to have a `chat` object for sending messages
                and an `instruction_list` with the `instruction_index` pointing to the current instruction.
            screenshot: The image representing the state of the GUI after the instruction was executed.

        Returns:
            bool: True if the LLM determines the instruction was successfully completed ('yes'),
                False otherwise ('no').
        """
        try:
            prompt = FIRST_EVALUATE_EXECUTION_PROMPT.format(instruction = self.instruction_list[self.instruction_index])
            response = self.chat.send_message([prompt, screenshot])
            response = self.chat.send_message(SECOND_EVALUATE_EXECUTION_PROMPT)
            response = self.chat.send_message(THIRD_EVALUATE_EXECUTION_PROMPT)

            final_response = parse_llm_response(response.text)

            logger.info("Did the execution went well? " + final_response)

            self._save_chat_history_to_file()
            return final_response.lower() == 'yes'
        
        except Exception as e:
            logger.error(f"Error in evaluate_execution: {e}")
            raise
    
    def is_last_instruction(self) -> bool:
        """
        Checks if the current instruction being processed is the last one in the instruction list.

        This function is crucial for controlling the flow of instruction execution,
        allowing the system to determine when all planned steps have been completed.

        Args:
            self: The instance of the class, expected to have an `instruction_list`
                (a list of instructions) and an `instruction_index` (the index
                of the currently active instruction).

        Returns:
            bool: True if the `instruction_index` points to the last element of
                `instruction_list`, False otherwise.
        """
        return len(self.instruction_list) - 1 == self.instruction_index
    
    def create_new_instruction(self):
        """
        Generates a new, single instruction to address a minor error without altering the existing instruction list.

        This function is invoked when a small issue can be resolved with a one-off instruction.
        It prompts the language model (LLM) to generate a corrective instruction based on the
        last evaluation. Crucially, this new instruction is not added to the ongoing `instruction_list`,
        nor does it modify the sequence of existing instructions. The agent is expected to execute
        this generated instruction immediately and then resume its progression through the
        original `instruction_list` as if no error had occurred.

        Args:
            None as it has all the information in the chat history.

        Returns:
            str: The newly generated single instruction from the LLM.
        """
        prompt = "Taking into account the last evaluation, respond only with the next instruction. don't add any comments."
        response =self.chat.send_message(prompt)
        self._save_chat_history_to_file()

        return response.text

    
    def get_next_instruction(self):
        """
        Retrieves the next instruction from the instruction list and advances the instruction index.
        Always the function is_last_instruction() should be called before to prevent index out of bound errors.

        This function increments the internal `instruction_index` and then returns the
        instruction at this new index. It is used to get the next instruction to pass it to the action expert.

        Args:
            self: The instance of the class, expected to have an `instruction_list`
                (a list of instructions) and an `instruction_index` (the index
                of the currently active instruction).

        Returns:
            str: The instruction string at the newly incremented `instruction_index`.
        """
        self.instruction_index += 1
        return self.instruction_list[self.instruction_index]
    
    def evaluate_error(self, screenshot):
        """
        Evaluates a detected error by querying a language model (LLM) to classify it
        as minor or major and suggest solutions.

        This function sends the current instruction, and a screenshot to the LLM. 
        The LLM then provides a detailed response that includes a reasoning process, 
        identifies the cause of the error, and proposes solutions. The error
        classification (minor or major) is based on specific criteria defined in the
        `EVALUATE_ERROR_PROMPT`, primarily whether the error can be resolved with a
        single follow-up instruction.

        Args:
            self: The instance of the class, providing access to the `chat` object
                for LLM interaction and the `instruction_list` with the current `instruction_index`.
            screenshot: The image representing the current state of the GUI where the
                        error occurred.

        Returns:
            str: The LLM's response, stripped of the "RESPONSE:" prefix, which
                contains the error classification (e.g., "Minor: ...", "Major: ...")
                and proposed solutions.
        """
        try:

            prompt = EVALUATE_ERROR_PROMPT.format(instruction = self.instruction_list[self.instruction_index])
            response =self.chat.send_message([screenshot, prompt])

            logger.info("This is the response of the reflection expert: " + response.text)

            final_response = parse_llm_response(response.text)
            self._save_chat_history_to_file()
            return final_response
        
        except Exception as e:
            logger.error(f"Error in evaluate_error: {e}")
            raise
    
if __name__ == "__main__":
    pass