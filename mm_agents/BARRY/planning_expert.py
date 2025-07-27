import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from .utils import parse_llm_response


logger = logging.getLogger("planning_expert")

DECOMPOSE_MAIN_TASK_PROMPT_TEMPLATE = """
This is the main task: "{main_task}"
Decompose it into a list of actionable subtasks. Each subtask must involve a clear action (e.g., 'Open the browser' instead of 'Locate the browser icon'). 
Subtasks must not be conditional (e.g., do not say 'If the window is not maximized, maximize it'). 
Instead, analyze the screenshot to make definitive decisions (e.g., if the window is not maximized, include 'Maximize the window' as a subtask; if it is maximized, do not include it). 
Subtasks should not be overly granular, as detailed instructions will be provided later. Avoid subtasks that involve taking screenshots, locating elements, or recording coordinates, as the agent has screen markers for execution. 
Prefer keyboard shortcuts or hotkeys (e.g., Ctrl+T for a new tab) to improve reliability, especially for actions where visual recognition might fail (e.g., empty bookmark bars)(you could make it visible by pressing ctrl + shift + B). 
Identify the active application or window in the screenshot to ensure subtasks align with the current context (e.g., browser, file explorer). 
Do not include a final subtask like 'Finish the task'; each subtask must be meaningful.

Reasoning Process:
1. Analyze the screenshot to identify the active application, visible elements (e.g., browser tabs, search bars, or file explorer), and window state (maximized or not).
2. Break down the main task into logical and small goals.
3. Make definitive decisions based on the screenshot (e.g., if the window is not maximized, include a maximize subtask).
4. Consider using keyboard shortcuts or hotkeys to avoid reliance on visual elements that may be hard to recognize.
5. Ensure subtasks are contextually appropriate based on the screenshot (e.g., if a browser is open, focus on browser-related actions).

Here's how I want you to structure your response:
1.  **Reasoning Process:** Write down your thought process here.
2.  **Final Subtasks:** After your reasoning, you MUST provide the final list of subtasks. This list MUST start with the exact phrase "RESPONSE:" on its own line, followed immediately by the subtasks.
    All text from "RESPONSE:" until the end of your response will be considered the list of subtasks. Each subtask should be separated by a semicolon ';'.

Example of how the subtask list should appear:
RESPONSE: Open the browser; Search for "example" on Google; Do something.
"""

RETHINK_SUBTASK_PROMPT_TEMPLATE = """
Re-evaluate the subtask list for the main task: "{main_task}".
Do not repeat approaches that failed, as indicated by this feedback (if any): {reflection_expert_feedback}.
If the current subtask "{current_subtask}" was completed successfully, determine the remaining steps.
Analyze the screenshot to identify the active application, visible elements (e.g., browser tabs, search bars), and current state.
Consider alternative workflows or keyboard shortcuts (e.g., Ctrl+T for a new tab) to achieve the task reliably, especially if visual recognition is challenging.
If the feedback indicates an issue, propose a new approach avoiding the failed steps. If no feedback is provided, continue from the current state.
Do not include subtasks for screenshots, locating elements, or recording coordinates, as the agent has screen markers.
Ensure subtasks align with the current application context shown in the screenshot.

Reasoning Process:
1. Review the feedback (if any) to identify what went wrong or what subtask was completed.
2. Analyze the screenshot to determine the active application, window state, and visible elements.
3. If feedback indicates failure, devise an alternative approach (e.g., use a different application, a different path to get to the same point or use hotkeys).
4. If the subtask was completed, identify the next logical steps to complete the main task.
5. Ensure subtasks are actionable, contextually relevant, and leverage hotkeys where possible.


Here's how I want you to structure your response:
1.  **Reasoning Process:**  Write down your thought process here.
2.  **Revised Subtask List:** After your reasoning, you MUST provide the revised list of remaining subtasks. This list MUST start with the exact phrase "RESPONSE:" on its own line, followed immediately by the subtasks. All text from "RESPONSE:" until the end of your response will be considered the revised list of subtasks. Each subtask should be separated by a semicolon ';'.

Example of how the revised subtask list should appear:
RESPONSE: Select the "Images" tab; Choose a dog image; Download the image
"""

DECOMPOSE_SUBTASK_PROMPT_TEMPLATE = """
Decompose the subtask "{current_subtask}" into detailed, actionable instructions. Analyze the screenshot to identify the active application, visible elements (e.g., address bar, search bar, buttons), and window state. Prefer keyboard shortcuts or hotkeys (e.g., Ctrl+T to open a new tab, Ctrl+A to select text) to ensure reliability, especially for elements without clear text or buttons. Combine related actions (e.g., click, select text with Ctrl+A, and type) into a single instruction where appropriate. Avoid instructions for screenshots, locating elements, or recording coordinates, as the agent has screen markers. If an element is ambiguous (e.g., multiple search bars), specify which one (e.g., 'the browser's address bar').

Reasoning Process:
1. Analyze the screenshot to identify the active application and relevant UI elements.
2. Break down the subtask into a sequence of executable instructions, prioritizing hotkeys for reliability.
3. Combine actions where logical (e.g., clicking and typing in a search bar).
4. Ensure clarity by specifying ambiguous elements (e.g., 'the browser's address bar' vs. 'the system's search bar').

These steps do not need to be overly precise if the environment details are not fully known yet.
However think that this steps will have to be translated into pyAutoGUI actions so it is not necessary to say
move the mouse to th icon. As in pyAutoGUI you give the coordinates when you click.
Please provide the decomposed steps as a clear list or sequence. 


Here's how I want you to structure your response:
1.  **Reasoning Process:** Write down your thought process here.
2.  **Revised Instruction List:** After your reasoning, you MUST provide the revised list of the instruction list. This list MUST start with the exact phrase "RESPONSE:" on its own line, followed immediately by the instructions. 
All text from "RESPONSE:" until the end of your response will be considered the revised list of instructions. Each instruction should be separated by a semicolon ';'.

Example of how the revised subtask list should appear:
RESPONSE: Click on the browser icon; Click on the search bar; Type dogs.
"""

IS_LAST_TASK_PROMPT_TEMPLATE = """
The subtask "{current_subtask}" was completed successfully.
The main task is: "{main_task}".
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
        self.last_task_for_test = "" # this is for the tests

        self.main_task = ""
        self.current_subtask = ""

        self.first_iter = True    
    
    def decompose_main_task(self, main_task, screenshot):
        """
        Receives the main task and uses an LLM to generate a list of subtasks.

        This function initiates the planning phase by instructing a language model (LLM)
        to break down a high-level `main_task` into smaller, manageable subtasks.
        The LLM is specifically prompted to provide these subtasks separated by semicolons.
        The `main_task` and the screenshot representing the current GUI
        state are sent to the LLM to provide necessary context for decomposition.
        The LLM's raw text response is then parsed into a list of individual subtasks.
        The first generated subtask is stored as `self.current_subtask` and is also
        returned as the initial subtask for subsequent action or execution.

        Args:
            main_task (str): The primary objective or high-level goal that needs to be decomposed.
            screenshot: screenshot of the current GUI environment. This provides visual and 
            interactive context to the LLM during the task decomposition process.

        Returns:
            str: The first subtask from the newly generated list of subtasks.
        """

        try:
            self.main_task = main_task
            prompt = DECOMPOSE_MAIN_TASK_PROMPT_TEMPLATE.format(main_task=main_task)
            response = self.chat.send_message([prompt, screenshot])

            subtasks_raw_string = parse_llm_response(response.text)
            
            # Split by semicolon and clean each individual task.
            subtask_list = [task.strip() for task in subtasks_raw_string.split(';') if task.strip()]
        
            logger.info(f"These are the subtasks created by the planning expert: {subtask_list}")

            self.current_subtask = subtask_list[0]
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

            return llm_response_text[0].lower() == 'yes'
        
        except Exception as e:
            logger.error(f"Error in is_main_task_done() of planning_expert: {e}")
            raise

    def rethink_subtask_list(self, reflection_expert_feedback: str, screenshot) -> None:
        """
        Revises the current planning of subtasks based on feedback and the current GUI state.

        This function is invoked when a re-evaluation of the subtask plan is necessary,
        for instance, due to an execution error, or after the successful completion of a previous subtask. 
        It constructs a prompt for a language model (LLM), providing it with `reflection_expert_feedback`, 
        the `self.current_subtask` just processed, the overall `self.main_task`, and the current `screenshot`.
        The LLM is tasked with generating a new, updated sequence of subtasks based on this comprehensive context.

        The first subtask from this newly generated list is then set as
        `self.current_subtask`, effectively updating the active task for the agent.

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
            str: The first subtask from the newly generated list of subtasks, which
                 becomes the new `self.current_subtask`.
        """
        try:
            prompt = RETHINK_SUBTASK_PROMPT_TEMPLATE.format(
                reflection_expert_feedback=reflection_expert_feedback,
                current_subtask=self.current_subtask,
                main_task=self.main_task
            )
            response = self.chat.send_message([prompt, screenshot])

            subtask_list_str = parse_llm_response(response.text)
            # Split by semicolon and clean each individual task.
            subtask_list = [task.strip() for task in subtask_list_str.split(';') if task.strip()]

            self.last_task_for_test = subtask_list[-1] # esto es solo para facilitar la prueba de casos en los test
            self.current_subtask = subtask_list[0]

            return self.current_subtask


        except Exception as e:
            logger.error(f"Error in rethink_subtask_list() of planning_expert: {e}")
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

            instruction_list_str = parse_llm_response(response.text)
            instruction_list = [task.strip() for task in instruction_list_str.split(';') if task.strip()]
            logger.info(f"These are the instructions for the task: {instruction_list}")
            
            return instruction_list
        
        except Exception as e:
            logger.error(f"Error in decompose_subtask() of planning_expert: {e}")
            raise
    
    def _set_current_task_as_last(self):
        self.current_subtask = self.last_task_for_test
    
    def _print_history(self):
        """
        Prints new messages from the chat history to the console since the last print.

        This internal helper function iterates through the `chat.history` list, starting
        from the `last_printed_index`. It prints each new message, formatted to show
        its role (e.g., 'user', 'model') and the text content. After printing, it updates
        `last_printed_index` to the current total number of messages in the history,
        ensuring that only new messages are printed in subsequent calls.

        Args:
            self: The instance of the class, expected to have a `chat` object with a
                `history` attribute (a list of message objects) and a `last_printed_index`
                attribute (an integer tracking the last printed message's index).

        Returns:
            None: This function performs side effects by printing to standard output
                and modifying `self.last_printed_index`.
        """
        for i in range(self.last_printed_index, len(self.chat.history)):
            message = self.chat.history[i]
            text_content = message.parts[0].text 
            print(f"{message.role}: {{ \"{text_content}\" }}")
        
        self.last_printed_index = len(self.chat.history)

    

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__) # Obtiene el directorio del script actual
    image_folder = os.path.join(current_dir, 'mocks')

    planning_expert = PlanningExpert()
    main_task = "descarga una foto de un perro"

    # caso 1 --------------------------------
    image_name = 'som_screenshot_1.png'
    image_path = os.path.join(image_folder, image_name)
    SOM = Image.open(image_path)

    print("caso 1, main task:", main_task)
    subtask = planning_expert.decompose_main_task(main_task, SOM)
    instruction_list = planning_expert.decompose_subtask(SOM)

    planning_expert._print_history()
    print("esta es la primera subtarea: " + subtask + "\n")
    print("esta es la lista de instrucciones: ", instruction_list)



    print("caso 1 terminado -----------------------------------------------\n")

    # caso 2.1 --------------------------------
    image_name = 'som_screenshot_2.png'
    image_path = os.path.join(image_folder, image_name)
    SOM = Image.open(image_path)

    print("caso 2.1: tarea no está acabada")
    done = planning_expert.is_main_task_done(SOM)
    if done:
        print("return {'done': True}")
    else:
        subtask = planning_expert.rethink_subtask_list("", SOM)
    
    instruction_list = planning_expert.decompose_subtask(SOM)
    
    planning_expert._print_history()


    print("caso 2.1 terminado -----------------------------------------------\n")

    
    # caso 3 --------------------------------

    image_name = 'som_screenshot_3.png'
    image_path = os.path.join(image_folder, image_name)
    SOM = Image.open(image_path)

    reflection_expert_feedback = """
    the task it is not done, it has click on video section instead of the image section. 
    It should press the image section and then download a photo.
    """

    planning_expert._set_current_task_as_last()

    print("caso 3: Ha acabado la lista pero el reflection expert dice que no está bien acabada")

    subtask = planning_expert.rethink_subtask_list(reflection_expert_feedback, SOM)

    planning_expert._print_history()

    print("caso 3 terminado -----------------------------------------------\n")

    # caso 2.2 --------------------------------

    print("caso 2.2: tarea sí está acabada")
    planning_expert._set_current_task_as_last()
    done = planning_expert.is_main_task_done(SOM)
    if done:
        print("return {'done': True}")
    else:
        subtask = planning_expert.rethink_subtask_list("", SOM)
    
    planning_expert._print_history()
    
    print("caso 2.2 terminado -----------------------------------------------\n")


