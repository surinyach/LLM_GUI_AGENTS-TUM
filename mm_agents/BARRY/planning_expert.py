import os
import logging
from typing import List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import re

logger = logging.getLogger("planning_expert")

DECOMPOSE_MAIN_TASK_PROMPT_TEMPLATE = """
This is the main task: "{main_task}"
Decompose it into subtasks. Don't do very fine grained subtask as later 
you will have to think about the instructions for each subtask. It is important that each subtask involves at least an action.
For example a subtask can't be just to look for a google chrome icon. The subtask should be "Open the browser". 
Then when you have to give the instructions for the SUBTASK you can say click on the browser icon or type crome on the searching bar or something like that.
Before every instruction, the agent who has to execute them, has a set of mark of the screen so avoid doing subtask or instructions
that consist on doing a screenshot, locating an element or recording coordinates. Don't add a last subtask that says finish the task.
It has to be a meaningful subtask.
The agent can press a combination of keys at the same time so don't put them in different subtask. The same with clicking and typing. 
If the window is not fully open add maximizing the window in the first task. If the window is already maximized DO NOT maximize it again and DO NOT MAKE SHURE to open it, do it ONLY if is way smaller than the full window.
Take into account that llms are not very good finding difficult regions that doesn't have text or buttons like an empty bookmark bar
DO NOT make instructions to release the click!!! it is not necessary, the action expert is not able to solve this type of instructions.


Here's how I want you to structure your response:
1.  **Reasoning Process:** First, think step-by-step about how to break down the main task. Consider the actions involved and ensure each subtask is actionable but not overly granular. Write down your thought process here.
2.  **Final Subtasks:** After your reasoning, you MUST provide the final list of subtasks. This list MUST start with the exact phrase "RESPONSE:" on its own line, followed immediately by the subtasks.
    All text from "RESPONSE:" until the end of your response will be considered the list of subtasks. Each subtask should be separated by a semicolon ';'.

Example of how the subtask list should appear:
RESPONSE: Open the browser; Search for "example" on Google; Do something; Finish task.
"""

RETHINK_SUBTASK_PROMPT_TEMPLATE = """
Rethink the subtask list. You MUST FORGET the previous subtask list you made. This re-evaluation is necessary either because:

A) There's an issue with the current approach, as highlighted by this feedback:
{reflection_expert_feedback}

OR

B) I have successfully finished the current subtask: "{current_subtask}", and I need to know the remaining steps to complete the main task.

Take into account the above reason (A or B), my past actions and messages, and the current state of my screen. 
Think if the current task solved solves the main task and there is no need to do more tasks.
If there was a problem (you can know this if there is feedback) think about alternative ways the main task could be solved. DON'T TRY TO DO THE SAME STEPS!!!** Consider different perspectives, workflows, or how similar problems are handled in other environments.
This is vital to avoid retrying solutions that have repeatedly failed. You should trust the systems more than you trust you so if the system says something is visible it is true. Take into account that you are not good reconizing things so if there is a hot key combination which would do it easier or faster use that option.
If you don't know what to do try a different approach or a different way. Re do the subtasks for the main task with the new approach.
Now, provide a revised subtask list with what is still left to do to accomplish the main task: {main_task}.

Here's how I want you to structure your response:
1.  **Reasoning Process:** First, analyze the provided feedback (if any), your past actions, and the current screen state. Determine whether the re-evaluation is due to an issue or a completed subtask. Think step-by-step about why the subtask list needs rethinking (if an issue was raised, considering alternatives), or what the next logical steps are (if the current subtask is finished). Based on this, formulate the revised list of remaining subtasks. Write down your comprehensive thought process here.
2.  **Revised Subtask List:** After your reasoning, you MUST provide the revised list of remaining subtasks. This list MUST start with the exact phrase "RESPONSE:" on its own line, followed immediately by the subtasks. All text from "RESPONSE:" until the end of your response will be considered the revised list of subtasks. Each subtask should be separated by a semicolon ';'.

Example of how the revised subtask list should appear:
RESPONSE: Select the "Images" tab; Choose a dog image; Download the image
"""

DECOMPOSE_SUBTASK_PROMPT_TEMPLATE = """
Taking into account any feedback provided in the previous message (if there was)
and a summary of actions already performed (if described in previous messages),
decompose the following specific subtask into a series of detailed steps or instructions.
Before every instruction, the agent who has to execute them, has a set of mark of the screen so avoid doing instructions
that consist on doing a screenshot, locating an element or recording coordinates.
If some steps can be done together like clicking, selecting the text with ctrl + A and typing put it all in the same instruction.
The answer should only contain the steps, don't add any comments.
{current_subtask}.

These steps do not need to be overly precise if the environment details are not fully known yet.
However think that this steps will have to be translated into pyAutoGUI actions so it is not necessary to say
move the mouse to th icon. As in pyAutoGUI you give the coordinates when you click.
Please provide the decomposed steps as a clear list or sequence. 

If there is something that could be ambiguous specifiy it. Like if there are 2 searchbars specify which one.

Here's how I want you to structure your response:
1.  **Reasoning Process:** First, analyze the provided feedback, your past actions, and the current screen state. 
Think step-by-step about why the instruction list needs rethinking (if an issue was raised) or what the next logical steps are. 
Based on this, formulate the instruction list for the current subtask. Write down your thought process here.
2.  **Revised Instruction List:** After your reasoning, you MUST provide the revised list of the instruction list. This list MUST start with the exact phrase "RESPONSE:" on its own line, followed immediately by the instructions. 
All text from "RESPONSE:" until the end of your response will be considered the revised list of instructions. Each instruction should be separated by a semicolon ';'.

Example of how the revised subtask list should appear:
RESPONSE: Click on the browser icon; Click on the search bar; Type dogs.
"""

IS_LAST_TASK_PROMPT_TEMPLATE = """
I correctly finished this subtask: {current_subtask}. This is the main task: {main_task}
Taking into account the subtask list you gave me is the main task done?
Take into account your last task decomposition into subtask. Respond only with 'yes' or 'no'.

Here's how I want you to structure your response:
1.  **Reasoning Process:** First, analyze the provided feedback, your past actions, and the current screen state. 
Think step-by-step about why the instruction list needs rethinking (if an issue was raised) or what the next logical steps are. 
Based on this, formulate the instruction list for the current subtask. Write down your thought process here.
2.  **Final answer:** After your reasoning, you MUST respond with 'RESPONSE:' followed only with a 'yes' or 'no'
Example: 
RESPONSE:yes

"""


class PlanningExpert:
    def __init__(self, model_id: str = "gemini-2.0-flash"):
        """
        Inicializa el experto en reflection.
        """
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY no encontrada en el archivo .env para ActionExpert")
            raise ValueError("GEMINI_API_KEY no encontrada en el archivo .env")
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_id)

        self.chat = self.model.start_chat(history=[])
        self.last_printed_index = 0 # this is for printing the chat history for debugging
        self.last_task_for_test = "" # this is for the tests

        self.main_task = ""
        self.current_subtask = ""
        # self.instruction_list = ""
        # self.subtask_list = []
        # self.current_subtask_index = 0

        self.first_iter = True


    def _parse_subtask_response(self, response_text: str, marker) -> List[str]:
        """
        Parses the raw text response from the LLM, by splitting on "SUBTASK_LIST:"
        and taking the latter part. Assumes subtasks are separated by ';'.

        Args:
            response_text (str): The full string response from the LLM.

        Returns:
            List[str]: A list containing each parsed subtask.
                       Returns an empty list if the marker is not found or no subtasks are present.
        """
        if not response_text:
            return []
        print("text to parse: ", response_text)
        parts = response_text.split("RESPONSE:", 1) # Use 1 to split only on the first occurrence

        if len(parts) < 2:
            print(f"Warning: Marker '{marker}' not found in the response.")
            return []

        # Grab the part after the marker and strip leading/trailing whitespace (including newlines)
        subtasks_raw_string = parts[1].strip()

        # Split by semicolon and clean each individual task.
        subtasks = [task.strip() for task in subtasks_raw_string.split(';') if task.strip()]
        return subtasks

    
    
    def decompose_main_task(self, main_task, SOM):
        """
        Receives the main task and uses an LLM to generate a list of subtasks.

        The LLM is prompted to decompose the main task into individual subtasks,
        separated by semicolons. This function then parses the LLM's response
        and stores the resulting subtasks.

        Args:
            main_task (str): The primary objective of the osworld test, provided as a string.

        Returns:
            str: The first subtask from the newly generated list of subtasks.
        """
        logger.info("Planning expert saves and decomposes main task.")

        try:
            self.main_task = main_task
            prompt = DECOMPOSE_MAIN_TASK_PROMPT_TEMPLATE.format(main_task=main_task)
            response = self.chat.send_message([prompt, SOM])
            marker = "SUBTASK_LIST:"
            subtask_list = self._parse_subtask_response(response.text, marker)
            logger.info(f"These are the subtasks created: {subtask_list}") 
            
            self.current_subtask = subtask_list[0]
            return self.current_subtask
    
        except Exception as e:
            logger.error(f"Error in decompose_main_task() of planning_expert: {e}")
            raise
    
    def main_task_done(self, SOM) -> bool:
        """
        Determines if the main task is complete by querying an LLM.

        This function informs the LLM that the current subtask has been completed
        and asks if any further work is needed. It specifically expects a 'yes' or 'no'
        response from the LLM.

        Returns:
            bool: True if the LLM responds 'yes' (meaning there's *more* work to do),
                  False if the LLM responds 'no' (meaning this *is* the last task).
        """
        logger.info("Querying LLM to check if this was the last task.")

        try:
            prompt = IS_LAST_TASK_PROMPT_TEMPLATE.format(current_subtask=self.current_subtask, main_task = self.main_task)
            response = self.chat.send_message([prompt, SOM])
            
            # Clean the LLM's response (remove whitespace, convert to lowercase)
            llm_response_text = self._parse_subtask_response(response.text, "RESPONSE:")

            logger.info(f"LLM responded to 'is_last_task' with: '{llm_response_text}'")
            return llm_response_text[0] == 'yes'
        
        except Exception as e:
            logger.error(f"Error in is_last_task() of planning_expert: {e}")
            raise

    def rethink_subtask_list(self, reflection_expert_feedback: str, SOM) -> None:
        """
        Revises the current list of subtasks based on feedback and current progress.

        This function is invoked when a re-evaluation of the subtask plan is necessary,
        either due to an execution error, incomplete reflection, or successful completion
        of a subtask. It communicates with an LLM, providing it with feedback,
        the completed subtask (if any), past actions, the overall main task,
        and the current screen state (SOM). The LLM generates a new, updated
        list of subtasks.

        Args:
            reflection_expert_feedback (str): Feedback from the reflection expert, indicating any issues.
                                              It can be an empty string if the instruction list was correctly complete.
            action_expert_feedback (str): A summary of the actions taken during the execution of the previous instruction list.
            SOM (any): The Set of mark (SOM) or current screen representation, providing context to the LLM.

        Returns:
            None: This function updates the `self.subtask_list` in place.
        """
        logger.info("Initiating subtask list re-evaluation.")
        try:
            prompt = RETHINK_SUBTASK_PROMPT_TEMPLATE.format(
                reflection_expert_feedback=reflection_expert_feedback,
                current_subtask=self.current_subtask,
                main_task=self.main_task
            )
            response = self.chat.send_message([prompt, SOM])
            marker = "SUBTASK_LIST:"
            subtask_list = self._parse_subtask_response(response.text, marker )
            self.last_task_for_test = subtask_list[-1] # esto es solo para facilitar la prueba de casos en los test
            self.current_subtask = subtask_list[0]
            logger.info(f"New subtask list received from LLM: {subtask_list}\n")

            return self.current_subtask


        except Exception as e:
            logger.error(f"Error in rethink_subtask_list() of planning_expert: {e}")
            raise
    
    def decompose_subtask(self, SOM) -> str:
        """
        Decomposes the current active subtask into a detailed list of instructions/steps.

        This function interacts with an LLM to break down the subtask, considering
        any prior feedback, completed actions, the overall main task, and the
        current screen state (SOM). It's designed to generate actionable steps
        for the action expert.

        Args:
            SOM (any): The State of Mind (SOM) or current screen representation,
                       providing visual and contextual information to the LLM.

        Returns:
            str: A string containing the LLM-generated instructions for the subtask.
        """
        logger.info("Decomposing the current subtask into instructions.")
        try:
            

            prompt = DECOMPOSE_SUBTASK_PROMPT_TEMPLATE.format(
                current_subtask=self.current_subtask,
            )

            response = self.chat.send_message([prompt, SOM])
            logger.info(f"Instructions created by LLM: {response.text}")
            marker = "INSTRUCTION_LIST:"
            instruction_list = self._parse_subtask_response(response.text, marker)
            
            return instruction_list
        
        except Exception as e:
            logger.error(f"Error in decompose_subtask() of planning_expert: {e}")
            raise
    
    def _set_current_task_as_last(self):
        self.current_subtask = self.last_task_for_test
    
    def _print_history(self):
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
    done = planning_expert.main_task_done(SOM)
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
    done = planning_expert.main_task_done(SOM)
    if done:
        print("return {'done': True}")
    else:
        subtask = planning_expert.rethink_subtask_list("", SOM)
    
    planning_expert._print_history()
    
    print("caso 2.2 terminado -----------------------------------------------\n")


