import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from .utils import parse_llm_response


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
RESPONSE: Open the browser; Search for "example" on Google; Do something.
"""

RETHINK_SUBTASK_PROMPT_TEMPLATE = """
Rethink the subtask list. You MUST FORGET the previous subtask list you made. This re-evaluation is necessary either because:

A) There's an issue with the current approach, as highlighted by this feedback:
{reflection_expert_feedback}

OR

B) I have successfully finished the current subtask: "{current_subtask}", and I need to know the remaining steps to complete the main task.

Take into account the above reason (A or B), my past actions and messages, and the current state of my screen. 
Think about the consequences of each task, for example if you close the tab and it is the only tab the window will close
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
Think if any task or instruction can be achieve by using a combination of hot keys. This is VERY IMPORTANT because this llm is bad at reconising elements that don't contain text so hotkeys are perfect
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


