import os
import logging
from typing import Dict, List, Tuple, NamedTuple
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from io import BytesIO
from .action_expert import ActionExpert
from .planning_expert import PlanningExpert
from .reflection_expert import ReflectionExpert
from .perception_expert import PerceptionExpert

from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

logging.basicConfig(
    level=logging.INFO,  # Set the minimum level for logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        # logging.FileHandler("app.log") # Uncomment to also log to a file
    ]
)

# Now, get the logger for this module
logger = logging.getLogger("desktopenv.agent")

class BarryAgent:
    def __init__(self, model: str = "gemini-2.0-flash", observation_type: str = "screenshot", action_space: str = "pyautogui"):
        """
        Initializes the agent with the configuration to interact with OSWorld and the Gemini API.
        """
        # Cargar variables de entorno desde .env
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY not found in the .env file")
            raise ValueError("GEMINI_API_KEY not found in the .env file")

        # Configurar la API de Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model)

        # Configurar parámetros del entorno
        self.observation_type = observation_type
        self.action_space = action_space
        
        # Historial para el agente
        self.trajectory_length = 0
        self.max_trajectory_length = 25
        self.call_user_count = 0
        self.call_user_tolerance = 3
        
        self.main_task = ""
        self.first_iteration = True

        self.action_expert = ActionExpert() 
        self.planning_expert = PlanningExpert()
        self.reflection_expert = ReflectionExpert()
        self.perception_expert = PerceptionExpert()
        
        self.graph = None
        self.graph_state = {} 

        # SOM screenshot and description
        self.screenshot = ""
        self.SOM_screenshot = ""
        self.SOM_description = ""

        self.sleep = False

        # LANG GRAPH

        # STATE --------------------------------------------------------------------

        class State(TypedDict):
            reflection_action: str
            reflection_planning: str

            osworld_action: List[str]
            done: bool
        

        graph_builder = StateGraph(State)

        # NODES -----------------------------------------------

        def start_router(state: State):
            #logger.info("I am in start_router")

            if self.first_iteration:
                #logger.info("I am going to the planning expert")
                return {"next": "planning_expert"}
                
            #logger.info("I am going to the reflection expert")
            return {"next": "reflection_expert"}
        
        def planning_expert(state: State):
            """
                case 1: This case only happens in the firs iteration:
                    In this case the main task is saved and decomposed in the planning expert.
                    The main task is decomposed in a list of strings. Each string is a subtask.
                    Then the planning expert returns the current subtask.

                case 2: Action experts finish the instruction_list and reflection_expert says it is correct:
                    First we ask the planning expert if this was the last task.
                    If it was the last done = true is returned.
                    If it wasn't the last task it tells the planning expert to rethink the rest of the subtask.
                    Rethinking the subtask list implies to delete the subtasks already made and generate a list that 
                    STRATS with the ones that are still needed to be done.
                    The planning expert returns the inmediate first task of the new subtask list.

                case 3: There is an error. Either execution error during the instruction list or because refelction expert don't think it is finished:
                    It only calls the function rethink_subtask_list() but with the reflection_expert_feedback and the action_expert_feedback.
                    It is the same function as the case 2. The planning expert creates a new subtask list and returns the first subtask.

                Common actions:
                In the past cases the planning expert always returns a subtask. So after every case this task must be decomposed into
                an instruction list. When we have the current subtask and instruction list we call reflection expert to save the subtask and instruction list.
                
            """
            logger.info("reflection planing: " + state["reflection_planning"])

            # case 1
            if self.first_iteration:
                subtask = self.planning_expert.decompose_main_task(self.main_task, self.screenshot)
                #loggerf"The planning expert decomposes the main task. This is the first subtask: {subtask}")

                self.first_iteration = False

            # case 2
            elif state["reflection_planning"] == "finish":
                done = self.planning_expert.is_main_task_done(self.screenshot)
                logger.info("is main task done?:")
                logger.info(done)

                if done:
                    return {"done": True}
                else:
                   subtask = self.planning_expert.rethink_subtask_list("", self.screenshot)
                   
            # case 3
            else:
                subtask = self.planning_expert.rethink_subtask_list(state["reflection_planning"], self.screenshot)

            # This has to be done after every case:
            instruction_list = self.planning_expert.decompose_subtask(self.screenshot)
            #loggerf"These are the instructions for the task: {instruction_list}")



            self.reflection_expert.set_subtask_and_instructions(subtask, instruction_list)
            self.action_expert.set_current_instruction(instruction_list[0])

        def action_expert(state: State):
            """
            Action expert case definition:
                1. Planning returns == done, meaning that the general task is fullfilled.
                2. Needs to keep executing instructions from the instruction list.

            Args:
                State: State of the graph containing the syncronised variables. 

            Return:
                1. state.osworld_action = done, notifying the benchmark the task is finished.
                2. state.osworld_action = pyautogui code to be executed by the benchmark.
            """

            # Case 1
            if (state["done"]):
                # logger.info("First case of the Barry Action Expert: Done")
                return {"osworld_action": "done"}
            
            # Case 2

            # Process the current instruction from the instruction list
            feedback = state["reflection_action"]
            action = self.action_expert.process_instruction(self.screenshot, self.SOM_screenshot, self.SOM_description, feedback)
            
            return {
                "osworld_action": action
            }                       
        
        def reflection_router(state: State):
            condition = state["reflection_planning"] != ""
            if condition:
                return {"next": "planning_expert"}

            # logger.info("Going back to the action expert")
            return {"next": "action_expert"}
        
        def reflection_expert(state: State):
            """
                Evaluates the most recent execution of the action expert. 
                Depending on this evaluation it performs one of the following cases:

                case 1: The instruction was successful, now it checks if it was the last instruction:
                    - it was the last instruction: {reflection_planning: 'finish'}
                    - it wasn't the last instruction: action_expert.set_current_instruction(next_instruction)
                
                case 2: The instruction has failed, evaluates if the error was minor or major:
                    - It was a minor error: {reflection_action: error and how to solve it}
                    - It was a major error: {reflection_planning: error and how to solve it}
            """
            #logger"evaluo si ha habido errores")
            successful = self.reflection_expert.evaluate_execution(self.screenshot)

            # case 1
            if successful:
                #logger"no ha habido errores \n")
                is_last_instruction = self.reflection_expert.is_last_instruction()
                if is_last_instruction:
                    #logger"Sí es la última instrucción")
                    return {
                        "reflection_planning": "finish",
                        "reflection_action": ""
                    }
                else:
                    #logger"no es la última instrucción")
                    next_instruction = self.reflection_expert.get_next_instruction()
                    #loggerf"esta es la siguiente instrucción {next_instruction}")
                    self.action_expert.set_current_instruction(next_instruction)
                    return {
                        "reflection_planning": "",
                        "reflection_action": ""
                    }
            
            # case 2
            #logger"SÍ ha habido errores \n")
            evaluated_error = self.reflection_expert.evaluate_error(self.screenshot, self.main_task)
            #loggerf"esta es la evaluación del error: {evaluated_error}")
            if evaluated_error.startswith("Minor:"):
                new_instruction = self.reflection_expert.create_new_instruction()
                self.action_expert.set_current_instruction(new_instruction)

                return {
                    "reflection_action": evaluated_error,
                    "reflection_planning": ""
                }
            else:
                return {
                    "reflection_action": "",
                    "reflection_planning": evaluated_error
                }

        

        # EDGES ----------------------------------------------------------

        graph_builder.add_node("start_router", start_router)
        graph_builder.add_node("planning_expert", planning_expert)
        graph_builder.add_node("action_expert", action_expert)
        graph_builder.add_node("reflection_expert", reflection_expert)
        graph_builder.add_node("reflection_router", reflection_router)

        graph_builder.add_edge(START, "start_router")
        graph_builder.add_conditional_edges(
            "start_router",
            lambda state: state.get("next"),
            {"planning_expert": "planning_expert", "reflection_expert": "reflection_expert"}
        )
        graph_builder.add_edge("planning_expert", "action_expert")
        graph_builder.add_edge("action_expert", END)
        graph_builder.add_edge("reflection_expert", "reflection_router")
        graph_builder.add_conditional_edges(
            "reflection_router",
            lambda state: state.get("next"),
            {"action_expert": "action_expert", "planning_expert": "planning_expert"}
        )

        # COMPILE ---------------------------------------------------

        self.graph = graph_builder.compile()

    def _process_new_screenshot(self, obs:dict):
        """
        Processes the new screenshot of the OSWorld environment through the Perception System.
        Generates the SOM of the screenshot and a description of its components.
        Stores the results in the self.screenshot, self.SOM_screenshot andself.SOM_description local variables.
        """
        if self.observation_type not in ["screenshot"]:
            raise ValueError(f"observation_type not supported: {self.observation_type}")

        if "screenshot" not in obs:
            raise ValueError("'screenshot' was not found in the recieved observation")
        
        self.perception_expert.store_screenshot(obs["screenshot"])
        self.perception_expert.process_screenshot()

        # Store the results in the local variables
        self.screenshot = self.perception_expert.get_screenshot()
        self.SOM_screenshot = self.perception_expert.get_som_screenshot()
        self.SOM_description = self.perception_expert.get_som_description()
    

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        """
        Sends the screenshot and the instruction to the Agent in order to generate PyAutoGUI actions.
        """
        # logger.info("Predict call in BarryAgent")
        self.trajectory_length += 1

        if self.trajectory_length > self.max_trajectory_length:
            logger.warning(f"Trajectory exceeds the maximum limit of {self.max_trajectory_length} steps")
            return "Maximum trajectory length exceeded", ["FAIL"]

        try:
            # Process the new screenshot and store it in the Perception Expert
            self._process_new_screenshot(obs)

            # If it's the first iteration, copy the task and add it to the history
            if self.first_iteration:
                self.main_task = instruction
                # Initialize the graph state with the main task if it's the first time
                # LangGraph will handle initializing instruction_list in the planning expert
                self.graph_state = {
                    "reflection_action": "",
                    "reflection_planning": "",
                    "done": False,
                    "osworld_action": "",
                }
            if self.sleep:
                logger.info("sleeeeep")
                self.sleep = False
                return "Task completed", ["time.sleep(1)"]
            
            self.sleep = True

            # If it's not the first iteration, the graph state is already saved from before
            final_state = self.graph.invoke(self.graph_state)
            self.graph_state = final_state
            osworld_action_to_return = final_state.get("osworld_action")
            is_done = final_state.get("done", False)

            if is_done and osworld_action_to_return == "done":
                # logger.info("BarryAgent: Task successfully completed.")
                return "Task completed", ["DONE"]

            if osworld_action_to_return:
                # logger.info(f"BarryAgent: Action decided by the agent: '{osworld_action_to_return}'")
                pyautogui_instructions = [line for line in osworld_action_to_return.strip().splitlines() if line]
                pyautogui_instructions.append("time.sleep(3)")
                logger.info("instructions to execute")
                logger.info(pyautogui_instructions)

                return "Next action determined", pyautogui_instructions
            
            else:
                logger.warning("BarryAgent: The graph did not produce a valid OSWorld action in this iteration.")
                return "FAIL: No OSWorld action generated in this cycle.", ["FAIL"]

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "FAIL: Exception occurred during prediction.", ["FAIL"]


    def reset(self, runtime_logger):
        """
        Resets the agent's state for a new task.
        """
        self.trajectory_length = 0
        self.call_user_count = 0
        self.first_iteration = True
        # logger.info("Agent reset")



if __name__ == "__main__":
    """
    Local test block for the agent.
    """
    try:
        # Create an instance of the agent
        agent = BarryAgent()
        
        # Reset the agent
        agent.reset(None)
        
        # Test instruction
        test_instruction = "List the files in the current directory"
        
        # Simulate test observation (you would need a real screenshot)
        test_obs = {
            "screenshot": b"",  # Place the real screenshot bytes here
        }
        
        print("Agent initialized successfully")

    except Exception as e:
        print(f"Test error: {e}")
        print("Make sure GEMINI_API_KEY is set in your .env file")