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
        Inicializa el agente con la configuración para interactuar con OSWorld y la API de Gemini.
        """
        # Cargar variables de entorno desde .env
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY no encontrada en el archivo .env")
            raise ValueError("GEMINI_API_KEY no encontrada en el archivo .env")

        # Configurar la API de Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model)

        # Configurar parámetros del entorno
        self.observation_type = observation_type
        self.action_space = action_space
        
        # Historial para el agente
        self.trajectory_length = 0
        self.max_trajectory_length = 50
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
        self.SOM_screenshot = ""
        self.SOM_description = ""

        # LANG GRAPH

        # STATE --------------------------------------------------------------------

        class State(TypedDict):
            action_expert_feedback: str
            reflection_expert_feedback: str
            execution_error:str
            finished_instructions:bool

            osworld_action: str
        

        graph_builder = StateGraph(State)

        # NODES -----------------------------------------------

        def start_router(state: State):
            logger.info("estoy en el start_router")

            if self.first_iteration:
                logger.info("me voy al planning expert")
                return {"next": "planning_expert"}
                
            logger.info("me voy al action expert")
            return {"next": "action_expert"}
        
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
                an instruction list. When we have the current subtask and instruction list we call the action expert and reflection expert
                to save the subtask and instruction list.
                
            """
            # case 1
            if self.first_iteration:
                logger.info("nodo planning expert: primera iteración")
                subtask = self.planning_expert.decompose_main_task(self.main_task, self.SOM_screenshot)
                self.first_iteration = False

            # case 2
            elif state["reflection_expert_feedback"] == "":
                done = self.planning_expert.is_last_task(self.SOM_screenshot)
                if done:
                    return {"done": True}
                else:
                   subtask = self.planning_expert.rethink_subtask_list("", state["action_expert_feedback"], self.SOM_screenshot)
                   
            # case 3
            else:
                subtask = self.planning_expert.rethink_subtask_list(state["reflection_expert_feedback"], state["action_expert_feedback"], self.SOM_screenshot)

            # This has to be done after every case:
            instruction_list = self.planning_expert.decompose_subtask()

            self.action_expert.set_subtask_and_instructions(subtask, instruction_list)
            self.reflection_expert.set_subtask_and_instructions(subtask, instruction_list)

        def action_expert(state: State):
            """
            Action expert case definition:
                1. Planning returns instruction list == done, meaning that the general task is fullfilled.
                2. Previous iteration has finished the instruction list, informing consequently the action router.
                3. An error has been produced trying to resolve the instruction.
                4. Needs to keep executing instructions from the instruction list.

            Args:
                State: State of the graph containing the syncronised variables. 

            Return:
                1. state.osworld_action = done, notifying the benchmark the task is finished.
                2. state.finished _instructions = True and state.action_expert_feedback = description of the actions done since now.
                3. state.execution_error = response containing the error and its description and state.action_expert_feedack =
                description of the actions done since now.
                4. state.osworld_action = pyautogui code to be executed by the benchmark.
            """

            state["execution_error"] = ""

            # Case 1
            if (self.action_expert.get_instruction_list() == "done"):
                logger.info("Primer caso del Barry Action Expert: Done")
                return {"osworld_action": "done"}

            # Process the current instruction from the instruction list
            action = self.action_expert.process_instruction(self.SOM_screenshot, self.SOM_description)
            
            # Case 2
            if (action == "finish"):
                logger.info("Segundo caso del Barry Action Expert: Se han terminado las instrucciones, finish")
                return {
                    "finished_instructions": True,
                    "action_expert_feedback": self.action_expert.summarize_done_instructions()
                }

            # Case 3
            if (action.startswith("error:")):
                logger.info("Tercer caso del Barry Action Expert: Se ha producido un error tratando de resolver la tarea")
                return {
                    "execution_error": action.split("error:", 1)[-1],
                    "action_expert_feedback": self.action_expert.summarize_done_instructions()
                }
        
            # Case 4
            return {
                "osworld_action": action
            }
        
        def action_router(state: State):
            condition = state["osworld_action"] == "finish" or state["execution_error"]
            if condition: 
                return {"next": "reflection_expert"}
            
            return {"next": "end"}
        
        def reflection_expert(state: State):
            """
            case 1: Action expert has an execution error during the execution of the instruction list

            case 2: Action expert finishes execution its instruction list without any execution error
            """
            # case 1
            if state["execution_error"]:
                reflection_expert_feedback = self.reflection_expert.execution_error_reflection(
                    state["execution_error"],
                    state["action_expert_feedback"],
                    self.SOM_screenshot) 
                
            # case 2
            else:
                reflection_expert_feedback = self.reflection_expert.finished_instructions_eval(
                    state["action_expert_feedback"],
                    self.SOM_screenshot) 
                if reflection_expert_feedback.startswith("Yes"):
                    reflection_expert_feedback = ""
            
            return {"reflection_expert_feedback": reflection_expert_feedback}

        

        # EDGES ----------------------------------------------------------

        graph_builder.add_node("start_router", start_router)
        graph_builder.add_node("planning_expert", planning_expert)
        graph_builder.add_node("action_expert", action_expert)
        graph_builder.add_node("action_router", action_router)
        graph_builder.add_node("reflection_expert", reflection_expert)

        graph_builder.add_edge(START, "start_router")
        graph_builder.add_conditional_edges(
            "start_router",
            lambda state: state.get("next"),
            {"planning_expert": "planning_expert", "action_expert": "action_expert"}
        )
        graph_builder.add_edge("planning_expert", "action_expert")
        graph_builder.add_edge("action_expert", "action_router")
        graph_builder.add_conditional_edges(
            "action_router",
            lambda state: state.get("next"),
            {"reflection_expert": "reflection_expert", "end": END}
        )
        graph_builder.add_edge("reflection_expert", "planning_expert")

        # COMPILE ---------------------------------------------------

        self.graph = graph_builder.compile()

    def _process_new_screenshot(self, obs:dict):
        """
        Processes the new screenshot of the OSWorld environment through the Perception System.
        Generates the SOM of the screenshot and a description of its components.
        Stores the results in the self.SOM_screenshot and self.SOM_description local variables.
        """
        if self.observation_type not in ["screenshot"]:
            raise ValueError(f"observation_type not supported: {self.observation_type}")

        if "screenshot" not in obs:
            raise ValueError("'screenshot' was not found in the recieved observation")
        
        self.perception_expert.store_screenshot(obs["screenshot"])
        self.perception_expert.process_screenshot()

        # Store the results in the local variables
        self.SOM_screenshot = self.perception_expert.get_som_screenshot()
        self.SOM_description = self.perception_expert.get_som_description()
    

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        """
        Sends the screenshot and the instruction to the Agent in order to generate pyautogui actions.
        """
        logger.info("Predict de barry agent")
        self.trajectory_length += 1
        
        if self.trajectory_length > self.max_trajectory_length:
            logger.warning(f"Trayectoria excede el límite máximo de {self.max_trajectory_length} pasos")
            return "Maximum trajectory length exceeded", ["FAIL"]
        
        try:
            # Process the new screenshot and store it in the Perception Expert
            self._process_new_screenshot(obs)
            
            # Si es la primera iteración copiamos la tarea y la añadimos al historial
            if self.first_iteration:
                self.main_task = instruction
                # Inicializa el estado del grafo con la tarea principal si es la primera vez
                # LangGraph se encargará de inicializar instruction_list en planning_expert
                self.graph_state = {
                    "action_expert_feedback": "",
                    "reflection_expert_feedback": "",
                    "info_for_error_expert": "",
                    "error_expert_feedback": "",

                    "done": False,
                    "osworld_action": "",
                }
            
            # si no es la primera iteración ya tienes el grafo guardado de antes

            final_state = self.graph.invoke(self.graph_state)
            self.graph_state = final_state
            osworld_action_to_return = final_state.get("osworld_action")
            is_done = final_state.get("done", False)

            if is_done and osworld_action_to_return == "done": # doble comprobación aun que con una ya sería suficiente
                logger.info("BarryAgent: Tarea finalizada con éxito.")
                return "se acabo lo que se daba", ["DONE"]

            if osworld_action_to_return:
                logger.info(f"BarryAgent: Acción decidida por el agente: '{osworld_action_to_return}'")
                return "esta es la siguiente acción", [osworld_action_to_return]
            else:
                logger.warning("BarryAgent: El grafo no produjo una acción de OSWorld válida en esta iteración.")
                return "FAIL: No OSWorld action generated in this cycle.", ["FAIL"]

            
        except Exception as e:
            logger.error(f"Error al procesar la solicitud a Gemini: {e}")
            return f"Error: {e}", ["FAIL"]


    def reset(self, runtime_logger):
        """
        Reinicia el estado del agente para una nueva tarea.
        """
        self.trajectory_length = 0
        self.call_user_count = 0
        self.first_iteration = True
        logger.info("Agente reiniciado")



if __name__ == "__main__":
    """
    Bloque de prueba local para el agente.
    """
    try:
        # Crear una instancia del agente
        agent = BarryAgent()
        
        # Reiniciar el agente
        agent.reset(None)
        
        # Instrucción de prueba
        test_instruction = "Listar los archivos en el directorio actual"
        
        # Simular observación de prueba (necesitarías un screenshot real)
        test_obs = {
            "screenshot": b"",  # Aquí iría los bytes del screenshot real
        }
        
        print("Agente inicializado correctamente")
        
    
    except Exception as e:
        print(f"Error en la prueba: {e}")
        print("Asegúrate de tener configurado GEMINI_API_KEY en tu archivo .env")