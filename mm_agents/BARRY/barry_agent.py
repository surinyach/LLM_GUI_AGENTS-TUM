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
from .error_expert import ErrorExpert

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
        

        # PERCEPTION SYSTEM
        self.SOM_screenshot = ""
        

        self.main_task = ""
        self.first_iteration = True

        self.action_expert = ActionExpert() 
        self.planning_expert = PlanningExpert()
        self.reflection_expert = ReflectionExpert()
        self.error_expert = ErrorExpert()
        
        self.graph = None
        self.graph_state = {} 

        

        # LANG GRAPH

        # STATE --------------------------------------------------------------------

        class State(TypedDict):
            #intruction_list: str    # no hace falta guardarla porque se la pasa directamente cuando las crea
            action_expert_feedback: str
            reflection_expert_feedback: str
            info_for_error_expert: str
            error_expert_feedback: str
            new_instruction_list: bool

            done: bool
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
            if state.get("done", False):
                return {"osworld_action": "done"}
            logger.info("estoy en el nodo del action expert voy a llamar al predict")
            osworld_action = self.action_expert.predict(self.SOM)
            logger.info(f"estoy en el nodo del action expert respuesta del predict: {osworld_action}")

            if osworld_action.startswith("error:"):
                logger.info(f"estoy dentro del if error del nodo del action expert, este es el error que ha detectado: {osworld_action}")
                return {"execution_error": osworld_action,
                        "osworld_action": None, 
                        "new_instruction_list": False}
            
            if osworld_action == 'finish':
                return {"done": True}

            
            return {"osworld_action": osworld_action, 
                    "new_instruction_list": False}
        
        def action_router(state: State):
            condition = not state.get("done", False) and state.get("execution_error", "")
            if condition: # "" en python es falsy y cualquier otro string será true
                return {"next": "reflection_expert"}
            
            return {"next": "end"} # para que el predict devuelva la acción y se pueda volver a llamar al predict después.
        


        def reflection_expert(state: State):
            """
            case 1: Action expert has an execution error during the execution of the instruction list

            case 2: Action expert finishes execution its instruction list without any execution error
            """
            # case 1
            if state["osworld_action"].startswith("error:"):
                reflection_expert_feedback = self.reflection_expert.execution_error_reflection(
                    state["osworld_action"],
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


        def reflection_router(state: State):
            if state.get("info_for_error_expert", None) is not None:
                return {"next": "error_expert"}
            return {"next": "planning_expert"}




        def error_expert(state: State):
            error_expert_feedback = self.error_expert.predict(state["info_for_error_expert"], self.SOM)
                
            return {"error_expert_feedback": error_expert_feedback,
                    "info_for_error_expert": None} # lo dejamos en blanco para la siguiente vez
        

        # EDGES ----------------------------------------------------------

        graph_builder.add_node("start_router", start_router)
        graph_builder.add_node("planning_expert", planning_expert)
        graph_builder.add_node("action_expert", action_expert)
        graph_builder.add_node("action_router", action_router)
        graph_builder.add_node("reflection_expert", reflection_expert)
        graph_builder.add_node("reflection_router", reflection_router)
        graph_builder.add_node("error_expert", error_expert)

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
        graph_builder.add_edge("reflection_expert", "reflection_router")
        graph_builder.add_conditional_edges(
            "reflection_router",
            lambda state: state.get("next"),
            {"planning_expert": "planning_expert", "error_expert": "error_expert"}
        )
        graph_builder.add_edge("error_expert", "reflection_expert")

        # COMPILE ---------------------------------------------------

        self.graph = graph_builder.compile()





    def process_observation(self, obs: Dict) -> Image.Image:
        """
        Procesa la captura de pantalla del entorno para convertirla a una imagen PIL compatible con Gemini.
        """
        if self.observation_type not in ["screenshot", "screenshot_a11y_tree"]:
            logger.error(f"observation_type debe ser 'screenshot' o 'screenshot_a11y_tree', recibido: {self.observation_type}")
            raise ValueError(f"observation_type no soportado: {self.observation_type}")

        if "screenshot" not in obs:
            logger.error("No se encontró 'screenshot' en la observación")
            raise ValueError("No se encontró 'screenshot' en la observación")

        try:
            screenshot = obs["screenshot"]
            # Manejar si screenshot es str
            if isinstance(screenshot, str):
                screenshot = screenshot.encode('utf-8')
            # Convertir a imagen PIL
            image = Image.open(BytesIO(screenshot))
            return image
        except Exception as e:
            logger.error(f"Error al procesar el screenshot: {e}")
            raise


    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        """
        Envía la captura de pantalla y la instrucción a Gemini para generar acciones pyautogui.
        Retorna una tupla con la respuesta de Gemini y la lista de acciones con estados apropiados.
        """
        logger.info("Predict de barry agent")
        self.trajectory_length += 1
        
        if self.trajectory_length > self.max_trajectory_length:
            logger.warning(f"Trayectoria excede el límite máximo de {self.max_trajectory_length} pasos")
            return "Maximum trajectory length exceeded", ["FAIL"]
        
        try:
            self.SOM = self.process_observation(obs)
            
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
                    "new_instruction_list": True,

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
        
        
        # Para probar realmente, necesitarías cargar un screenshot real
        # response, actions = agent.predict(test_instruction, test_obs)
        # print(f"Respuesta: {response}")
        # print(f"Acciones: {actions}")
        
    except Exception as e:
        print(f"Error en la prueba: {e}")
        print("Asegúrate de tener configurado GEMINI_API_KEY en tu archivo .env")