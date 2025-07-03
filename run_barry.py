import argparse
import datetime
import json
import logging
import os
import sys

from tqdm import tqdm

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from mm_agents.barry_agent import BarryAgent

# Logger Configs
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(
    os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8"
)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)

logger = logging.getLogger("desktopenv.experiment")

def config() -> argparse.Namespace:
    """
    Configura los argumentos de la línea de comandos para la evaluación.
    """
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation for BarryAgent"
    )

    # Environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--headless", action="store_true", help="Run in headless machine")
    parser.add_argument("--action_space", type=str, default="pyautogui", help="Action type")
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type"
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)

    # Agent config
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples")

    # Model config - CAMBIO IMPORTANTE: Usar --model en lugar de --vision_model para consistencia
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Vision model for BarryAgent")
    # Mantener --vision_model para retrocompatibilidad
    parser.add_argument("--vision_model", type=str, default=None, help="Deprecated: use --model instead")

    # Example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--test_all_meta_path", type=str, default="evaluation_examples/test_all.json")

    # Logging related
    parser.add_argument("--result_dir", type=str, default="./results")
    
    parser.add_argument("--task", type=str, default=None, help="Especificar una tarea específica dentro del dominio")

    args = parser.parse_args()
    
    # NUEVO: Manejar retrocompatibilidad entre --model y --vision_model
    if args.vision_model is not None and args.model == "gemini-2.0-flash":
        args.model = args.vision_model
        logger.warning("--vision_model is deprecated, use --model instead")
    
    return args

def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    """
    Ejecuta la evaluación para todas las tareas definidas en test_all_meta.
    """
    scores = []
    max_steps = args.max_steps

    logger.info("Args: %s", args)
    cfg_args = {
        "path_to_vm": args.path_to_vm,
        "headless": args.headless,
        "action_space": args.action_space,
        "observation_type": args.observation_type,
        "screen_width": args.screen_width,
        "screen_height": args.screen_height,
        "sleep_after_execution": args.sleep_after_execution,
        "max_steps": args.max_steps,
        "model": args.model,   # CAMBIO: Usar args.model consistentemente
        "result_dir": args.result_dir,
    }

    agent = BarryAgent(
        vision_model=args.model,   # CAMBIO: Usar args.model
        action_space=args.action_space,
        observation_type=args.observation_type,
    )

    env = DesktopEnv(
        path_to_vm=args.path_to_vm,
        action_space=agent.action_space,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        os_type="Ubuntu",
        require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
    )

    for domain in tqdm(test_all_meta, desc="Domain"):
        for example_id in tqdm(test_all_meta[domain], desc="Example", leave=False):
            config_file = os.path.join(
                args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
            )
            with open(config_file, "r", encoding="utf-8") as f:
                example = json.load(f)

            logger.info(f"[Domain]: {domain}")
            logger.info(f"[Example ID]: {example_id}")

            instruction = example["instruction"]
            logger.info(f"[Instruction]: {instruction}")
            cfg_args["instruction"] = instruction
            cfg_args["start_time"] = datetime.datetime.now().strftime("%Y:%m:%d-%H:%M:%S")

            example_result_dir = os.path.join(
                args.result_dir,
                args.action_space,
                args.observation_type,
                args.model,   # CAMBIO: Usar args.model consistentemente
                domain,
                example_id,
            )
            os.makedirs(example_result_dir, exist_ok=True)

            try:
                logger.info(f"Starting evaluation for {domain}/{example_id}")
                
                # CRÍTICO: Asegurar que lib_run_single.run_single_example escriba result.txt
                logger.info(f"Aquí se llama a run_single_example con este env {env}")
                lib_run_single.run_single_example(
                    agent,
                    env,
                    example,
                    max_steps,
                    instruction,
                    args,
                    example_result_dir,
                    scores,
                )
                
                # CRÍTICO: Verificar que result.txt fue creado después de la ejecución
                result_file = os.path.join(example_result_dir, "result.txt")
                if not os.path.exists(result_file):
                    logger.error(f"CRÍTICO: result.txt no fue creado para {domain}/{example_id}")
                    logger.error("Esto indica un problema en lib_run_single.run_single_example")
                    # Crear result.txt con score 0 como fallback
                    with open(result_file, "w") as f:
                        f.write("0.0")
                    scores.append(0.0)
                else:
                    # Verificar que el contenido del archivo es válido
                    try:
                        with open(result_file, "r") as f:
                            score_content = f.read().strip()
                        score = float(score_content)
                        logger.info(f"Task {domain}/{example_id} completed with score: {score}")
                    except (ValueError, IOError) as e:
                        logger.error(f"Error reading result.txt for {domain}/{example_id}: {e}")
                        # Corregir archivo corrupto
                        with open(result_file, "w") as f:
                            f.write("0.0")
                        
            except Exception as e:
                logger.error(f"Exception in {domain}/{example_id}: {e}")
                # MEJORA: Verificar si env.controller existe antes de llamar end_recording
                if hasattr(env, 'controller') and hasattr(env.controller, 'end_recording'):
                    try:
                        env.controller.end_recording(
                            os.path.join(example_result_dir, "recording.mp4")
                        )
                    except Exception as recording_error:
                        logger.warning(f"Could not end recording: {recording_error}")
                
                # MEJORA: Crear el archivo de trayectoria con más información del error
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    error_info = {
                        "Error": f"Exception in {domain}/{example_id}",
                        "Exception": str(e),
                        "Type": type(e).__name__,
                        "Timestamp": datetime.datetime.now().isoformat()
                    }
                    f.write(json.dumps(error_info))
                    f.write("\n")
                
                # CRÍTICO: Escribir un archivo de resultado con score 0 para tareas fallidas
                result_file = os.path.join(example_result_dir, "result.txt")
                with open(result_file, "w") as f:
                    f.write("0.0")
                scores.append(0.0)
                logger.info(f"Task {domain}/{example_id} failed with score: 0.0")

    env.close()
    if scores:
        logger.info(f"Average score: {sum(scores) / len(scores)}")
    else:
        logger.info("No scores recorded")

def get_unfinished(action_space, use_model, observation_type, result_dir, total_file_json):
    """
    Identifica las tareas no completadas para evitar repetirlas.
    CRÍTICO: Esta función determina qué tareas ya están completadas.
    """
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    logger.info(f"Checking for finished tasks in: {target_dir}")

    if not os.path.exists(target_dir):
        logger.info(f"Target directory does not exist: {target_dir} - All tasks will be run")
        return total_file_json

    finished = {}
    total_checked = 0
    total_finished = 0
    
    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue
            
        finished[domain] = []
        
        for example_id in os.listdir(domain_path):
            if example_id == "onboard":
                continue
                
            example_path = os.path.join(domain_path, example_id)
            if not os.path.isdir(example_path):
                continue
                
            total_checked += 1
            result_file = os.path.join(example_path, "result.txt")
            
            if not os.path.exists(result_file):
                # CRÍTICO: Limpiar tareas incompletas más agresivamente
                logger.info(f"Cleaning incomplete task: {domain}/{example_id} (no result.txt)")
                for file in os.listdir(example_path):
                    file_path = os.path.join(example_path, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Could not remove {file_path}: {e}")
            else:
                # CRÍTICO: Verificar que result.txt tiene contenido válido
                try:
                    with open(result_file, "r") as f:
                        content = f.read().strip()
                    
                    if not content:
                        logger.warning(f"Empty result.txt for {domain}/{example_id} - marking as incomplete")
                        os.remove(result_file)
                        continue
                    
                    # Verificar que el contenido es un número válido
                    score = float(content)
                    finished[domain].append(example_id)
                    total_finished += 1
                    logger.debug(f"Found completed task: {domain}/{example_id} with score {score}")
                    
                except (ValueError, IOError) as e:
                    logger.warning(f"Invalid result.txt for {domain}/{example_id}: {e} - marking as incomplete")
                    try:
                        os.remove(result_file)
                    except:
                        pass

    logger.info(f"Scan complete: {total_finished}/{total_checked} tasks finished")

    if not finished:
        logger.info("No finished tasks found")
        return total_file_json

    # Filtrar tareas completadas del total
    for domain, examples in finished.items():
        if domain in total_file_json:
            original_count = len(total_file_json[domain])
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]
            remaining_count = len(total_file_json[domain])
            logger.info(f"Domain {domain}: {original_count - remaining_count} completed, {remaining_count} remaining")

    return total_file_json

def get_result(action_space, use_model, observation_type, result_dir, total_file_json):
    """
    Calcula la tasa de éxito de las tareas completadas.
    """
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []
    completed_tasks = 0
    failed_to_read = 0

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    result_file = os.path.join(example_path, "result.txt")
                    if os.path.exists(result_file):
                        try:
                            with open(result_file, "r") as f:
                                score = float(f.read().strip())
                            all_result.append(score)
                            completed_tasks += 1
                        except Exception as e:
                            logger.warning(f"Could not read result for {domain}/{example_id}: {e}")
                            all_result.append(0.0)
                            failed_to_read += 1

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        success_rate = sum(all_result) / len(all_result) * 100
        print(f"Current Success Rate: {success_rate:.2f}% ({completed_tasks} tasks completed, {failed_to_read} failed to read)")
        return all_result

if __name__ == "__main__":
    """
    Punto de entrada para ejecutar la evaluación.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    # NUEVO BLOQUE: Filtrar por tarea específica si se proporciona --task
    if args.task is not None:
        if args.domain == "all":
            logger.error("When --task is specified, --domain must also be specified and not 'all'.")
            sys.exit(1)
        
        # Asumir que test_all_meta ya ha sido filtrado a un solo dominio por el bloque anterior
        domain_name = list(test_all_meta.keys())[0] 
        if args.task not in test_all_meta[domain_name]:
            logger.error(f"Task '{args.task}' not found in domain '{domain_name}'.")
            sys.exit(1)
        
        # Si se proporciona una tarea específica, establecemos test_file_list para que contenga solo esa tarea
        # y saltamos la llamada a get_unfinished para este filtrado.
        test_file_list = {domain_name: [args.task]}
        logger.info(f"Explicitly set to execute only task: {args.task} in domain: {domain_name}")
    else:
        # Si no se especifica una tarea, usamos get_unfinished para determinar las tareas restantes
        # en el flujo de evaluación general.
        test_file_list = get_unfinished(
            args.action_space,
            args.model,
            args.observation_type,
            args.result_dir,
            test_all_meta,
        )
    
    left_info = ""
    total_remaining = 0
    for domain in test_file_list:
        count = len(test_file_list[domain])
        left_info += f"{domain}: {count}\n"
        total_remaining += count
    
    logger.info(f"Left tasks ({total_remaining} total):\n{left_info}")

    # CAMBIO: Usar args.model consistentemente
    get_result(
        args.action_space,
        args.model,   # CAMBIO: Usar args.model en lugar de args.vision_model
        args.observation_type,
        args.result_dir,
        test_all_meta, # get_result puede seguir usando el meta completo para reportar resultados generales
    )
    
    if total_remaining > 0:
        test(args, test_file_list)
    else:
        logger.info("No tasks remaining to execute")