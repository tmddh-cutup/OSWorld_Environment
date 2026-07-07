"""Script to run end-to-end evaluation on the benchmark.
Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

import argparse
import datetime
import json
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=True)

GUI_AGENT_BENCHMARK_REPO = os.environ.get(
    "GUI_AGENT_BENCHMARK_REPO",
    r"D:\SeungO\gui_agent\guiAgent_OSworld_benchmark",
)

if GUI_AGENT_BENCHMARK_REPO and GUI_AGENT_BENCHMARK_REPO not in sys.path:
    sys.path.insert(0, GUI_AGENT_BENCHMARK_REPO)

print("[OSWorld DEBUG] GUI_AGENT_BENCHMARK_REPO =", GUI_AGENT_BENCHMARK_REPO)
print("[OSWorld DEBUG] MODEL_BACKEND =", os.environ.get("MODEL_BACKEND"))
print("[OSWorld DEBUG] MODEL_PATH =", os.environ.get("MODEL_PATH"))

from tqdm import tqdm

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from mm_agents.agent import PromptAgent

# Almost deprecated since it's not multi-env, use run_multienv_*.py instead

#  Logger Configs {{{ #
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
#  }}} Logger Configs #

logger = logging.getLogger("desktopenv.experiment")


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument(
        "--provider_name", type=str, default="vmware",
        help="Virtualization provider (vmware, docker, aws, azure, gcp, virtualbox)"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless machine"
    )
    parser.add_argument(
        "--action_space", type=str, default="pyautogui", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="a11y_tree",
        help="Observation type",
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )

    # multi GUI agent config
    parser.add_argument(
    "--model_backend",
    type=str,
    default=os.environ.get("MODEL_BACKEND", "qwen3vl").strip('"').strip("'"),
)
    parser.add_argument(
    "--model_path",
    type=str,
    default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct").strip('"').strip("'"),
)
    parser.add_argument("--agent_type", type=str, default="act_only")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--max_step_fails", type=int, default=3)
    parser.add_argument("--history_n", type=int, default=6)
    parser.add_argument(
        "--coordinate_type",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
    )
    parser.add_argument(
        "--api_backend",
        type=str,
        default="dashscope",
        choices=["dashscope", "openai"],
    )
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--thinking_budget", type=int, default=8192)
    parser.add_argument("--agent_tmp_dir", type=str, default="./results/_multi_gui_agent_tmp")
    parser.add_argument("--verbose_agent", action="store_true")

    # lm config
    parser.add_argument("--model", type=str, default="qwen3-vl-8b")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--stop_token", type=str, default=None)

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_all.json"
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results")
    args = parser.parse_args()

    return args


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    scores = []
    max_steps = args.max_steps

    # log args
    logger.info("Args: %s", args)
    # set wandb project
    cfg_args = {
        "path_to_vm": args.path_to_vm,
        "provider_name": args.provider_name,
        "headless": args.headless,
        "action_space": args.action_space,
        "observation_type": args.observation_type,
        "screen_width": args.screen_width,
        "screen_height": args.screen_height,
        "sleep_after_execution": args.sleep_after_execution,
        "max_steps": args.max_steps,
        "max_trajectory_length": args.max_trajectory_length,


        # model / agent config
        "model": args.model,
        "model_backend": args.model_backend,
        "agent_type": args.agent_type,
        "max_iterations": args.max_iterations,
        "max_step_fails": args.max_step_fails,
        "history_n": args.history_n,
        "coordinate_type": args.coordinate_type,
        "api_backend": args.api_backend,
        "enable_thinking": args.enable_thinking,
        "thinking_budget": args.thinking_budget,

        # Lm config
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "stop_token": args.stop_token,

        # Logging
        "result_dir": args.result_dir,
        "agent_tmp_dir": args.agent_tmp_dir,
        "verbose_agent": args.verbose_agent,
    }

    if args.agent_type == "gui_agent_osworld":
        from mm_agents.gui_agent_osworld import GUIAgentOSWorld

        logger.info("[Agent] Using gui_agent_osworld GUIAgentOSWorld")
        logger.info("[Agent] GUI_AGENT_BENCHMARK_REPO=%s", os.environ.get("GUI_AGENT_BENCHMARK_REPO"))

        agent = GUIAgentOSWorld(
            model_name=args.model,
            observation_type=args.observation_type,
            action_space=args.action_space,
            max_steps=args.max_steps,
            model_path=os.environ.get("MODEL_PATH"),
        )
        if not hasattr(agent, "action_space"):
            agent.action_space = args.action_space
    
    elif args.agent_type == "gui_agent_qwen3_act_baseline":
        from gui_agent.osworld.act_baseline_agent import GUIAgentOSWorldActBaseline
        
        agent = GUIAgentOSWorldActBaseline(
        model_name=args.model,
        observation_type=args.observation_type,
        action_space=args.action_space,
        max_steps=args.max_steps,
        model_path=args.model_path,
        )
    
    elif args.agent_type == "plan_act_baseline":
        from gui_agent.osworld.plan_act_baseline_wrapper import GUIAgentPlanActBaselineOSWorld

        agent = GUIAgentPlanActBaselineOSWorld(
            model_name=args.model,
            observation_type=args.observation_type,
            action_space=args.action_space,
            max_steps=args.max_steps,
            model_path=getattr(args, "model_path", None),
        )

    elif args.agent_type == "act_only":
        from gui_agent.osworld.compare_agents import GUIAgentActOnlyOSWorld

        logger.info("[Agent] Using act_only GUIAgentActOnlyOSWorld")
        agent = GUIAgentActOnlyOSWorld(
            model_name=args.model,
            observation_type=args.observation_type,
            action_space=args.action_space,
            max_steps=args.max_steps,
            model_backend=args.model_backend,
            model_path=args.model_path,
            max_step_fails=args.max_step_fails,
            history_n=args.history_n,
            agent_tmp_dir=args.agent_tmp_dir,
        )

    elif args.agent_type == "plan_act":
        from gui_agent.osworld.compare_agents import GUIAgentPlanActOSWorld

        logger.info("[Agent] Using plan_act GUIAgentPlanActOSWorld")
        agent = GUIAgentPlanActOSWorld(
            model_name=args.model,
            observation_type=args.observation_type,
            action_space=args.action_space,
            max_steps=args.max_steps,
            model_backend=args.model_backend,
            model_path=args.model_path,
            max_step_fails=args.max_step_fails,
            history_n=args.history_n,
            agent_tmp_dir=args.agent_tmp_dir,
        )
    
    else:
        from multi_guiagent_osworld.agent_factory import build_agent

        agent = build_agent(args)

    

    env = DesktopEnv(
        provider_name=args.provider_name,
        path_to_vm=args.path_to_vm,
        action_space=agent.action_space,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        os_type = "Ubuntu",
        require_a11y_tree=args.observation_type
        in ["a11y_tree", "screenshot_a11y_tree", "som"],
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
            # wandb each example config settings
            cfg_args["instruction"] = instruction
            cfg_args["start_time"] = datetime.datetime.now().strftime(
                "%Y:%m:%d-%H:%M:%S"
            )
            # run.config.update(cfg_args)

            example_result_dir = os.path.join(
                args.result_dir,
                args.action_space,
                args.observation_type,
                args.model,
                args.model_backend,
                args.agent_type,
                domain,
                example_id,
            )
            os.makedirs(example_result_dir, exist_ok=True)
            # example start running
            try:
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
            except Exception as e:
                logger.error(f"Exception in {domain}/{example_id}: {e}")
                # Only attempt to end recording if controller exists (not Docker provider)
                if hasattr(env, 'controller') and env.controller is not None:
                    env.controller.end_recording(
                        os.path.join(example_result_dir, "recording.mp4")
                    )
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    f.write(
                        json.dumps(
                            {"Error": f"Time limit exceeded in {domain}/{example_id}"}
                        )
                    )
                    f.write("\n")

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores) if scores else 0}")


def get_unfinished(
    action_space, use_model, model_backend, agent_type, observation_type, result_dir, total_file_json
):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model, model_backend, agent_type,)

    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        # empty all files under example_id
                        for file in os.listdir(example_path):
                            os.remove(os.path.join(example_path, file))
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json


def get_result(action_space, use_model, model_backend, agent_type, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model, model_backend, agent_type)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        # empty all files under example_id
                        try:
                            all_result.append(
                                float(
                                    open(
                                        os.path.join(example_path, "result.txt"), "r"
                                    ).read()
                                )
                            )
                        except:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


if __name__ == "__main__":
    ####### The complete version of the list of examples #######
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    
    # save args to json in result_dir/action_space/observation_type/model/args.json
    path_to_args = os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        args.model,
        args.model_backend,
        args.agent_type,
        "args.json",
    )
    os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
    with open(path_to_args, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    test_file_list = get_unfinished(
        args.action_space,
        args.model,
        args.model_backend,
        args.agent_type,
        args.observation_type,
        args.result_dir,
        test_all_meta,
    )
    left_info = ""
    for domain in test_file_list:
        left_info += f"{domain}: {len(test_file_list[domain])}\n"
    logger.info(f"Left tasks:\n{left_info}")

    get_result(
        args.action_space,
        args.model,
        args.model_backend,
        args.agent_type,
        args.observation_type,
        args.result_dir,
        test_all_meta,
    )
    test(args, test_file_list)
