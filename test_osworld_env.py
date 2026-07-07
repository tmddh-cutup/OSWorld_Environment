import io
import time
from PIL import Image
from desktop_env.desktop_env import DesktopEnv

example = {
    "id": "debug-001",
    "instruction": "Please open the desktop and do nothing else.",
    "config": [],
    "evaluator": None,
}
env = None

try:
    env = DesktopEnv(
        provider_name="vmware",
        path_to_vm=r"D:\OSWorldHost\OSWorld\vmware_vm_data\Ubuntu0\Ubuntu0.vmx",
        action_space="pyautogui",
        screen_size=(1920, 1080),
        headless=False,
        os_type="Ubuntu",
        require_a11y_tree=False,
        client_password="password",
    )

    obs = env.reset(task_config=example)
    print("obs type:", type(obs))
    print("obs keys:", obs.keys())

    Image.open(io.BytesIO(obs["screenshot"])).save("before.png")
    print("saved: before.png")

    action = "import time; pyautogui.click(500, 300, button='right'); time.sleep(2)"
    print("action:", action)

    obs, reward, done, info = env.step(action)

    Image.open(io.BytesIO(obs["screenshot"])).save("after.png")
    print("saved: after.png")

    print("reward:", reward)
    print("done:", done)
    print("info:", info)

    print("Waiting 5 seconds before close...")
    time.sleep(5)

except Exception as e:
    print("ERROR:", type(e).__name__, e)

finally:
    try:
        if env is not None:
            env.close()
            print("env closed")
    except Exception as close_e:
        print("close error:", close_e)