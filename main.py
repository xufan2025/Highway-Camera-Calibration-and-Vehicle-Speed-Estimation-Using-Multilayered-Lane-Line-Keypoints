import subprocess
import sys


def run_command(command):

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"failed: {command}")


if __name__ == "__main__":


    cmd1 = (
        'conda run -n venv1 python main1.py'
    )

    run_command(cmd1)

    cmd2 = (
        'conda run -n venv2 python track.py '
        '--strong-sort-weights strong_sort\\deep\\checkpoint\\osnet_x1_0.pth '
        '--save-vid '
        '--show-vid '
        '--yolo-weights detect\\runs\\train\\exp_221226\\weights\\best.pt '
        '--source MOT16_eval\\1.mp4'
    )
    run_command(cmd2)


