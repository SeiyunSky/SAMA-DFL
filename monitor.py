"""
系统资源监控脚本
用于vGPU-32GB环境的实时监控
"""
import subprocess
import time
import sys
from datetime import datetime


def get_gpu_memory():
    """获取GPU显存使用情况"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        used, total = map(int, result.stdout.strip().split(','))
        return used, total
    except:
        return None, None


def get_gpu_util():
    """获取GPU利用率"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        util = int(result.stdout.strip())
        return util
    except:
        return None


def get_process_count():
    """获取Python进程数"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.split('\n')
        python_procs = [l for l in lines if 'python' in l.lower() and 'run_experiments' in l]
        return len(python_procs)
    except:
        return 0


def monitor(interval=5, duration=None):
    """
    监控系统资源

    参数:
        interval: 监控间隔（秒）
        duration: 监控时长（秒），None表示持续监控
    """
    print("="*60)
    print("vGPU-32GB 资源监控")
    print("="*60)
    print(f"监控间隔: {interval}秒")
    print("按 Ctrl+C 停止监控\n")

    start_time = time.time()

    try:
        while True:
            timestamp = datetime.now().strftime('%H:%M:%S')

            # GPU信息
            mem_used, mem_total = get_gpu_memory()
            gpu_util = get_gpu_util()
            num_procs = get_process_count()

            if mem_used is not None:
                mem_percent = mem_used / mem_total * 100
                print(f"[{timestamp}] "
                      f"GPU显存: {mem_used}/{mem_total}MB ({mem_percent:.1f}%) | "
                      f"GPU利用率: {gpu_util}% | "
                      f"Python进程: {num_procs}")
            else:
                print(f"[{timestamp}] 无法获取GPU信息")

            # 检查是否超时
            if duration and (time.time() - start_time) > duration:
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n监控已停止")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        interval = int(sys.argv[1])
    else:
        interval = 5

    monitor(interval=interval)
