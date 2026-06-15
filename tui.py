"""
SAMA-DFL Experiment Dashboard (TUI)
交互式实验管理界面
"""
import sys
import subprocess
import threading
import time
import os
from pathlib import Path

# Windows UTF-8 编码修复（必须在 rich 导入前）
os.environ.setdefault('PYTHONUTF8', '1')
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.prompt import Prompt, Confirm
from rich.rule import Rule

PROJ_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJ_DIR))

from utils import check_all_datasets, download_dataset

DATA_DIR = str(PROJ_DIR / 'data')

console = Console()

# 显存占用阈值：超过此值不再向该卡补任务
GPU_MEM_THRESHOLD = 0.70


# ──────────────────────────────────────────────────────────
# GPU 信息
# ──────────────────────────────────────────────────────────

def get_gpu_info(index=0):
    try:
        result = subprocess.run(
            ['nvidia-smi',
             f'--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode != 0:
            return None
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if index >= len(lines):
            return None
        parts = lines[index].split(',')
        return {
            'mem_used': int(parts[0].strip()),
            'mem_total': int(parts[1].strip()),
            'util': int(parts[2].strip()),
            'temp': int(parts[3].strip()),
        }
    except Exception:
        return None


def get_gpu_count():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode != 0:
            return 1
        return len([l for l in result.stdout.strip().splitlines() if l.strip()])
    except Exception:
        return 1


def _gpu_mem_pct(gpu_index):
    info = get_gpu_info(gpu_index)
    if info is None:
        return 0.0
    return info['mem_used'] / info['mem_total']


def _render_one_gpu(info, label):
    if info is None:
        return Text(f"{label}: 不可用", style="dim")
    pct = info['mem_used'] / info['mem_total']
    bar_len = 20
    filled = int(bar_len * pct)
    bar = "#" * filled + "." * (bar_len - filled)
    color = "green" if pct < 0.7 else ("yellow" if pct < 0.9 else "red")
    mem_text = Text()
    mem_text.append(f"{label} MEM  [{bar}] ", style="white")
    mem_text.append(f"{info['mem_used']}/{info['mem_total']} MB ({pct:.0%})", style=color)

    util_pct = info['util'] / 100
    u_filled = int(bar_len * util_pct)
    u_bar = "#" * u_filled + "." * (bar_len - u_filled)
    u_color = "green" if util_pct < 0.5 else ("yellow" if util_pct < 0.85 else "red")
    util_text = Text()
    util_text.append(f"{label} UTIL [{u_bar}] ", style="white")
    util_text.append(f"{info['util']}%", style=u_color)
    util_text.append(f"   Temp {info['temp']}C",
                     style="green" if info['temp'] < 75 else "yellow")
    return Text.assemble(mem_text, "\n", util_text)


def render_gpu_panel():
    gpu_count = get_gpu_count()
    lines = []
    for i in range(gpu_count):
        info = get_gpu_info(i)
        lines.append(_render_one_gpu(info, f"GPU{i}"))

    content = Text()
    for i, line in enumerate(lines):
        content.append_text(line)
        if i < len(lines) - 1:
            content.append("\n")

    panel_size = 3 + (gpu_count - 1) * 2
    return Panel(content, title="[bold cyan]GPU[/bold cyan]", border_style="cyan"), panel_size


# ──────────────────────────────────────────────────────────
# 结果扫描
# ──────────────────────────────────────────────────────────

def scan_results():
    results_dir = PROJ_DIR / 'results'
    if not results_dir.exists():
        return []
    files = list(results_dir.glob('*.png'))
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)


def render_results_table():
    files = scan_results()
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta", expand=True)
    table.add_column("文件名", style="cyan", no_wrap=True)
    table.add_column("大小", justify="right", style="green")
    table.add_column("时间", style="dim")

    if not files:
        table.add_row("[dim]暂无结果文件[/dim]", "", "")
    else:
        for f in files[:8]:
            size = f"{f.stat().st_size / 1024:.1f} KB"
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime('%m-%d %H:%M')
            table.add_row(f.name, size, mtime)

    return Panel(table, title=f"[bold magenta]已有结果 ({len(files)} 个)[/bold magenta]",
                 border_style="magenta")


# ──────────────────────────────────────────────────────────
# 数据集检测与下载
# ──────────────────────────────────────────────────────────

def render_dataset_panel(status: dict):    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("数据集", style="bold", width=12)
    table.add_column("状态")
    for name, ok in status.items():
        if ok:
            table.add_row(name.upper(), "[green]OK[/green]")
        else:
            table.add_row(name.upper(), "[red]缺失  (未下载)[/red]")
    all_ok = all(status.values())
    border = "green" if all_ok else "yellow"
    title = "[bold green]数据集状态[/bold green]" if all_ok else "[bold yellow]数据集状态  (有缺失)[/bold yellow]"
    return Panel(table, title=title, border_style=border)


def download_with_progress(name: str):
    script = (
        f"import sys; sys.path.insert(0, r'{PROJ_DIR}'); "
        f"from utils import download_dataset; "
        f"download_dataset('{name}', r'{DATA_DIR}')"
    )
    cmd = [sys.executable, '-u', '-c', script]
    console.print(f"\n[bold cyan]正在下载 {name.upper()}...[/bold cyan]")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=str(PROJ_DIR))
    with Live(console=console, refresh_per_second=4, screen=False) as live:
        lines = []
        for line in proc.stdout:
            lines.append(line.rstrip())
            if len(lines) > 10:
                lines = lines[-10:]
            text = Text()
            for l in lines:
                text.append(l + "\n", style="dim white")
            live.update(Panel(text, title=f"[cyan]下载 {name.upper()}[/cyan]", border_style="cyan"))
        proc.wait()
    if proc.returncode == 0:
        console.print(f"[green]✓ {name.upper()} 下载完成[/green]")
    else:
        console.print(f"[red]✗ {name.upper()} 下载失败 (exit {proc.returncode})[/red]")


def check_and_prompt_datasets():
    status = check_all_datasets(DATA_DIR)
    console.print(render_dataset_panel(status))
    missing = [name for name, ok in status.items() if not ok]
    if not missing:
        return
    console.print(f"[yellow]缺失数据集: {', '.join(m.upper() for m in missing)}[/yellow]")
    if Confirm.ask("是否现在下载缺失的数据集?", default=True):
        for name in missing:
            download_with_progress(name)
        console.print(render_dataset_panel(check_all_datasets(DATA_DIR)))


# ──────────────────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────────────────

EXPERIMENTS = {
    'A': {
        'label': '理论验证实验',
        'items': [
            ('A1', 'lemma41',    '引理4.1  幅度对齐公式验证'),
            ('A2', 'convergence','B1  收敛速率测量'),
            ('A3', 'kappa',      'B2  κ 值对比 SAMA vs BALANCE'),
            ('A4', 'consensus',  'B3  共识直径 vs 理论上界'),
            ('A5', 'lyapunov',   'B4  Lyapunov 函数单调性'),
        ]
    },
    'B': {
        'label': '性能对比实验',
        'items': [
            ('B1', 'multi_attack_table',  'C1  MNIST 多攻击汇总（8方法×6攻击）'),
            ('B2', 'cifar10_attack_table','C2  CIFAR-10 多攻击汇总（8方法×6攻击）'),
            ('B3', 'ablation',            '消融实验（4种变体）'),
            ('B4', 'client_scale',        '客户端数量扩展性（n=20/30/40）'),
        ]
    },
    'C': {
        'label': '参数扫描实验',
        'items': [
            ('C1', 'byz_sweep',   'C3  拜占庭比例扫描 (0.1~0.4)'),
            ('C2', 'noniid_sweep','C4  Non-IID 程度扫描 (α=0.1/0.2/0.3)'),
        ]
    },
}

PERF_GROUPS = {'B', 'C'}
NO_ATTACK_PROMPT = {'multi_attack_table', 'cifar10_attack_table', 'client_scale'}


# ──────────────────────────────────────────────────────────
# 单个实验进程
# ──────────────────────────────────────────────────────────

class ExperimentRunner:
    def __init__(self, exp_key, label, attack=None):
        self.exp_key = exp_key
        self.label = label
        self.attack = attack
        self.running = False
        self.done = False
        self.returncode = None
        self.output_lines = []
        self.start_time = None
        self.proc = None
        self._lock = threading.Lock()

    def start(self):
        self.running = True
        self.start_time = datetime.now()
        env = os.environ.copy()
        if self.attack:
            env['ATTACK_TYPE'] = self.attack
        if hasattr(self, 'env_extra') and self.env_extra:
            env.update(self.env_extra)
        cmd = [sys.executable, str(PROJ_DIR / 'run_experiments.py'),
               '--experiment', self.exp_key]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env, cwd=str(PROJ_DIR),
        )
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self):
        for line in self.proc.stdout:
            with self._lock:
                self.output_lines.append(line.rstrip())
                if len(self.output_lines) > 200:
                    self.output_lines = self.output_lines[-200:]
        self.proc.wait()
        self.returncode = self.proc.returncode
        self.running = False
        self.done = True

    def stop(self):
        if self.proc and self.running:
            self.proc.terminate()
            self.running = False

    def get_last_lines(self, n=6):
        with self._lock:
            return list(self.output_lines[-n:])

    def elapsed(self):
        if self.start_time is None:
            return ""
        s = int((datetime.now() - self.start_time).total_seconds())
        return f"{s // 60:02d}:{s % 60:02d}"

    def status_str(self):
        if self.done:
            return "[green]完成[/green]" if self.returncode == 0 else f"[red]失败({self.returncode})[/red]"
        if self.running:
            return f"[cyan]运行中 {self.elapsed()}[/cyan]"
        return "[dim]等待[/dim]"


# ──────────────────────────────────────────────────────────
# 并行批量运行
# ──────────────────────────────────────────────────────────

def run_parallel(jobs: list):
    """
    双卡动态调度：GPU显存 < 70% 才向该卡补任务。
    每张卡维护独立的 active 槽，任务按卡轮流分配。
    jobs: list of ExperimentRunner
    """
    gpu_count = get_gpu_count()
    pending = list(jobs)
    # active[gpu_idx] = list of running jobs on that gpu
    active = [[] for _ in range(gpu_count)]
    # 给每个 job 预先分配目标卡（轮询）
    for i, job in enumerate(jobs):
        job.target_gpu = i % gpu_count
    interrupted = False

    try:
        with Live(console=console, refresh_per_second=2, screen=False) as live:
            while pending or any(active):
                # 按卡补任务
                for gpu_idx in range(gpu_count):
                    mem_pct = _gpu_mem_pct(gpu_idx)
                    candidates = [j for j in pending if j.target_gpu == gpu_idx]
                    if candidates and mem_pct < GPU_MEM_THRESHOLD:
                        job = candidates[0]
                        pending.remove(job)
                        job.env_extra = {'CUDA_VISIBLE_DEVICES': str(gpu_idx)}
                        job.start()
                        active[gpu_idx].append(job)

                # 移除已完成
                for gpu_idx in range(gpu_count):
                    active[gpu_idx] = [j for j in active[gpu_idx] if not j.done]

                all_active = [j for slot in active for j in slot]
                live.update(_render_parallel_status(jobs, all_active, pending, active, gpu_count))
                time.sleep(0.5)

    except KeyboardInterrupt:
        interrupted = True
        for slot in active:
            for j in slot:
                j.stop()
        console.print("\n[yellow]已中断，正在终止所有子进程...[/yellow]")

    if not interrupted:
        console.print(Rule("[bold green]全部实验完成[/bold green]"))
        for job in jobs:
            mark = "[green]✓[/green]" if job.returncode == 0 else "[red]✗[/red]"
            console.print(f"  {mark} {job.label}  耗时 {job.elapsed()}")


def _render_parallel_status(all_jobs, all_active, pending, active_by_gpu, gpu_count):
    gpu_panel, gpu_size = render_gpu_panel()

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold", expand=True)
    table.add_column("实验", style="white")
    table.add_column("卡", width=5)
    table.add_column("状态", width=20)
    table.add_column("最新输出", style="dim")

    for job in all_jobs:
        last = job.get_last_lines(1)
        last_line = last[-1] if last else ""
        if len(last_line) > 55:
            last_line = last_line[:52] + "..."
        gpu_label = f"[cyan]GPU{job.target_gpu}[/cyan]" if hasattr(job, 'target_gpu') else ""
        table.add_row(job.label, gpu_label, job.status_str(), last_line)

    active_counts = "/".join(str(len(slot)) for slot in active_by_gpu)
    slots_text = f"运行: {active_counts}  排队: {len(pending)}"
    jobs_panel = Panel(table, title=f"[bold cyan]实验进度  {slots_text}[/bold cyan]",
                       border_style="cyan")

    layout = Layout()
    layout.split_column(
        Layout(gpu_panel, size=gpu_size),
        Layout(jobs_panel),
    )
    return layout


# ──────────────────────────────────────────────────────────
# 主菜单渲染
# ──────────────────────────────────────────────────────────

def render_menu():
    table = Table(box=box.ROUNDED, expand=True, show_header=False, padding=(0, 1))
    table.add_column("组", style="bold yellow", width=4)
    table.add_column("编号", style="bold cyan", width=4)
    table.add_column("描述", style="white")

    for group_key, group in EXPERIMENTS.items():
        table.add_row(f"[{group_key}]", "", f"[bold]{group['label']}[/bold]")
        for idx_key, exp_key, label in group['items']:
            table.add_row("", idx_key, label)

    table.add_row("", "", "")
    table.add_row("[*]", "all",  "[dim]并行运行全部实验[/dim]")
    table.add_row("[Q]", "quit", "[dim]退出[/dim]")

    return Panel(table, title="[bold white]SAMA-DFL Experiment Dashboard[/bold white]",
                 border_style="blue")


# ──────────────────────────────────────────────────────────
# 单个实验（单独运行时仍保留 Live 输出）
# ──────────────────────────────────────────────────────────

def run_single(exp_key, label, attack=None):
    job = ExperimentRunner(exp_key, label, attack)
    job.start()

    try:
        with Live(console=console, refresh_per_second=4, screen=False) as live:
            while job.running:
                lines = job.get_last_lines(14)
                log_text = Text()
                for line in lines:
                    if 'ERROR' in line or 'Error' in line or 'FAILED' in line:
                        log_text.append(line + "\n", style="red")
                    elif 'acc=' in line.lower() or 'accuracy' in line.lower() or '%' in line:
                        log_text.append(line + "\n", style="green")
                    elif 'Round' in line or 'round' in line or 'Epoch' in line:
                        log_text.append(line + "\n", style="cyan")
                    elif line.startswith('[') or '===' in line:
                        log_text.append(line + "\n", style="yellow")
                    else:
                        log_text.append(line + "\n", style="dim white")

                layout = Layout()
                gpu_panel, gpu_size = render_gpu_panel()
                layout.split_column(
                    Layout(gpu_panel, size=gpu_size),
                    Layout(Panel(log_text,
                                 title=f"[bold cyan]{label}  {job.elapsed()}[/bold cyan]",
                                 border_style="cyan")),
                    Layout(Panel(
                        f"[dim]攻击: [yellow]{attack or '默认'}[/yellow]    按 Ctrl+C 中断[/dim]",
                        border_style="dim"
                    ), size=3),
                )
                live.update(layout)
                time.sleep(0.25)
    except KeyboardInterrupt:
        job.stop()
        console.print("\n[yellow]已中断[/yellow]")
        return

    if job.returncode == 0:
        console.print(Rule(f"[green]✓ {label} 完成  耗时 {job.elapsed()}[/green]"))
    else:
        console.print(Rule(f"[red]✗ {label} 失败 (exit {job.returncode})[/red]"))
    for line in job.get_last_lines(6):
        console.print(f"  [dim]{line}[/dim]")


# ──────────────────────────────────────────────────────────
# 攻击类型选择
# ──────────────────────────────────────────────────────────

def choose_attack():
    console.print()
    console.print("[bold]选择攻击类型[/bold]（留空使用配置文件默认值）")
    console.print("  [cyan]1[/cyan] Gaussian        [cyan]2[/cyan] Label Flipping")
    console.print("  [cyan]3[/cyan] Omniscient      [cyan]4[/cyan] Krum Attack")
    console.print("  [cyan]5[/cyan] Trim Attack")
    choice = Prompt.ask("攻击", default="").strip()
    mapping = {
        '1': 'gaussian', '2': 'label_flipping',
        '3': 'omniscient', '4': 'krum_attack', '5': 'trim_attack',
        'gaussian': 'gaussian', 'label_flipping': 'label_flipping',
        'omniscient': 'omniscient', 'krum_attack': 'krum_attack',
        'trim_attack': 'trim_attack',
    }
    return mapping.get(choice, None)


# ──────────────────────────────────────────────────────────
# 查找实验
# ──────────────────────────────────────────────────────────

def resolve_experiment(choice):
    choice = choice.strip().upper()
    for group_key, group in EXPERIMENTS.items():
        for idx_key, exp_key, label in group['items']:
            if choice in (idx_key.upper(), exp_key.upper(), exp_key.lower()):
                needs_attack = group_key in PERF_GROUPS and exp_key not in NO_ATTACK_PROMPT
                return exp_key, label, needs_attack
    return None


# ──────────────────────────────────────────────────────────
# 主循环
# ──────────────────────────────────────────────────────────

def main():
    console.clear()
    gpu_panel, _ = render_gpu_panel()
    console.print(gpu_panel)
    check_and_prompt_datasets()
    console.print(render_results_table())

    while True:
        console.print()
        console.print(render_menu())
        console.print()

        choice = Prompt.ask("[bold white]输入编号或命令[/bold white]",
                            default="").strip().lower()

        if choice in ('q', 'quit', 'exit'):
            console.print("\n[dim]Bye.[/dim]")
            break

        if choice == 'all':
            # 收集所有实验，并行跑
            jobs = []
            for group_key, group in EXPERIMENTS.items():
                for idx_key, exp_key, label in group['items']:
                    attack = None  # 各实验用配置文件默认攻击
                    jobs.append(ExperimentRunner(exp_key, label, attack))
            run_parallel(jobs)
            console.print()
            console.print(render_results_table())
            continue

        # 整组并行运行
        if choice.upper() in EXPERIMENTS:
            group_key = choice.upper()
            attack = choose_attack() if group_key in PERF_GROUPS else None
            group = EXPERIMENTS[group_key]
            jobs = [
                ExperimentRunner(exp_key, label,
                                 None if exp_key in NO_ATTACK_PROMPT else attack)
                for _, exp_key, label in group['items']
            ]
            run_parallel(jobs)
            console.print()
            console.print(render_results_table())
            continue

        # 单个实验
        result = resolve_experiment(choice)
        if result is None:
            console.print(f"[red]未知实验: {choice}[/red]")
            continue

        exp_key, label, needs_attack = result
        attack = choose_attack() if needs_attack else None
        run_single(exp_key, label, attack)

        console.print()
        console.print(render_results_table())


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]退出[/dim]")
