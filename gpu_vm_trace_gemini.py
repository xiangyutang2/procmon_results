#!/usr/bin/env python3
"""
NVBit GPU Virtual Memory Address Access Tracer.

Runs a target command with NVBit's mem_trace.so preloaded, parses the sampled
warp memory accesses it prints, and writes them to a CSV.

Exit codes:
  0  normal
  1  setup error (e.g. mem_trace.so not found)
  2  target fell back to CPU ("no usable GPU found"); nothing could be traced
  3  target ran but zero kernel launches were observed (GPU probably unused)
"""
import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import threading
import time

DEFAULT_SO = os.path.expanduser(
    "~/Downloads/try/temp/nvbit_release_x86_64/tools/build/mem_trace/mem_trace.so"
)
DEFAULT_CMD = [
    os.path.expanduser(
        "~/Downloads/try/temp/nvbit_release_x86_64/test-apps/vectoradd/build/vectoradd"
    )
]
DEFAULT_OUTPUT = os.path.expanduser("~/Downloads/try/vram_trace.csv")

# ollama ships its GPU backend in a subfolder that llama-server does not search
# by itself. Pass --cuda-dir "" to skip this setup for other workloads.
DEFAULT_CUDA_DIR = "/usr/local/lib/ollama/cuda_v13"
OLLAMA_LIB_DIR = "/usr/local/lib/ollama"

KILL_GRACE_S = 10.0  # SIGTERM -> SIGKILL grace; NVBit needs time to flush on exit

EXIT_OK, EXIT_SETUP, EXIT_NO_GPU, EXIT_NO_LAUNCHES = 0, 1, 2, 3

CPU_FALLBACK_MARKERS = ("no usable GPU found",)

LAUNCH_RE = re.compile(
    r"MEMTRACE: CTX (0x[0-9a-fA-F]+) - LAUNCH - Kernel pc 0x[0-9a-fA-F]+ - "
    r"Kernel name (.+?) - grid launch id (\d+)"
)
# New format adds " - kind K - space S - bytes B - wsrc W" after the opcode; the optional
# group keeps the parser working (with blank columns) if mem_trace.so was not rebuilt.
ACCESS_RE = re.compile(
    r"MEMTRACE: CTX (0x[0-9a-fA-F]+) - grid_launch_id (\d+) - "
    r"CTA (\d+,\d+,\d+) - warp (\d+) - (\S+)"
    r"(?: - kind (\w+) - space (\w+) - bytes (\d+) - wsrc (\w+))? - (.+)"
)
INV_RE = re.compile(r"MEMTRACE_INV: opcode=(\S+) .*?decision=(\S+) .*?width=(\d+)\((\w+)\)")
NOT_TRACED_VRAM = ("UNSUPPORTED_TEX", "UNSUPPORTED_TMA", "NO_MREF")


def footprint(addrs, nbytes):
    """Unique 128B cache lines and 32B sectors touched by [a, a+nbytes) over all lanes."""
    width = max(nbytes, 1)
    lines, sectors = set(), set()
    for a in addrs:
        last = a + width - 1
        lines.update(range(a >> 7, (last >> 7) + 1))
        sectors.update(range(a >> 5, (last >> 5) + 1))
    return len(lines), len(sectors)


def strip_hex_leading_zeros(addr_str):
    """Strip leading zeros from a hex address while preserving 0x prefix (e.g. 0x0000777c -> 0x777c, 0x0000 -> 0x0)."""
    s = addr_str.strip()
    if s.startswith(("0x", "0X")):
        digits = s[2:].lstrip("0")
        return "0x" + (digits if digits else "0")
    digits = s.lstrip("0")
    return "0x" + (digits if digits else "0")


def terminate_process_group(proc, reason="Timeout reached", grace=KILL_GRACE_S):
    """Safely terminates the target workload and any child processes."""
    if proc.poll() is not None:
        return
    print(f"\n[STOPPING] {reason}. Terminating traced workload (PID {proc.pid})...", flush=True)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        print("[!] Target ignored SIGTERM; sending SIGKILL...", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass


def build_env(sample_period, mem_trace_so, cuda_dir):
    env = os.environ.copy()
    env["LD_PRELOAD"] = mem_trace_so
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    env["SAMPLE_PERIOD"] = str(sample_period)
    env["NOBANNER"] = "1"
    env["ACK_CTX_INIT_LIMITATION"] = "1"
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"

    if cuda_dir:
        # Join only non-empty entries: a trailing ":" would add the current
        # directory to the library search path.
        parts = [cuda_dir, OLLAMA_LIB_DIR, env.get("LD_LIBRARY_PATH", "")]
        env["LD_LIBRARY_PATH"] = ":".join(p for p in parts if p)

        backend = os.path.join(cuda_dir, "libggml-cuda.so")
        if os.path.isfile(backend):
            env["GGML_BACKEND_PATH"] = backend
        else:
            print(f"[WARNING] {backend} not found; GGML_BACKEND_PATH not set.", flush=True)
    return env


def run_gpu_vm_trace(
    target_cmd,
    output_csv=DEFAULT_OUTPUT,
    sample_period=10007,
    duration=None,
    start_delay=0.0,
    kernel_filter=None,
    mem_trace_so=DEFAULT_SO,
    cuda_dir=DEFAULT_CUDA_DIR,
    start_after_file=None,
):
    if not os.path.isfile(mem_trace_so):
        print(f"[ERROR] mem_trace.so not found: {mem_trace_so}", flush=True)
        return EXIT_SETUP

    env = build_env(sample_period, mem_trace_so, cuda_dir)

    launch_id_to_kernel = {}
    kernel_name_to_id = {"unknown": 0}
    rows_written = 0
    rows_new_format = 0
    inventory = {}  # opcode -> (decision, width_bytes, width_src)
    inv_path = output_csv + ".opcodes.txt"
    kernels_path = output_csv + ".kernels.csv"
    inv_file = None
    timed_out = threading.Event()
    interrupted = threading.Event()
    cpu_fallback = threading.Event()

    csv_headers = [
        "host_epoch_s",  # wall-clock time (s since epoch) when the row was parsed
        "ctx",
        "grid_launch_id",
        "kernel_id",
        "cta_id",
        "warp_id",
        "opcode",
        "op_kind",            # LOAD / STORE / ATOMIC / ASYNC_COPY (cp.async, global source) / OTHER
        "mem_space",          # memory space NVBit reported for the instruction
        "bytes_per_thread",   # payload per active lane (blank if mem_trace.so is old, 0 = unknown)
        "width_src",          # opcode = explicit suffix, default32 = assumed 32-bit, unknown
        "active_threads",
        "bytes_requested",    # active_threads * bytes_per_thread (before overlap)
        "unique_128B_cache_lines",  # lines touched by [addr, addr+bytes) over all lanes
        "unique_32B_sectors",
        "min_vaddr",
        "max_vaddr",          # highest start address
        "max_vaddr_end",      # one past the last byte touched
        *[f"lane_{i}_vaddr" for i in range(32)],
    ]

    dur_str = f"{duration}s" if duration else "until workload exit"
    print(f"--> Running target with GPU-side SAMPLE_PERIOD={sample_period} (Duration: {dur_str})", flush=True)
    if start_delay > 0:
        print(f"--> Start delay enabled: skipping trace writes for the first {start_delay}s", flush=True)
    if start_after_file:
        print(f"--> Start gate: skipping trace writes until '{start_after_file}' exists", flush=True)
    print(f"--> Target command: {' '.join(target_cmd)}", flush=True)
    print(f"--> Output CSV: {output_csv}", flush=True)

    # start_new_session=True places the target in a new process group so we can
    # cleanly stop the workload without prematurely killing this trace reader.
    process = subprocess.Popen(
        target_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    # Intercept SIGINT/SIGTERM to terminate the child process group first,
    # allowing the stdout loop below to finish reading any flushed NVBit buffers.
    def _on_signal(signum, frame):
        if not interrupted.is_set():
            interrupted.set()
            sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            threading.Thread(
                target=terminate_process_group,
                args=(process, f"Interrupted by signal ({sig_name})"),
                daemon=True,
            ).start()

    old_sigint = signal.signal(signal.SIGINT, _on_signal)
    old_sigterm = signal.signal(signal.SIGTERM, _on_signal)

    timer = None
    if duration is not None and duration > 0:
        def _on_timeout():
            timed_out.set()
            terminate_process_group(process, f"Predetermined duration ({duration}s) reached")

        timer = threading.Timer(duration, _on_timeout)
        timer.daemon = True
        timer.start()

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    trace_start_time = time.time() + start_delay
    gate_open = start_after_file is None
    last_gate_check = 0.0
    last_flush = time.time()

    try:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
            f.flush()

            # Read continuously until EOF so any lines flushed during shutdown are captured
            for raw in process.stdout:
                line = raw.strip()
                if not line:
                    continue

                # The target silently ran on the CPU: no CUDA context, nothing to trace.
                if not cpu_fallback.is_set() and any(m in line for m in CPU_FALLBACK_MARKERS):
                    cpu_fallback.set()
                    threading.Thread(
                        target=terminate_process_group,
                        args=(process, "target fell back to CPU; no kernels can be traced"),
                        daemon=True,
                    ).start()

                # 0. Opcode inventory printed by mem_trace.so goes to a sidecar file
                pos = line.find("MEMTRACE_INV")
                if pos >= 0:
                    # the target's own log can leave a partial line in front of ours
                    if pos > 0 and line[:pos].strip():
                        print(f"[TARGET] {line[:pos].rstrip()}", flush=True)
                    line = line[pos:]
                    if inv_file is None:
                        inv_file = open(inv_path, "w")
                    inv_file.write(line + "\n")
                    m = INV_RE.match(line)
                    if m:
                        inventory[m.group(1)] = (m.group(2), int(m.group(3)), m.group(4))
                    continue

                # 1. Always capture kernel launch metadata (even before the trace window opens)
                if " - LAUNCH - " in line:
                    m = LAUNCH_RE.search(line)
                    if m:
                        ctx, kname, launch_id = m.groups()
                        launch_id_to_kernel[(ctx, launch_id)] = kname
                        if kname not in kernel_name_to_id:
                            kernel_name_to_id[kname] = len(kernel_name_to_id)
                    continue

                pos = line.find("MEMTRACE: ")
                if pos < 0:
                    print(f"[TARGET] {line}", flush=True)
                    continue
                if pos > 0:
                    # llama-server and NVBit share stdout; a partial log line can precede ours
                    if line[:pos].strip():
                        print(f"[TARGET] {line[:pos].rstrip()}", flush=True)
                    line = line[pos:]

                now = time.time()

                # Skip writing memory traces until start_delay has elapsed
                if now < trace_start_time:
                    continue

                # Skip writing memory traces until the start gate file exists
                if not gate_open:
                    if now - last_gate_check >= 0.1:
                        last_gate_check = now
                        gate_open = os.path.exists(start_after_file)
                        if gate_open:
                            print("[TRACE] start gate opened; recording accesses", flush=True)
                    if not gate_open:
                        continue

                # 2. Parse sampled warp memory access lines
                if " - CTA " in line:
                    m = ACCESS_RE.search(line)
                    if not m:
                        continue

                    ctx, launch_id, cta_id, warp_id, opcode, kind, space, nbytes_s, wsrc, addrs_str = m.groups()
                    kname = launch_id_to_kernel.get((ctx, launch_id), "unknown")

                    if kernel_filter and kernel_filter not in kname:
                        continue

                    addrs = addrs_str.strip().split()
                    if len(addrs) != 32:
                        continue

                    try:
                        vals = [int(a, 16) for a in addrs]
                    except ValueError:
                        continue
                    int_addrs = [v for v in vals if v != 0]  # inactive lanes are logged as 0
                    active_threads = len(int_addrs)
                    nbytes = int(nbytes_s) if nbytes_s is not None else 0
                    if nbytes_s is not None:
                        rows_new_format += 1
                    if int_addrs:
                        unique_cache_lines, unique_sectors = footprint(int_addrs, nbytes)
                    else:
                        unique_cache_lines = unique_sectors = 0
                    min_vaddr = hex(min(int_addrs)) if int_addrs else "0x0"
                    max_vaddr = hex(max(int_addrs)) if int_addrs else "0x0"
                    max_vaddr_end = hex(max(int_addrs) + nbytes) if int_addrs else "0x0"

                    kid = kernel_name_to_id.get(kname, 0)
                    short_ctx = strip_hex_leading_zeros(ctx)
                    short_addrs = [strip_hex_leading_zeros(a) for a in addrs]

                    writer.writerow([
                        f"{now:.6f}",
                        short_ctx,
                        launch_id,
                        kid,
                        cta_id,
                        warp_id,
                        opcode,
                        kind if kind is not None else "",
                        space if space is not None else "",
                        nbytes_s if nbytes_s is not None else "",
                        wsrc if wsrc is not None else "",
                        active_threads,
                        active_threads * nbytes,
                        unique_cache_lines,
                        unique_sectors,
                        min_vaddr,
                        max_vaddr,
                        max_vaddr_end,
                        *short_addrs,
                    ])
                    rows_written += 1

                    # Flush about once a second so a SIGKILL loses at most ~1s of rows
                    if now - last_flush >= 1.0:
                        f.flush()
                        last_flush = now

    finally:
        if inv_file is not None:
            inv_file.close()
        if timer is not None:
            timer.cancel()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        process.wait()

    # Write separate kernel lookup table mapping kernel_id -> kernel_name
    with open(kernels_path, "w", newline="") as kf:
        kwriter = csv.writer(kf)
        kwriter.writerow(["kernel_id", "kernel_name"])
        for kname, kid in sorted(kernel_name_to_id.items(), key=lambda item: item[1]):
            kwriter.writerow([kid, kname])

    if cpu_fallback.is_set():
        status = "aborted: no usable GPU (ran on CPU)"
    elif timed_out.is_set():
        status = "timed out"
    elif interrupted.is_set():
        status = "interrupted by user"
    else:
        status = f"exited with code {process.returncode}"

    print(
        f"[COMPLETE] Target {status}. Saw {len(launch_id_to_kernel)} kernel launches, "
        f"saved {rows_written} sampled warp accesses to '{output_csv}' "
        f"({len(kernel_name_to_id)} kernels in '{kernels_path}').",
        flush=True,
    )

    if inventory:
        print(f"[INVENTORY] {len(inventory)} distinct memory opcodes seen; details in '{inv_path}'.", flush=True)
        missed = sorted(op for op, (dec, _, _) in inventory.items() if dec in NOT_TRACED_VRAM)
        if missed:
            print("[WARNING] VRAM-touching opcodes that are NOT traced (no usable address operand): "
                  + ", ".join(missed), flush=True)
        unk = sorted(op for op, (dec, _, src) in inventory.items() if dec == "TRACED" and src == "unknown")
        if unk:
            print("[WARNING] traced opcodes with unknown payload width: " + ", ".join(unk), flush=True)
    if rows_written and rows_new_format == 0:
        print("[WARNING] No rows carried the new kind/space/bytes fields: mem_trace.so looks stale. "
              "Rebuild it from the updated mem_trace.cu / inject_funcs.cu / common.h.", flush=True)

    if cpu_fallback.is_set():
        print(
            "[ERROR] The GPU backend was not loaded. Check --cuda-dir, that the model is "
            "offloaded (e.g. -ngl 99), and run the target once without NVBit to confirm.",
            flush=True,
        )
        return EXIT_NO_GPU
    if not launch_id_to_kernel:
        print(
            "[WARNING] 0 kernel launches seen - the GPU was probably not used, or NVBit did "
            "not hook the CUDA driver. Re-run with TOOL_VERBOSE=1 to debug.",
            flush=True,
        )
        return EXIT_NO_LAUNCHES
    return EXIT_OK


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NVBit GPU Virtual Memory Address Access Tracer")
    parser.add_argument("-o", "--output", type=str, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument(
        "-c", "--period", type=int, default=10007,
        help="GPU-side warp instruction sampling period (prime recommended, default: 10007)"
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=None,
        help="Max duration in seconds before terminating target (default: run until completion)"
    )
    parser.add_argument(
        "--start-delay", type=float, default=0.0,
        help="Seconds to wait before recording traces (useful for skipping model load)"
    )
    parser.add_argument(
        "--start-after-file", type=str, default=None,
        help="Do not record accesses until this file exists (the orchestrator creates it "
             "when the server is ready, so model load/warmup is skipped)"
    )
    parser.add_argument("-k", "--kernel", type=str, default=None, help="Filter by kernel name substring")
    parser.add_argument("--so-path", type=str, default=DEFAULT_SO, help="Path to NVBit mem_trace.so")
    parser.add_argument(
        "--cuda-dir", type=str, default=DEFAULT_CUDA_DIR,
        help=f"Directory with libggml-cuda.so (default: {DEFAULT_CUDA_DIR}). "
             "Pass '' to skip the ollama library-path setup for non-ollama workloads."
    )
    parser.add_argument(
        "cmd", nargs=argparse.REMAINDER,
        help="Target command to trace (prefix with '--', e.g., -- python3 infer.py)"
    )

    args = parser.parse_args()

    target_cmd = args.cmd[1:] if (args.cmd and args.cmd[0] == "--") else args.cmd
    if not target_cmd:
        target_cmd = DEFAULT_CMD

    sys.exit(run_gpu_vm_trace(
        target_cmd=target_cmd,
        output_csv=args.output,
        sample_period=args.period,
        duration=args.duration,
        start_delay=args.start_delay,
        kernel_filter=args.kernel,
        mem_trace_so=args.so_path,
        cuda_dir=args.cuda_dir,
        start_after_file=args.start_after_file,
    ))
