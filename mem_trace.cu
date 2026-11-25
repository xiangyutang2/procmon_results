/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <cstring>
#include <initializer_list>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

// #define USE_ASYNC_STREAM
// NOTE: USE_ASYNC_STREAM version of channel.hpp can cause deadlock if two
// kernels have some dependency, since after each kernel launch, a device
// synchronization is needed for USE_ASYNC_STREAM version. Otherwise, the async
// copy in ChannelHost->recv() and a CUDA API call following a kernel launch
// can cause a deadlock, where the CUDA API call can take the CUDA context lock
// and waits for the kernel to finish, but the async copy in ChannelHost->recv()
// is waiting for the aforementioned CUDA context lock, thus stalling the kernel
// to finish (the instrumented kernel now needs to flush channel bufferto host
// side to finish by using the async copy).

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

#define HEX(x)                                                            \
    "0x" << std::setfill('0') << std::setw(16) << std::hex << (uint64_t)x \
         << std::dec

#define CHANNEL_SIZE (1l << 20)

enum class RecvThreadState {
    INIT,
    WORKING,
    STOP,
    FINISHED,
};

struct CTXstate {
    /* context id */
    int id;

    /* Channel used to communicate from GPU to CPU receiving thread */
    ChannelDev* channel_dev = nullptr;
    ChannelHost channel_host;

    /* tool module */
    CUmodule tool_module;

    /* flush channel function */
    CUfunction flush_channel_func;

    // Start with INIT, so that if no kernel is launched in the ctx, there is
    // no need to wait on the thread at the context termination.
    // After initialization, set it to WORKING to make recv thread get data,
    // parent thread sets it to STOP to make recv thread stop working.
    // recv thread sets it to FINISHED when it cleans up.
    // parent thread should wait until the state becomes FINISHED to clean up.
    volatile RecvThreadState recv_thread_done = RecvThreadState::INIT;
    // whether the context and the channel need a synchronization.
    bool need_sync = false;
};

#include "tool_func/flush_channel.c"

/* lock */
std::recursive_mutex mutex;
std::recursive_mutex cuda_event_mutex;

/* map to store context state */
std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
uint32_t sample_period = 10007;
/* print a one-time opcode inventory (MEMTRACE_INV lines) */
int inventory_enabled = 1;

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

/* grid launch id, incremented at every launch */
uint64_t global_grid_launch_id = 0;

void* recv_thread_fun(void* args);

void nvbit_at_init() {
    setvbuf(stdout, NULL, _IOLBF, 0);
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    GET_VAR_INT(sample_period, "SAMPLE_PERIOD", 10007,
            "Sample 1 out of every N warp memory instructions (use a prime number)");
    GET_VAR_INT(inventory_enabled, "MEMTRACE_INVENTORY", 1,
                "Print every distinct memory opcode seen with its classification");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

/* ===================== BEGIN opcode classification ===================== */
/* Decides which memory instructions access VRAM, which operand holds the VRAM
 * address, and how many bytes each thread moves. Kept free of tool state so it
 * can be unit-tested on its own. */

enum OpKind {
    OPK_LOAD = 0,
    OPK_STORE = 1,
    OPK_ATOMIC = 2,
    OPK_ASYNC_COPY = 3, /* cp.async: global -> shared; we record the global source */
    OPK_OTHER = 4
};
static const char* const OpKindStr[] = {"LOAD", "STORE", "ATOMIC",
                                        "ASYNC_COPY", "OTHER"};

enum WidthSrc {
    WSRC_UNKNOWN = 0,   /* width could not be determined (bytes = 0) */
    WSRC_OPCODE = 1,    /* explicit width suffix in the opcode (.64, .128, .U8, ...) */
    WSRC_DEFAULT32 = 2, /* no suffix: SASS default of 32 bits (assumed) */
    WSRC_MREF = 3       /* NVBit MREF operand nbytes */
};
static const char* const WidthSrcStr[] = {"unknown", "opcode", "default32",
                                          "mref"};

enum Decision {
    DEC_TRACE = 0,
    DEC_SKIP_SHARED,      /* shared memory: not VRAM */
    DEC_SKIP_CONSTANT,    /* constant bank */
    DEC_SKIP_NODATA,      /* fences, barriers, cache-control / prefetch hints */
    DEC_UNSUPPORTED_TEX,  /* texture/surface: no plain address operand */
    DEC_UNSUPPORTED_TMA,  /* TMA: needs the tensor-map path (see mem_trace_tma) */
    DEC_NO_MREF,          /* looks like a VRAM access but has no memory-reference operand */
    DEC_SKIP_OTHER
};
static const char* const DecisionStr[] = {
    "TRACED",          "SKIP_SHARED",     "SKIP_CONSTANT",
    "SKIP_NODATA",     "UNSUPPORTED_TEX", "UNSUPPORTED_TMA",
    "NO_MREF",         "SKIP_OTHER"};

struct OpInfo {
    Decision decision;
    OpKind kind;
    bool allowlisted; /* true: explicit opcode list; false: generic fallback rule */
};

static bool in_list(const char* s, std::initializer_list<const char*> list) {
    for (const char* e : list) {
        if (strcmp(s, e) == 0) return true;
    }
    return false;
}

/* Explicit list of SASS opcodes (short form) that read/write VRAM. */
static bool allowlist_kind(const char* base, OpKind* kind) {
    if (in_list(base, {"LDG", "LD", "LDL"})) { *kind = OPK_LOAD; return true; }
    if (in_list(base, {"STG", "ST", "STL"})) { *kind = OPK_STORE; return true; }
    if (in_list(base, {"ATOMG", "ATOM", "RED"})) { *kind = OPK_ATOMIC; return true; }
    if (in_list(base, {"LDGSTS"})) { *kind = OPK_ASYNC_COPY; return true; }
    return false;
}

static OpInfo classify_instr(const char* base, InstrType::MemorySpace space,
                             bool is_load, bool is_store, bool is_tma,
                             int nmref) {
    using MS = InstrType::MemorySpace;
    OpInfo r = {DEC_SKIP_OTHER, OPK_OTHER, false};

    if (is_tma || strncmp(base, "UTMA", 4) == 0 || strncmp(base, "UBLK", 4) == 0) {
        r.decision = DEC_UNSUPPORTED_TMA;
        return r;
    }

    /* explicit list wins over whatever memory space NVBit reports (e.g. the
     * non-coherent LDG.E.CONSTANT must not be mistaken for a constant-bank read) */
    OpKind k;
    if (allowlist_kind(base, &k)) {
        r.allowlisted = true;
        r.kind = k;
        r.decision = nmref > 0 ? DEC_TRACE : DEC_NO_MREF;
        return r;
    }

    if (in_list(base, {"LDS", "STS", "LDSM", "STSM", "ATOMS"}) ||
        space == MS::SHARED || space == MS::DISTRIBUTED_SHARED) {
        r.decision = DEC_SKIP_SHARED;
        return r;
    }
    if (in_list(base, {"CCTL", "CCTLL", "CCTLT", "LDGDEPBAR", "DEPBAR", "MEMBAR",
                       "ERRBAR", "FENCE", "PREFETCH", "PREFETCHU", "QSPC"})) {
        r.decision = DEC_SKIP_NODATA;
        return r;
    }
    if (in_list(base, {"TEX", "TLD", "TLD4", "TLDS", "TMML", "TXD", "TXQ", "SULD",
                       "SUST", "SURED", "SUATOM"}) ||
        space == MS::SURFACE || space == MS::TEXTURE) {
        r.decision = DEC_UNSUPPORTED_TEX;
        return r;
    }
    if (space == MS::CONSTANT) {
        r.decision = DEC_SKIP_CONSTANT;
        return r;
    }
    /* generic rule for opcodes not on the list but reported as VRAM-touching */
    if (space == MS::GLOBAL || space == MS::GENERIC || space == MS::LOCAL ||
        space == MS::GLOBAL_TO_SHARED) {
        r.kind = (space == MS::GLOBAL_TO_SHARED) ? OPK_ASYNC_COPY
                 : is_load                        ? OPK_LOAD
                 : is_store                       ? OPK_STORE
                                                  : OPK_OTHER;
        r.decision = nmref > 0 ? DEC_TRACE : DEC_NO_MREF;
        return r;
    }
    return r; /* DEC_SKIP_OTHER */
}

/* bytes moved per thread for one width token of an opcode, 0 if not a width */
static uint32_t width_token_bytes(const std::string& t) {
    if (t == "U8" || t == "S8") return 1;
    if (t == "U16" || t == "S16" || t == "F16" || t == "BF16") return 2;
    if (t == "32" || t == "U32" || t == "S32" || t == "F32" || t == "F16x2" ||
        t == "BF16x2" || t == "F16X2" || t == "BF16X2")
        return 4;
    if (t == "64" || t == "U64" || t == "S64" || t == "F64") return 8;
    if (t == "128") return 16;
    return 0; /* E, CONSTANT, STRONG, GPU, BYPASS, LTC128B, ADD, ... */
}

struct WidthInfo {
    uint32_t bytes;
    WidthSrc src;
};

static WidthInfo parse_opcode_width(const char* opcode, OpKind kind) {
    const std::string s(opcode);
    size_t pos = s.find('.');
    while (pos != std::string::npos) {
        const size_t next = s.find('.', pos + 1);
        const std::string tok =
            s.substr(pos + 1, next == std::string::npos ? std::string::npos
                                                        : next - pos - 1);
        const uint32_t b = width_token_bytes(tok);
        if (b) return {b, WSRC_OPCODE};
        pos = next;
    }
    /* no explicit width: SASS loads/stores default to 32 bits */
    if (kind != OPK_OTHER) return {4, WSRC_DEFAULT32};
    return {0, WSRC_UNKNOWN};
}

struct MrefPick {
    int mref_id;   /* index among the MREF operands (for nvbit_add_call_arg_mref_addr64) */
    int op_idx;    /* operand index in the instruction */
    int nmref;     /* number of MREF operands */
    int op_nbytes; /* operand_t.nbytes of the chosen operand, -1 if none */
};

/* Picks the operand holding the VRAM address. Normal instructions have one MREF.
 * cp.async (LDGSTS) has two: shared destination first, global source second; we
 * prefer the operand with a memory descriptor, else the last one. */
static MrefPick pick_mref(Instr* instr, bool prefer_global_source) {
    MrefPick p = {-1, -1, 0, -1};
    int first_id = -1, first_op = -1, last_id = -1, last_op = -1;
    int desc_id = -1, desc_op = -1;
    int n = 0;
    for (int i = 0; i < instr->getNumOperands(); i++) {
        const InstrType::operand_t* op = instr->getOperand(i);
        if (op->type != InstrType::OperandType::MREF) continue;
        if (first_id < 0) { first_id = n; first_op = i; }
        last_id = n;
        last_op = i;
        if (op->u.mref.has_desc && desc_id < 0) { desc_id = n; desc_op = i; }
        n++;
    }
    p.nmref = n;
    if (n == 0) return p;
    if (prefer_global_source) {
        p.mref_id = desc_id >= 0 ? desc_id : last_id;
        p.op_idx = desc_id >= 0 ? desc_op : last_op;
    } else {
        p.mref_id = first_id;
        p.op_idx = first_op;
    }
    p.op_nbytes = instr->getOperand(p.op_idx)->nbytes;
    return p;
}

/* Prefer NVBit's MREF operand byte width when positive; fall back to parsing
 * width tokens from the opcode string (and 32-bit default for non-OTHER ops). */
static WidthInfo resolve_width(const char* opcode, OpKind kind,
                               const MrefPick& pick) {
    if (pick.op_nbytes > 0) {
        return {(uint32_t)pick.op_nbytes, WSRC_MREF};
    }
    return parse_opcode_width(opcode, kind);
}

/* Runtime opcode inventory: one line per distinct (opcode, decision), plus
 * static instruction counts printed at context termination. */
/* inventory_enabled is declared with the other global control variables */
struct InvEntry {
    std::string opcode;
    Decision decision;
    bool allowlisted;
    uint64_t count;
};
static std::map<std::string, InvEntry> inventory;

static void inventory_record(Instr* instr, const char* opcode, const char* base,
                             InstrType::MemorySpace space, const OpInfo& info,
                             const WidthInfo& w, const MrefPick& pick) {
    if (!inventory_enabled) return;
    const std::string key = std::string(opcode) + "#" +
                            std::to_string((int)info.decision) + "#" +
                            (info.allowlisted ? "list" : "fallback");
    auto it = inventory.find(key);
    if (it != inventory.end()) {
        it->second.count++;
        return;
    }
    inventory[key] = {opcode, info.decision, info.allowlisted, 1};
    const int sp = (int)space;
    const char* space_str =
        (sp >= 0 && sp < (int)(sizeof(InstrType::MemorySpaceStr) /
                               sizeof(InstrType::MemorySpaceStr[0])))
            ? InstrType::MemorySpaceStr[sp]
            : "?";
    printf(
        "MEMTRACE_INV: opcode=%s base=%s space=%s decision=%s rule=%s kind=%s "
        "width=%u(%s) getSize=%d nmref=%d mref_op_nbytes=%d load=%d store=%d "
        "sass=%s\n",
        opcode, base, space_str, DecisionStr[info.decision],
        info.allowlisted ? "list" : "fallback", OpKindStr[info.kind], w.bytes,
        WidthSrcStr[w.src], instr->getSize(), pick.nmref, pick.op_nbytes,
        (int)instr->isLoad(), (int)instr->isStore(), instr->getSass());
}

static void inventory_print_counts() {
    if (!inventory_enabled) return;
    for (auto& kv : inventory) {
        printf(
            "MEMTRACE_INV_COUNT: opcode=%s decision=%s rule=%s static_instrs=%llu\n",
            kv.second.opcode.c_str(), DecisionStr[kv.second.decision],
            kv.second.allowlisted ? "list" : "fallback",
            (unsigned long long)kv.second.count);
    }
    fflush(stdout);
}

/* Classifies one static instruction and, if it accesses VRAM, injects the
 * tracing call on the operand holding the VRAM address. Returns true if
 * instrumented. */
static bool instrument_mem_instr(Instr* instr, uint64_t channel_dev) {
    const InstrType::MemorySpace space = instr->getMemorySpace();
    const char* base = instr->getOpcodeShort();
    const bool is_tma = instr->isTMAMem() || instr->isTMATensor();

    /* quick reject: not a memory instruction at all */
    OpKind dummy;
    if (space == InstrType::MemorySpace::NONE && !is_tma &&
        !allowlist_kind(base, &dummy)) {
        return false;
    }

    const bool prefer_global_source =
        (strcmp(base, "LDGSTS") == 0) ||
        space == InstrType::MemorySpace::GLOBAL_TO_SHARED;
    const MrefPick pick = pick_mref(instr, prefer_global_source);
    const OpInfo info = classify_instr(base, space, instr->isLoad(),
                                       instr->isStore(), is_tma, pick.nmref);
    const char* opcode = instr->getOpcode();
    const WidthInfo w = resolve_width(opcode, info.kind, pick);

    inventory_record(instr, opcode, base, space, info, w, pick);
    if (info.decision != DEC_TRACE) return false;

    if (opcode_to_id_map.find(opcode) == opcode_to_id_map.end()) {
        int new_id = (int)opcode_to_id_map.size();
        opcode_to_id_map[opcode] = new_id;
        id_to_opcode_map[new_id] = std::string(opcode);
    }
    const int opcode_id = opcode_to_id_map[opcode];
    const uint32_t meta = (uint32_t)info.kind | ((uint32_t)w.src << 8) |
                          (((uint32_t)space & 0xff) << 16);

    /* argument order must match instrument_mem() in inject_funcs.cu */
    nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
    nvbit_add_call_arg_guard_pred_val(instr);             /* int pred */
    nvbit_add_call_arg_const_val32(instr, opcode_id);     /* int opcode_id */
    nvbit_add_call_arg_const_val32(instr, sample_period); /* uint32_t sample_period */
    nvbit_add_call_arg_const_val32(instr, w.bytes);       /* uint32_t bytes_per_thread */
    nvbit_add_call_arg_const_val32(instr, meta);          /* uint32_t meta: kind|wsrc<<8|space<<16 */
    nvbit_add_call_arg_mref_addr64(instr, pick.mref_id);  /* uint64_t addr */
    nvbit_add_call_arg_launch_val64(instr, 0);            /* uint64_t grid_launch_id */
    nvbit_add_call_arg_const_val64(instr, channel_dev);   /* uint64_t pchannel_dev */
    return true;
}
/* ====================== END opcode classification ====================== */

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        /* get vector of instructions of function "f" */
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);

        if (verbose) {
            printf(
                "MEMTRACE: CTX %p, Inspecting CUfunction %p name %s at address "
                "0x%lx\n",
                ctx, f, nvbit_get_func_name(ctx, f),
                nvbit_get_func_addr(ctx, f));
        }

        uint32_t cnt = 0;
        /* iterate on all the static instructions in the function */
        for (auto instr : instrs) {
            if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
                cnt++;
                continue;
            }
            /* classify the instruction; if it accesses VRAM, inject the tracing
             * call on the operand that holds the VRAM address */
            if (instrument_mem_instr(instr, (uint64_t)ctx_state->channel_dev) &&
                verbose) {
                instr->printDecoded();
            }
            cnt++;
        }
    }
}

void init_context_state(CUcontext ctx) {
    CTXstate* ctx_state = ctx_state_map[ctx];
    ctx_state->recv_thread_done = RecvThreadState::WORKING;
    CUDA_SAFECALL(
        cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev)));
    ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
                                 ctx_state->channel_dev, recv_thread_fun, ctx);
    nvbit_set_tool_thread(ctx_state->channel_host.get_thread_id());
}

static void enter_kernel_launch(CUcontext ctx, CUfunction func,
                                uint64_t& grid_launch_id, nvbit_api_cuda_t cbid,
                                void* params, bool stream_capture = false,
                                bool build_graph = false) {
    /* instrument */
    instrument_function_if_needed(ctx, func);

    int nregs = 0;
    CUDA_SAFECALL(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

    int shmem_static_nbytes = 0;
    CUDA_SAFECALL(cuFuncGetAttribute(
        &shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));

    /* get function name and pc */
    const char* func_name = nvbit_get_func_name(ctx, func);
    uint64_t pc = nvbit_get_func_addr(ctx, func);

    // during stream capture or manual graph build, no kernel is launched, so
    // do not set launch argument, do not print kernel info, do not increase
    // grid_launch_id. All these should be done at graph node launch time.
    if (!stream_capture && !build_graph) {
        /* set grid launch id at launch time */
        nvbit_set_at_launch(ctx, func, (uint64_t)grid_launch_id);

        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            printf(
                "MEMTRACE: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
                "Kernel name %s - grid launch id %ld - grid size %d,%d,%d "
                "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
                "id %ld\n",
                (uint64_t)ctx, pc, func_name, grid_launch_id,
                p->config->gridDimX, p->config->gridDimY, p->config->gridDimZ,
                p->config->blockDimX, p->config->blockDimY,
                p->config->blockDimZ, nregs,
                shmem_static_nbytes + p->config->sharedMemBytes,
                (uint64_t)p->config->hStream);
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            printf(
                "MEMTRACE: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
                "Kernel name %s - grid launch id %ld - grid size %d,%d,%d "
                "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
                "id %ld\n",
                (uint64_t)ctx, pc, func_name, grid_launch_id, p->gridDimX,
                p->gridDimY, p->gridDimZ, p->blockDimX, p->blockDimY,
                p->blockDimZ, nregs, shmem_static_nbytes + p->sharedMemBytes,
                (uint64_t)p->hStream);
        }
        fflush(stdout);
        // increment grid launch id for next launch
        // grid id can be changed here, since nvbit_set_at_launch() has copied
        // its value above.
        grid_launch_id++;
    }

    /* enable instrumented code to run */
    nvbit_enable_instrumented(ctx, func, true);
}

static void leave_kernel_launch(CUcontext ctx, CTXstate* ctx_state) {
#ifdef USE_ASYNC_STREAM
    // make sure user kernel finishes to avoid deadlock
    CUDA_SAFECALL(cudaDeviceSynchronize());
    /* push a flush channel kernel */
    void* args[] = {&ctx_state->channel_dev};
    nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1, 1, 0,
                        nullptr, args, nullptr);

    /* Make sure GPU is idle */
    CUDA_SAFECALL(cudaDeviceSynchronize());
#endif
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    cuda_event_mutex.lock();

    /* we prevent re-entry on this callback when issuing CUDA functions inside
     * this function */
    if (skip_callback_flag) {
        cuda_event_mutex.unlock();
        return;
    }
    skip_callback_flag = true;

    /* Skip callbacks for contexts not yet initialized in ctx_state_map
     * (e.g. green-context-derived CUcontexts where nvbit_at_ctx_init has
     * not yet run). Using operator[] on an unknown key would insert a null
     * CTXstate* and cause a segfault when dereferenced below. */
    if (ctx_state_map.find(ctx) == ctx_state_map.end()) {
        skip_callback_flag = false;
        cuda_event_mutex.unlock();
        return;
    }

    CTXstate* ctx_state = ctx_state_map[ctx];

    switch (cbid) {
		case API_CUDA_cuGraphLaunch: {
            if (!is_exit) {
                ctx_state->need_sync = true;
                if (verbose >= 1) {
                    printf("MEMTRACE: CTX 0x%016lx - GRAPHLAUNCH\n", (uint64_t)ctx);
                    fflush(stdout);
                }
            } else {
                leave_kernel_launch(ctx, ctx_state);
            }
        } break;
        // Identify all the possible CUDA launch events without stream
        // parameters, they will not get involved with cuda graph
        case API_CUDA_cuLaunch:
        case API_CUDA_cuLaunchGrid: {
            cuLaunch_params* p = (cuLaunch_params*)params;
            CUfunction func = p->f;
            if (!is_exit) {
                ctx_state->need_sync = true;
                enter_kernel_launch(ctx, func, global_grid_launch_id, cbid,
                                    params);
            } else {
                leave_kernel_launch(ctx, ctx_state);
            }
        } break;
        // To support kernel launched by cuda graph (in addition to existing
        // kernel launche method), we need to do:
        //
        // 1. instrument kernels at cudaGraphAddKernelNode event. This is for
        // cases that kernels are manually added to a cuda graph.
        // 2. distinguish captured kernels when kernels are recorded to a graph
        // using stream capture. cudaStreamIsCapturing() tells us whether a
        // stream is capturiong.
        // 3. per-kernel instruction counters, since cuda graph can launch
        // multiple kernels at the same time.
        //
        // Three cases:
        //
        // 1. original kernel launch:
        //     1a. for any kernel launch without using a stream, we instrument
        //     it before it is launched, call cudaDeviceSynchronize after it is
        //     launched and read the instruction counter of the kernel.
        //     1b. for any kernel launch using a stream, but the stream is not
        //     capturing, we do the same thing as 1a.
        //
        //  2. cuda graph using stream capturing: if a kernel is launched in a
        //  stream and the stream is capturing. We instrument the kernel before
        //  it is launched and do nothing after it is launched, because the
        //  kernel is not running until cudaGraphLaunch. Instead, we issue a
        //  cudaStreamSynchronize after cudaGraphLaunch is done and reset the
        //  instruction counters, since a cloned graph might be launched
        //  afterwards.
        //
        //  3. cuda graph manual: we instrument the kernel added by
        //  cudaGraphAddKernelNode and do the same thing for cudaGraphLaunch
        //  as 2.
        //
        // The above method should handle most of cuda graph launch cases.
        // kernel launches with stream parameter, they can be used for cuda
        // graph
        case API_CUDA_cuLaunchKernel_ptsz:
        case API_CUDA_cuLaunchKernel:
        case API_CUDA_cuLaunchCooperativeKernel:
        case API_CUDA_cuLaunchCooperativeKernel_ptsz:
        case API_CUDA_cuLaunchKernelEx:
        case API_CUDA_cuLaunchKernelEx_ptsz:
        case API_CUDA_cuLaunchGridAsync: {
            CUfunction func;
            CUstream hStream;

            if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                cbid == API_CUDA_cuLaunchKernelEx) {
                cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
                func = p->f;
                hStream = p->config->hStream;
            } else if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                       cbid == API_CUDA_cuLaunchKernel ||
                       cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz ||
                       cbid == API_CUDA_cuLaunchCooperativeKernel) {
                cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
                func = p->f;
                hStream = p->hStream;
            } else {
                cuLaunchGridAsync_params* p = (cuLaunchGridAsync_params*)params;
                func = p->f;
                hStream = p->hStream;
            }

            cudaStreamCaptureStatus streamStatus;
            /* check if the stream is capturing, if yes, do not sync */
            CUDA_SAFECALL(cudaStreamIsCapturing(hStream, &streamStatus));
            if (!is_exit) {
                bool stream_capture =
                    (streamStatus == cudaStreamCaptureStatusActive);
                ctx_state->need_sync = true;
                enter_kernel_launch(ctx, func, global_grid_launch_id, cbid,
                                    params, stream_capture);
            } else {
                if (streamStatus != cudaStreamCaptureStatusActive) {
                    if (verbose >= 1) {
                        printf("kernel %s not captured by cuda graph\n",
                               nvbit_get_func_name(ctx, func));
                    }
                    leave_kernel_launch(ctx, ctx_state);
                } else {
                    if (verbose >= 1) {
                        printf("kernel %s captured by cuda graph\n",
                               nvbit_get_func_name(ctx, func));
                    }
                }
            }
        } break;
        case API_CUDA_cuGraphAddKernelNode: {
            cuGraphAddKernelNode_params* p =
                (cuGraphAddKernelNode_params*)params;
            CUfunction func = p->nodeParams->func;

            if (!is_exit) {
                // cuGraphAddKernelNode_params->nodeParams is the same as
                // cuLaunchKernel_params up to sharedMemBytes
                ctx_state->need_sync = true;
                enter_kernel_launch(ctx, func, global_grid_launch_id, cbid,
                                    (void*)p->nodeParams, false, true);
            }
        } break;
        default:
            break;
    };

    skip_callback_flag = false;
    cuda_event_mutex.unlock();
}

void* recv_thread_fun(void* args) {
    CUcontext ctx = (CUcontext)args;

    mutex.lock();
    /* get context state from map */
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    ChannelHost* ch_host = &ctx_state->channel_host;
    mutex.unlock();
    char* recv_buffer = (char*)malloc(CHANNEL_SIZE);

    while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
        /* receive buffer from channel */
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
        if (num_recv_bytes > 0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                mem_access_t* ma =
                    (mem_access_t*)&recv_buffer[num_processed_bytes];

                std::stringstream ss;
                ss << "CTX " << HEX(ctx) << " - grid_launch_id "
                   << ma->grid_launch_id << " - CTA " << ma->cta_id_x << ","
                   << ma->cta_id_y << "," << ma->cta_id_z << " - warp "
                   << ma->warp_id << " - " << id_to_opcode_map[ma->opcode_id]
                   << " - kind " << OpKindStr[ma->op_kind <= OPK_OTHER ? ma->op_kind : OPK_OTHER]
                   << " - space "
                   << (ma->mem_space < sizeof(InstrType::MemorySpaceStr) /
                                           sizeof(InstrType::MemorySpaceStr[0])
                           ? InstrType::MemorySpaceStr[ma->mem_space]
                           : "?")
                   << " - bytes " << ma->bytes_per_thread << " - wsrc "
                   << WidthSrcStr[ma->width_src <= WSRC_MREF ? ma->width_src : WSRC_UNKNOWN]
                   << " - ";

                for (int i = 0; i < 32; i++) {
                    ss << HEX(ma->addrs[i]) << " ";
                }

                printf("MEMTRACE: %s\n", ss.str().c_str());
                num_processed_bytes += sizeof(mem_access_t);
            }
            fflush(stdout);
        }
    }
    free(recv_buffer);
    ctx_state->recv_thread_done = RecvThreadState::FINISHED;
    return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
    mutex.lock();
    if (verbose) {
        printf("MEMTRACE: STARTING CONTEXT %p\n", ctx);
    }
    assert(ctx_state_map.find(ctx) == ctx_state_map.end());
    CTXstate* ctx_state = new CTXstate;
    ctx_state_map[ctx] = ctx_state;

    nvbit_load_tool_module(ctx, (const void*)flush_channel_bin,
                           &ctx_state->tool_module);
    nvbit_find_function_by_name(ctx, ctx_state->tool_module, "flush_channel",
                                &ctx_state->flush_channel_func);

    mutex.unlock();
}

void nvbit_tool_init(CUcontext ctx) {
    mutex.lock();
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    init_context_state(ctx);
    mutex.unlock();
}

void nvbit_at_ctx_term(CUcontext ctx) {
    mutex.lock();
    skip_callback_flag = true;
    if (verbose) {
        printf("MEMTRACE: TERMINATING CONTEXT %p\n", ctx);
    }
    /* get context state from map */
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* once per process: static instruction counts per (opcode, decision) */
    static bool inventory_counts_printed = false;
    if (!inventory_counts_printed) {
        inventory_counts_printed = true;
        inventory_print_counts();
    }

    // flush channel if there is a kernel launch before context termination
    // without a device synchronization.
    if (ctx_state->need_sync) {
        void* args[] = {&ctx_state->channel_dev};
        nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1,
                            1, 0, nullptr, args, nullptr);
        CUDA_SAFECALL(cudaDeviceSynchronize());
    }

    /* Notify receiver thread and wait for receiver thread to
     * notify back */
    if (ctx_state->recv_thread_done != RecvThreadState::INIT) {
        ctx_state->recv_thread_done = RecvThreadState::STOP;
        while (ctx_state->recv_thread_done != RecvThreadState::FINISHED);
    }

    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
    skip_callback_flag = false;
    delete ctx_state;
    ctx_state_map.erase(ctx);
    mutex.unlock();
}

void nvbit_at_graph_node_launch(CUcontext ctx, CUfunction func, CUstream stream,
                                uint64_t launch_handle) {
    func_config_t config = {0};
    const char* func_name = nvbit_get_func_name(ctx, func);
    uint64_t pc = nvbit_get_func_addr(ctx, func);

    mutex.lock();
    nvbit_set_at_launch(ctx, func, (uint64_t)global_grid_launch_id, stream,
                        launch_handle);
    nvbit_get_func_config(ctx, func, &config);

    printf(
        "MEMTRACE: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
        "Kernel name %s - grid launch id %ld - grid size %d,%d,%d "
        "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
        "id %ld\n",
        (uint64_t)ctx, pc, func_name, global_grid_launch_id, config.gridDimX,
        config.gridDimY, config.gridDimZ, config.blockDimX, config.blockDimY,
        config.blockDimZ, config.num_registers,
        config.shmem_static_nbytes + config.shmem_dynamic_nbytes,
        (uint64_t)stream);
    fflush(stdout);
    // grid id can be changed here, since nvbit_set_at_launch() has copied its
    // value above.
    global_grid_launch_id++;
    mutex.unlock();
}