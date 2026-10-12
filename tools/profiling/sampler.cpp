// Out-of-process sampling profiler for one thread of a running process (dbghelp stack walks).
// sampler.exe <pid> <seconds> <out.txt> [interval_ms=1] [tid=busiest]
// Each sample suspends the thread only long enough to copy its context and stack; the walk and
// symbolization run afterwards. Output: "count frame;frame;...;leaf" lines, most frequent first.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>
#include <tlhelp32.h>
#include <timeapi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "winmm.lib")

namespace {

struct Captured {
	CONTEXT              context;
	uint64_t             stack_base = 0;
	std::vector<uint8_t> stack;
};

HANDLE          g_process = nullptr;
const Captured* g_current = nullptr;

BOOL CALLBACK ReadMemory(HANDLE process, DWORD64 address, PVOID buffer, DWORD size, LPDWORD read) {
	if (g_current != nullptr && address >= g_current->stack_base &&
	    address + size <= g_current->stack_base + g_current->stack.size()) {
		std::memcpy(buffer, g_current->stack.data() + (address - g_current->stack_base), size);
		*read = size;
		return TRUE;
	}
	SIZE_T done = 0;
	const BOOL ok = ReadProcessMemory(process, reinterpret_cast<LPCVOID>(address), buffer, size, &done);
	*read = static_cast<DWORD>(done);
	return ok;
}

uint64_t ThreadCpu(HANDLE thread) {
	FILETIME c, e, k, u;
	if (!GetThreadTimes(thread, &c, &e, &k, &u)) return 0;
	return ((static_cast<uint64_t>(k.dwHighDateTime) << 32u) | k.dwLowDateTime) +
	       ((static_cast<uint64_t>(u.dwHighDateTime) << 32u) | u.dwLowDateTime);
}

DWORD BusiestThread(DWORD pid, size_t rank = 0) {
	std::vector<std::pair<DWORD, HANDLE>> threads;
	HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	THREADENTRY32 entry {sizeof(entry)};
	for (BOOL ok = Thread32First(snap, &entry); ok; ok = Thread32Next(snap, &entry)) {
		if (entry.th32OwnerProcessID != pid) continue;
		if (HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID)) {
			threads.emplace_back(entry.th32ThreadID, h);
		}
	}
	CloseHandle(snap);
	std::vector<uint64_t> before;
	for (auto& t: threads) before.push_back(ThreadCpu(t.second));
	Sleep(1000);
	std::vector<std::pair<uint64_t, DWORD>> busy;
	for (size_t i = 0; i < threads.size(); i++) {
		busy.emplace_back(ThreadCpu(threads[i].second) - before[i], threads[i].first);
		CloseHandle(threads[i].second);
	}
	std::sort(busy.rbegin(), busy.rend());
	for (size_t i = 0; i < busy.size() && i < 5; i++) {
		std::fprintf(stderr, "thread %lu: %.0f%%\n", busy[i].second, busy[i].first / 1e5);
	}
	return rank < busy.size() ? busy[rank].second : 0;
}

std::string Name(DWORD64 address, DWORD inline_context) {
	alignas(SYMBOL_INFO) char storage[sizeof(SYMBOL_INFO) + 512];
	auto* symbol         = reinterpret_cast<SYMBOL_INFO*>(storage);
	symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
	symbol->MaxNameLen   = 511;
	DWORD64 displacement = 0;
	BOOL    ok           = FALSE;
	if (inline_context != 0) {
		ok = SymFromInlineContext(g_process, address, inline_context, &displacement, symbol);
	}
	if (!ok) ok = SymFromAddr(g_process, address, &displacement, symbol);
	IMAGEHLP_MODULE64 module {sizeof(module)};
	if (ok) {
		// Symbols outside the emulator get their module as a prefix (exports can mislead).
		if (SymGetModuleInfo64(g_process, address, &module) &&
		    _stricmp(module.ModuleName, "kyty_emulator") != 0) {
			return std::string(module.ModuleName) + "!" + symbol->Name;
		}
		return symbol->Name;
	}
	char text[96];
	if (SymGetModuleInfo64(g_process, address, &module)) {
		std::snprintf(text, sizeof(text), "%s+0x%llx", module.ModuleName,
		              static_cast<unsigned long long>(address - module.BaseOfImage));
	} else {
		std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(address));
	}
	return text;
}

} // namespace

int main(int argc, char** argv) {
	if (argc < 4) {
		std::fprintf(stderr, "sampler <pid> <seconds> <out.txt> [interval_ms] [tid]\n");
		return 2;
	}
	const DWORD pid      = std::strtoul(argv[1], nullptr, 10);
	const double seconds = std::atof(argv[2]);
	const DWORD interval = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 1;
	// tid: a thread id, or rN for the N-th busiest thread (r0 = busiest, the default).
	const bool  by_rank  = argc > 5 && argv[5][0] == 'r';
	DWORD       tid      = argc > 5 && !by_rank ? std::strtoul(argv[5], nullptr, 10) : 0;
	const auto  rank     = by_rank ? std::strtoul(argv[5] + 1, nullptr, 10) : 0;
	g_process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
	if (g_process == nullptr) {
		std::fprintf(stderr, "OpenProcess failed: %lu\n", GetLastError());
		return 1;
	}
	if (tid == 0) tid = BusiestThread(pid, rank);
	HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
	                           FALSE, tid);
	if (thread == nullptr) {
		std::fprintf(stderr, "OpenThread %lu failed: %lu\n", tid, GetLastError());
		return 1;
	}
	SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_FAIL_CRITICAL_ERRORS | SYMOPT_LOAD_LINES);
	if (!SymInitialize(g_process, nullptr, TRUE)) {
		std::fprintf(stderr, "SymInitialize failed: %lu\n", GetLastError());
		return 1;
	}
	timeBeginPeriod(1);
	std::map<std::vector<std::pair<DWORD64, DWORD>>, uint32_t> stacks;
	Captured captured;
	uint32_t samples = 0, failures = 0;
	const ULONGLONG end = GetTickCount64() + static_cast<ULONGLONG>(seconds * 1000.0);
	while (GetTickCount64() < end) {
		if (SuspendThread(thread) == static_cast<DWORD>(-1)) break;
		captured.context              = {};
		captured.context.ContextFlags = CONTEXT_FULL;
		bool ok = GetThreadContext(thread, &captured.context) != 0;
		if (ok) {
			MEMORY_BASIC_INFORMATION info {};
			const uint64_t sp = captured.context.Rsp;
			if (VirtualQueryEx(g_process, reinterpret_cast<LPCVOID>(sp), &info, sizeof(info)) != 0) {
				const uint64_t top = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
				const uint64_t size = std::min<uint64_t>(top - sp, 1u << 20u);
				captured.stack.resize(size);
				SIZE_T done = 0;
				ReadProcessMemory(g_process, reinterpret_cast<LPCVOID>(sp), captured.stack.data(), size, &done);
				captured.stack.resize(done);
				captured.stack_base = sp;
			} else {
				captured.stack.clear();
			}
		}
		ResumeThread(thread);
		if (!ok) {
			++failures;
			Sleep(interval);
			continue;
		}
		STACKFRAME_EX frame {};
		frame.StackFrameSize   = sizeof(frame);
		frame.AddrPC.Offset    = captured.context.Rip;
		frame.AddrPC.Mode      = AddrModeFlat;
		frame.AddrFrame.Offset = captured.context.Rbp;
		frame.AddrFrame.Mode   = AddrModeFlat;
		frame.AddrStack.Offset = captured.context.Rsp;
		frame.AddrStack.Mode   = AddrModeFlat;
		CONTEXT walk_context   = captured.context;
		g_current              = &captured;
		std::vector<std::pair<DWORD64, DWORD>> frames;
		while (frames.size() < 96 &&
		       StackWalkEx(IMAGE_FILE_MACHINE_AMD64, g_process, thread, &frame, &walk_context, ReadMemory,
		                   SymFunctionTableAccess64, SymGetModuleBase64, nullptr, SYM_STKWALK_DEFAULT)) {
			if (frame.AddrPC.Offset == 0) break;
			frames.emplace_back(frame.AddrPC.Offset, frame.InlineFrameContext);
		}
		g_current = nullptr;
		if (frames.empty()) frames.emplace_back(captured.context.Rip, 0);
		++stacks[frames];
		++samples;
		Sleep(interval);
	}
	timeEndPeriod(1);
	std::fprintf(stderr, "samples %u failures %u unique %zu\n", samples, failures, stacks.size());
	// Symbolize unique frames once.
	std::map<std::pair<DWORD64, DWORD>, std::string> names;
	for (const auto& [frames, count]: stacks) {
		for (const auto& f: frames) names.emplace(f, std::string {});
	}
	for (auto& [f, name]: names) name = Name(f.first, f.second);
	// Leaf frames also get their source line (self time by line).
	std::map<std::pair<DWORD64, DWORD>, std::string> leaf_names;
	SymSetOptions(SymGetOptions() | SYMOPT_LOAD_LINES);
	for (const auto& [frames, count]: stacks) {
		const auto& f = frames.front();
		if (leaf_names.contains(f)) continue;
		IMAGEHLP_LINE64 line {sizeof(line)};
		DWORD           displacement = 0;
		BOOL ok = f.second != 0 ? SymGetLineFromInlineContext(g_process, f.first, f.second, 0, &displacement, &line)
		                        : FALSE;
		if (!ok) ok = SymGetLineFromAddr64(g_process, f.first, &displacement, &line);
		std::string name = names[f];
		if (ok && line.FileName != nullptr) {
			const char* file = std::strrchr(line.FileName, 0x5c);
			name += " @" + std::string(file != nullptr ? file + 1 : line.FileName) + ":" + std::to_string(line.LineNumber);
		}
		leaf_names[f] = name;
	}
	std::vector<std::pair<uint32_t, std::string>> lines;
	std::unordered_map<std::string, uint32_t> merged;
	for (const auto& [frames, count]: stacks) {
		std::string line;
		for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
			if (!line.empty()) line += ';';
			line += it + 1 == frames.rend() ? leaf_names[*it] : names[*it];
		}
		merged[line] += count;
	}
	for (auto& [line, count]: merged) lines.emplace_back(count, line);
	std::sort(lines.rbegin(), lines.rend());
	FILE* out = std::fopen(argv[3], "w");
	if (out == nullptr) return 1;
	std::fprintf(out, "# samples %u tid %lu\n", samples, tid);
	for (const auto& [count, line]: lines) std::fprintf(out, "%u %s\n", count, line.c_str());
	std::fclose(out);
	SymCleanup(g_process);
	return 0;
}
