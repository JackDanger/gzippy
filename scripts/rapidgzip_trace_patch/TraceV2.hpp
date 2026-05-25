#pragma once
/**
 * Chrome trace JSON instrumentation matching gzippy/src/decompress/parallel/trace_v2.rs.
 * Activated by env GZIPPY_TIMELINE. pid=2 for rapidgzip.
 */

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

namespace tracev2 {

constexpr int RAPIDGZIP_PID = 2;
constexpr size_t FLUSH_THRESHOLD = 4096;

inline std::atomic<bool>&
enabled_flag()
{
    static std::atomic<bool> f{false};
    return f;
}

inline std::ofstream&
trace_file()
{
    static std::ofstream f;
    static std::once_flag once;
    std::call_once(once, [] {
        const char* path = std::getenv("GZIPPY_TIMELINE");
        if (path == nullptr) {
            return;
        }
        f.open(path, std::ios::out | std::ios::trunc);
        if (f.is_open()) {
            f << "[\n";
            enabled_flag().store(true, std::memory_order_relaxed);
        }
    });
    return f;
}

inline std::mutex&
file_mutex()
{
    static std::mutex m;
    return m;
}

inline std::chrono::steady_clock::time_point
anchor()
{
    static auto a = std::chrono::steady_clock::now();
    return a;
}

inline bool
is_enabled()
{
    if (enabled_flag().load(std::memory_order_relaxed)) {
        return true;
    }
    trace_file();
    (void)anchor();
    return enabled_flag().load(std::memory_order_relaxed);
}

inline std::atomic<uint32_t>&
next_tid()
{
    static std::atomic<uint32_t> n{1};
    return n;
}

inline uint32_t
current_tid()
{
    static thread_local uint32_t tid = next_tid().fetch_add(1, std::memory_order_relaxed);
    return tid;
}

inline uint64_t
now_us()
{
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now - anchor()).count();
}

/**
 * Per-thread buffer wrapped in a class whose dtor flushes to the shared
 * file under the global mutex. This is what guarantees that worker
 * threads' events make it to disk — std::string alone has no flush hook.
 */
class ThreadBuffer
{
public:
    ThreadBuffer() = default;

    ~ThreadBuffer()
    {
        flush();
    }

    void
    append(const char* line, size_t len)
    {
        m_buf.append(line, len);
        if (m_buf.size() >= FLUSH_THRESHOLD) {
            flush();
        }
    }

    void
    flush()
    {
        if (m_buf.empty()) {
            return;
        }
        std::lock_guard<std::mutex> g(file_mutex());
        auto& f = trace_file();
        if (f.is_open()) {
            f.write(m_buf.data(), m_buf.size());
        }
        m_buf.clear();
    }

private:
    std::string m_buf;
};

inline ThreadBuffer&
per_thread_buffer()
{
    static thread_local ThreadBuffer buf;
    return buf;
}

inline void
flush_all()
{
    per_thread_buffer().flush();
}

inline void
emit_begin(const char* name, const char* args_body)
{
    if (!is_enabled()) {
        return;
    }
    const auto ts = now_us();
    const auto tid = current_tid();
    char line[512];
    int n;
    if (args_body == nullptr || args_body[0] == '\0') {
        n = std::snprintf(line, sizeof(line),
                          "{\"name\":\"%s\",\"ph\":\"B\",\"ts\":%llu,\"pid\":%d,\"tid\":%u},\n",
                          name, static_cast<unsigned long long>(ts), RAPIDGZIP_PID, tid);
    } else {
        n = std::snprintf(line, sizeof(line),
                          "{\"name\":\"%s\",\"ph\":\"B\",\"ts\":%llu,\"pid\":%d,\"tid\":%u,\"args\":{%s}},\n",
                          name, static_cast<unsigned long long>(ts), RAPIDGZIP_PID, tid, args_body);
    }
    if (n > 0) {
        per_thread_buffer().append(line, static_cast<size_t>(n));
    }
}

inline void
emit_end(const char* name)
{
    if (!is_enabled()) {
        return;
    }
    const auto ts = now_us();
    const auto tid = current_tid();
    char line[256];
    int n = std::snprintf(line, sizeof(line),
                          "{\"name\":\"%s\",\"ph\":\"E\",\"ts\":%llu,\"pid\":%d,\"tid\":%u},\n",
                          name, static_cast<unsigned long long>(ts), RAPIDGZIP_PID, tid);
    if (n > 0) {
        per_thread_buffer().append(line, static_cast<size_t>(n));
    }
}

inline void
emit_instant(const char* name, const char* args_body, char scope)
{
    if (!is_enabled()) {
        return;
    }
    const auto ts = now_us();
    const auto tid = current_tid();
    char line[512];
    int n;
    if (args_body == nullptr || args_body[0] == '\0') {
        n = std::snprintf(line, sizeof(line),
                          "{\"name\":\"%s\",\"ph\":\"i\",\"ts\":%llu,\"pid\":%d,\"tid\":%u,\"s\":\"%c\"},\n",
                          name, static_cast<unsigned long long>(ts), RAPIDGZIP_PID, tid, scope);
    } else {
        n = std::snprintf(line, sizeof(line),
                          "{\"name\":\"%s\",\"ph\":\"i\",\"ts\":%llu,\"pid\":%d,\"tid\":%u,\"s\":\"%c\",\"args\":{%s}},\n",
                          name, static_cast<unsigned long long>(ts), RAPIDGZIP_PID, tid, scope, args_body);
    }
    if (n > 0) {
        per_thread_buffer().append(line, static_cast<size_t>(n));
    }
}

class ScopedSpan
{
public:
    explicit ScopedSpan(const char* name) : m_name(name) { emit_begin(name, nullptr); }
    ScopedSpan(const char* name, const char* args_body) : m_name(name) { emit_begin(name, args_body); }
    ~ScopedSpan() { emit_end(m_name); }
    ScopedSpan(const ScopedSpan&) = delete;
    ScopedSpan& operator=(const ScopedSpan&) = delete;

private:
    const char* const m_name;
};

}  // namespace tracev2
