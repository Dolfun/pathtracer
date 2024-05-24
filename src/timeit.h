#pragma once
#include <chrono>
#include <fmt/chrono.h>

inline void timeit(const std::string& info, auto&& f) {
	auto t1 = std::chrono::steady_clock::now();
	std::forward<decltype(f)>(f)();
	auto t2 = std::chrono::steady_clock::now();
	fmt::println("{}: {}", info, std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1));
}