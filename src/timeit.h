#pragma once
#include <chrono>
#include <fmt/chrono.h>

inline void timeit(const std::string& info, auto&& f) {
	using namespace std::chrono;

	auto t1 = steady_clock::now();
	std::forward<decltype(f)>(f)();
	auto t2 = steady_clock::now();

	fmt::print("{:<16} -> ", info);
	
	auto duration = duration_cast<milliseconds>(t2 - t1);
	if (duration.count() < 100) {
		fmt::println("{}", duration);

	} else if (duration.count() < 10000) {
		fmt::println("{:.2}s", duration.count() / 1000.0f);

	} else {
		duration = duration_cast<seconds>(duration);
		fmt::println("{}", duration);
	}
}