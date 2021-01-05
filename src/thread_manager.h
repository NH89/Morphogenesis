#pragma once
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
//#include "fluid_system.h"

/** test data struct for demo */
struct TestData {
    int counter = 0;
    std::string justMakingSure = "Not working!";
};

/**
 * The ThreadManager handles latency hiding, by controlling both the kernel management and disk writing threads
 * and the exchange of information between them.
 * The implementation is based on the C++11 threading API (https://en.cppreference.com/w/cpp/thread)
 */
class ThreadManager {
public:
    ThreadManager() = default;
    /** Creates and starts kernel manager and disk write threads. */
    void create();
    /** Gracefully shuts down kernel manager and disk write threads. */
    void teardown();
    /** Mainly for testing, blocks until both threads quit. */
    void await();

private:
    std::thread diskWriteThread;
    std::thread kernelManagerThread;

    std::atomic<bool> running { false }; // true if teardown() has not been called

    std::mutex condMutex; // mutex for all condition variables
    std::condition_variable dataReady; // notifies the disk write thread that data is ready for writing
    std::condition_variable diskReady; // notifies the kernel manager thread that the disk writer thread is finished
    std::atomic<bool> isDiskReady { true }; // true if disk has finished being written to by disk write thread

    // pretend global data like FBuf, just for test
    TestData globalTestData;

    void diskWriteFunc();
    void kernelManagerFunc();
};
