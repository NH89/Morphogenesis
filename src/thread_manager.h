#pragma once
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "fluid_system.h"

/** test data struct for demo */
struct TestData {
    int counter = 0;
    std::string justMakingSure = "Need more time";
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
    /*** Asks threads to shut down and waits for them to finalise. */
    void teardown();
    /** Mainly for testing, blocks until both threads quit. **/
    void await();

private:
    std::thread diskWriteThread;
    std::thread kernelManagerThread;
    std::atomic<bool> running {}; // apparently this counts as an initialiser?
    std::condition_variable dataReady; // true if data is ready for the disk write thread
    std::mutex dataReadyMutex;

    // pretend global data like FBuf, just for test
    TestData globalTestData;

    void diskWriteFunc();
    void kernelManagerFunc();
};