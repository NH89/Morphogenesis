#include "thread_manager.h"

// disk write thread function
void ThreadManager::diskWriteFunc() {
    std::cout << "Disk write thread started" << std::endl;

    while (running){
        // 1. wait for message that GPU is done processing
        // 2. convert data to VTP format
        // 3. write to disk, release mutex/semaphore

        // wait for data to become ready
        std::unique_lock<std::mutex> lock(dataReadyMutex);
        dataReady.wait(lock);
        std::cout << "data may be ready in diskWriteFunc" << std::endl;
        // dataReady is also notified if ThreadManager::teardown() is called, so we need to check if we're no longer
        // running (since the data won't actually be ready in that case)
        if (!running) break;

        // if we get here, we can process the data successfully
        std::cout << "data is confirmed OK, string is: " << globalTestData.justMakingSure << ", counter is: "
            << globalTestData.counter << std::endl;
    }
}

// kernel manager function
void ThreadManager::kernelManagerFunc() {
    std::cout << "Kernel manager thread started" << std::endl;

    while (running){
        // pretend to compute for a while
        std::this_thread::sleep_for(std::chrono::seconds(1));
        globalTestData.counter++;
        globalTestData.justMakingSure = "It worked!";

        // notify disk write thread that data is ready
        dataReady.notify_all();
    }
}

void ThreadManager::create() {
    std::cout << "Creating threads" << std::endl;
    running = true;
    diskWriteThread = std::thread(&ThreadManager::diskWriteFunc, this);
    kernelManagerThread = std::thread(&ThreadManager::kernelManagerFunc, this);
}

void ThreadManager::await(){
    diskWriteThread.join();
    kernelManagerThread.join();
}

void ThreadManager::teardown() {
    std::cout << "Destroying threads and waiting for completion" << std::endl;
    running = false;
    // this is needed to stop the data thread from sitting forever
    dataReady.notify_all();

    // wait for threads to finish to prevent errors
    diskWriteThread.join();
    kernelManagerThread.join();
}