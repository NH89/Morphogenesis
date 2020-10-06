#include "thread_manager.h"

// disk write thread function
void ThreadManager::diskWriteFunc() {
    std::cout << "Disk write thread started" << std::endl;

    while (running){
        // wait for data to become ready
        std::unique_lock<std::mutex> lock(condMutex);
        dataReady.wait(lock);
        // dataReady is also notified if ThreadManager::teardown() is called, so we need to check if we're no longer
        // running (since the data won't actually be ready in that case)
        if (!running) break;

        // if we get here, we can process the data successfully
        std::cout << "Data received in disk write thread, string is: " << globalTestData.justMakingSure << ", counter is: "
            << globalTestData.counter << std::endl;

        // pretend to write to disk, and inform the kernel manager we have finished
        isDiskReady = false;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        isDiskReady = true;
        diskReady.notify_all();
        std::cout << "Disk thread finished writing" << std::endl;
    }
}

// kernel manager function
void ThreadManager::kernelManagerFunc() {
    std::cout << "Kernel manager thread started" << std::endl;

    while (running){
        auto begin = std::chrono::steady_clock::now();

        // pretend to compute for a while
        std::this_thread::sleep_for(std::chrono::seconds(1));
        globalTestData.counter++;
        globalTestData.justMakingSure = "It worked!";

        // before notifying the disk write thread that we're ready, ensure the disk write thread has finished
        std::unique_lock<std::mutex> lock(condMutex);
        if (!isDiskReady){
            std::cout << "Disk thread not yet ready, waiting for it to finish" << std::endl;
            diskReady.wait(lock);
            if (!running) break;
        }

        // now that we've confirmed disk thread is ready to receive data, send the data its way
        std::cout << "Finished waiting for disk thread, sending it new data" << std::endl;
        dataReady.notify_all();

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time = end - begin;
        std::cout << "Time taken: " << time.count() << " seconds" << std::endl;
    }
}

void ThreadManager::create() {
    std::cout << "Creating threads" << std::endl;
    running = true;
    isDiskReady = true; // disk thread starts ready to receive data
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
    // notify both threads if they're blocked waiting on a condition variable
    dataReady.notify_all();
    diskReady.notify_all();

    // wait for threads to finish to prevent errors (graceful shutdown)
    diskWriteThread.join();
    kernelManagerThread.join();
}
