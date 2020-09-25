#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include "fluid_system.h"
#include "thread_manager.h"

int main(){
    std::cout << "Hello from thread demo" << std::endl;
    ThreadManager manager;
    manager.create();

    // the purpose of this loop is to prove that the threads are working and also to stop the app from exiting straight away
    for (int i = 0; i < 5; i++){
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << "main loop running" << std::endl;
    }

    manager.teardown();
    return EXIT_SUCCESS;
}