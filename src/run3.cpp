#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <unistd.h>
#include <curand_kernel.h>
#include <chrono>
#include <cstring>
#include "cutil_math.h"
#include "fluid_system.h"


void FluidSystem::Run3Simulation(){
    printf("\n\n Run2Simulation(), m_FParams.debug=%i.   launchParams.save_csv==%c,   launchParams.save_vtp==%c ##############################################\n", m_FParams.debug,   launchParams.save_csv,  launchParams.save_vtp );
    Init_FCURAND_STATE_CUDA ();
    auto old_begin = std::chrono::steady_clock::now();
    TransferPosVelVeval ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After TransferPosVelVeval, before 1st timestep", 1/*mbDebug*/);
    setFreeze(true);
    m_Debug_file=0;
    																									if (m_FParams.debug>0)std::cout<<"\n\nFreeze()"<<-1<<"\n"<<std::flush;
    Run2PhysicalSort();
    InitializeBondsCUDA();		// ### prevent bond formation

    																									if(launchParams.save_csv=='y'||launchParams.save_vtp=='y') TransferFromCUDA ();
    																									cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "Run2Simulation After TransferFromCUDA", mbDebug);
    																									if(launchParams.save_csv=='y') SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+90);
    																									if(launchParams.save_vtp=='y') SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+90);
    																									if (m_FParams.debug>0)cout << "\n File# " << launchParams.file_num << ". " << std::flush;
    launchParams.file_num+=100;

    /////////
    for ( ; launchParams.file_num<launchParams.freeze_steps; launchParams.file_num+=100 ) {
        																									std::cout<<"\n\nfile_num="<<launchParams.file_num<<", of "<<launchParams.num_files<<"\n"<<std::flush;
        m_Debug_file=0;
        m_Frame=launchParams.file_num;
        launchParams.file_increment=0;                                                                      // used within Run2InnerPhysicalLoop();
        for ( int j=0; j<launchParams.steps_per_file; j++ ) {
            for (int k=0; k<launchParams.steps_per_InnerPhysicalLoop; k++) {
                std::cout<<"\tk="<<k;
                Run2InnerPhysicalLoop();                                                                    // Run2InnerPhysicalLoop();
            }
            if(launchParams.gene_activity=='y') Run2GeneAction();                                           // Run2GeneAction();
            if(launchParams.remodelling=='y') Run2Remodelling(launchParams.steps_per_InnerPhysicalLoop);                                          // Run2Remodelling();

            Run2PhysicalSort();                                                                             // Run2PhysicalSort();                // sort required for SavePointsVTP2
            ZeroVelCUDA ();                                                                                 // remove velocity, kinetic energy and momentum
        }
        																									if(launchParams.save_csv=='y'||launchParams.save_vtp=='y') TransferFromCUDA ();
        cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "Run2Simulation After TransferFromCUDA", mbDebug);
        																									if(launchParams.save_csv=='y') SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+90);
        																									if(launchParams.save_vtp=='y') SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+90);
        																									if (m_FParams.debug>0)cout << "\n File# " << launchParams.file_num << ". " << std::flush;
    }
    setFreeze(false);                                                                                       // freeze=false => bonds can be broken now.
    																										printf("\n\nFreeze finished, starting normal Run ##############################################\n\n");
    //Run2PhysicalSort();

    																										cout<<"\n launchParams.file_num = "<< launchParams.file_num <<"   <launchParams.num_files = "<< launchParams.num_files  << std::flush;

    for ( ; launchParams.file_num<launchParams.num_files; launchParams.file_num+=100 ) {
        																									std::cout<<"\n\nfile_num="<<launchParams.file_num<<", of "<<launchParams.num_files<<"\n"<<std::flush;
        m_Debug_file=0;
        m_Frame=launchParams.file_num;
        launchParams.file_increment=0;                                                                      // used within Run2InnerPhysicalLoop();
        for ( int j=0; j<launchParams.steps_per_file; j++ ) {
            for (int k=0; k<launchParams.steps_per_InnerPhysicalLoop; k++) {
                std::cout<<"\tk="<<k;
                Run2InnerPhysicalLoop();                                                                    // Run2InnerPhysicalLoop();
            }
            if(launchParams.gene_activity=='y') Run2GeneAction();                                           // Run2GeneAction();
            if(launchParams.remodelling=='y') Run2Remodelling(launchParams.steps_per_InnerPhysicalLoop);    // Run2Remodelling();

            Run2PhysicalSort();                                                                             // Run2PhysicalSort();                // sort required for SavePointsVTP2
        }
        																									auto begin = std::chrono::steady_clock::now();
																											if(launchParams.save_csv=='y'||launchParams.save_vtp=='y') TransferFromCUDA ();
        																									cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "Run2Simulation After TransferFromCUDA", mbDebug);
        																									if(launchParams.save_csv=='y') SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+90);
        																									if(launchParams.save_vtp=='y') SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+90);
        																									if (m_FParams.debug>0)cout << "\n File# " << launchParams.file_num << ". " << std::flush;

        																									auto end = std::chrono::steady_clock::now();
        																									std::chrono::duration<double> time = end - begin;
        																									std::chrono::duration<double> begin_dbl = begin - old_begin;
        																									/*if(launchParams.debug>0)*/ std::cout	<<"\nOuter loop duration : " 			<< begin_dbl.count() 		<<" seconds. "
        																									                                        <<"\nTime taken to write files for "	<< NumPoints() 				<<" particles : " 	<< time.count() << " seconds. "
        																									                                        <<"\nlaunchParams.num_files="			<< launchParams.num_files 	<< "\n" 			<< std::endl;
        																									old_begin = begin;

        //if (mActivePoints < 500 ){std::cout<<"\n(mActivePoints < 500) stopping. chk why I am loosing particles?"<<std::flush;  Exit();}        // temp chk for why I am loosing particles.
    }
    //launchParams.file_num++;

    TransferFromCUDA ();
   // m_FParams.debug = 2; //	### temporary
     																				/*if(launchParams.debug>0)*/ std::cout << "\nWriting final files \n" << std::endl;
    SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+99);
    SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+99);   // save "end condition", even if not saving the series.


    WriteSimParams ( launchParams.outPath );
    WriteGenome( launchParams.outPath );
    WriteSpecificationFile_fromLaunchParams( launchParams.outPath );
}
