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
    printf("\n\n### Run2Simulation(), m_FParams.debug=%i.   launchParams.save_csv==%c,   launchParams.save_vtp==%c ##############################################\n", m_FParams.debug,   launchParams.save_csv,  launchParams.save_vtp );
    																								time_point_Run3_[0]	= std::chrono::steady_clock::now();
    Init_FCURAND_STATE_CUDA ();
    																								time_point_Run3_[1]	= std::chrono::steady_clock::now();
    auto old_begin = std::chrono::steady_clock::now();
    TransferPosVelVeval ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After TransferPosVelVeval, before 1st timestep", 1/*mbDebug*/);
    																								time_point_Run3_[2]	= std::chrono::steady_clock::now();
    setFreeze(true);
    																								time_point_Run3_[3]	= std::chrono::steady_clock::now();
    m_Debug_file=0;
    																									if (m_FParams.debug>0)std::cout<<"\n\nFreeze()"<<-1<<"\n"<<std::flush;
    Run2PhysicalSort();
    																								time_point_Run3_[4]	= std::chrono::steady_clock::now();
    InitializeBondsCUDA();		// ### prevent bond formation
    																								time_point_Run3_[5]	= std::chrono::steady_clock::now();

    																									if(launchParams.save_csv=='y'||launchParams.save_vtp=='y') TransferFromCUDA ();
    																									cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "Run2Simulation After TransferFromCUDA", mbDebug);
    																									if(launchParams.save_csv=='y') SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+90);
    																									if(launchParams.save_vtp=='y') SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+90);
    																									if (m_FParams.debug>0)cout << "\n File# " << launchParams.file_num << ". " << std::flush;
    launchParams.file_num+=100;

    /////////
    																								time_point_Run3_[6]	= std::chrono::steady_clock::now();


																											std::cerr<<"\n\nFreeze steps:  freeze file_num="<<launchParams.file_num<<", of "<<launchParams.freeze_steps<<"\n"<<std::flush;
        																									std::cout<<"\n\nFreeze steps:  freeze file_num="<<launchParams.file_num<<", of "<<launchParams.freeze_steps<<"\n"<<std::flush;
    for ( ; launchParams.file_num<launchParams.freeze_steps; launchParams.file_num+=100 ) {
																											std::cerr<<"\n\nfreeze file_num="<<launchParams.file_num<<", of "<<launchParams.freeze_steps<<"\n"<<std::flush;
        																									std::cout<<"\n\nfreeze file_num="<<launchParams.file_num<<", of "<<launchParams.freeze_steps<<"\n"<<std::flush;
        m_Debug_file=0;
        m_Frame=launchParams.file_num;
        launchParams.file_increment=0;                                                                      // used within Run2InnerPhysicalLoop();
        for ( int j=0; j<launchParams.steps_per_file; j++ ) {
            for (int k=0; k<launchParams.steps_per_InnerPhysicalLoop; k++) {
    																								time_point_Run3_[7]	= std::chrono::steady_clock::now();
                std::cout<<"\tk="<<k;
    																								time_point_Run3_[8]	= std::chrono::steady_clock::now();
                Run2InnerPhysicalLoop();                                                                    // Run2InnerPhysicalLoop();
            }
    																								time_point_Run3_[9]	= std::chrono::steady_clock::now();
            if(launchParams.gene_activity=='y') Run2GeneAction();                                           // Run2GeneAction();
    																								time_point_Run3_[10]	= std::chrono::steady_clock::now();
            if(launchParams.remodelling=='y') Run2Remodelling(launchParams.steps_per_InnerPhysicalLoop);                                          // Run2Remodelling();
    																								time_point_Run3_[11]	= std::chrono::steady_clock::now();

    																								time_point_Run3_[12]	= std::chrono::steady_clock::now();
            Run2PhysicalSort();                                                                             // Run2PhysicalSort();                // sort required for SavePointsVTP2
    																								time_point_Run3_[13]	= std::chrono::steady_clock::now();
            ZeroVelCUDA ();                                                                                 // remove velocity, kinetic energy and momentum
    																								time_point_Run3_[14]	= std::chrono::steady_clock::now();
        }
        																									if(launchParams.save_csv=='y'||launchParams.save_vtp=='y') TransferFromCUDA ();
        cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "Run2Simulation After TransferFromCUDA", mbDebug);
        																									if(launchParams.save_csv=='y') SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+90);
        																									if(launchParams.save_vtp=='y') SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+90);
        																									if (m_FParams.debug>0)cout << "\n File# " << launchParams.file_num << ". " << std::flush;
    }
    																								time_point_Run3_[15]	= std::chrono::steady_clock::now();
    setFreeze(false);                                                                                       // freeze=false => bonds can be broken now.
    																								time_point_Run3_[16]	= std::chrono::steady_clock::now();
    																										printf("\n\n### Freeze finished, starting normal Run ##############################################\n\n");
    //Run2PhysicalSort();

    																										cout<<"\n launchParams.file_num = "<< launchParams.file_num <<"   <launchParams.num_files = "<< launchParams.num_files  << std::flush;
    																								time_point_Run3_[17]	= std::chrono::steady_clock::now();

    for ( ; launchParams.file_num<launchParams.num_files; launchParams.file_num+=100 ) {
																											std::cerr<<"\n\nfile_num="<<launchParams.file_num<<", of "<<launchParams.num_files<<"\n"<<std::flush;
        																									std::cout<<"\n\nfile_num="<<launchParams.file_num<<", of "<<launchParams.num_files<<"\n"<<std::flush;
        m_Debug_file=0;
        m_Frame=launchParams.file_num;
        launchParams.file_increment=0;                                                                      // used within Run2InnerPhysicalLoop();
    																								time_point_Run3_[18]	= std::chrono::steady_clock::now();
        for ( int j=0; j<launchParams.steps_per_file; j++ ) {
            for (int k=0; k<launchParams.steps_per_InnerPhysicalLoop; k++) {
                std::cout<<"\tk="<<k;
    																								time_point_Run3_[19]	= std::chrono::steady_clock::now();
                Run2InnerPhysicalLoop();                                                                    // Run2InnerPhysicalLoop();
            }
    																								time_point_Run3_[20]	= std::chrono::steady_clock::now();
            if(launchParams.gene_activity=='y') Run2GeneAction();                                           // Run2GeneAction();
    																								time_point_Run3_[20]	= std::chrono::steady_clock::now();
            if(launchParams.remodelling=='y') Run2Remodelling(launchParams.steps_per_InnerPhysicalLoop);    // Run2Remodelling();
    																								time_point_Run3_[21]	= std::chrono::steady_clock::now();

            Run2PhysicalSort();                                                                             // Run2PhysicalSort();                // sort required for SavePointsVTP2
        }
    																								time_point_Run3_[22]	= std::chrono::steady_clock::now();
        																									auto begin = std::chrono::steady_clock::now();
																											if(launchParams.save_csv=='y'||launchParams.save_vtp=='y') TransferFromCUDA ();
        																									cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "Run2Simulation After TransferFromCUDA", mbDebug);
        																									if(launchParams.save_csv=='y') SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+90);
        																									if(launchParams.save_vtp=='y') SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+90);
        																									if (m_FParams.debug>0)cout << "\n File# " << launchParams.file_num << ". " << std::flush;
/*
        																									// auto end = std::chrono::steady_clock::now();
        																									// std::chrono::duration<double> time = end - begin;
        																									// std::chrono::duration<double> begin_dbl = begin - old_begin;
        																									// /_*if(launchParams.debug>0)*_/ std::cout	<<"\nOuter loop duration : " 			<< begin_dbl.count() 		<<" seconds. "
        																									//                                         <<"\nTime taken to write files for "	<< NumPoints() 				<<" particles : " 	<< time.count() << " seconds. "
        																									//                                         <<"\nlaunchParams.num_files="			<< launchParams.num_files 	<< "\n" 			<< std::endl;
        																									// old_begin = begin;
                                 //
                                 //                                                                    // timing Run3Simulation
                                 //                                                                            std::cout	<<"\n\ntiming Run3Simulation, nanoseconds"
                                 //                                                                            			<<"\n = "		<<	( time_point_Run3_[0]	- time_point_Run3_[1]	).count()
                                 //                                                                            			<<"\n setFreeze(true)			= "		<<	( time_point_Run3_[1]	- time_point_Run3_[2]	).count()
                                 //                                                                            			<<"\n = "		<<	( time_point_Run3_[2]	- time_point_Run3_[3]	).count()
                                 //                                                                            			<<"\n Run2PhysicalSort()		= "		<<	( time_point_Run3_[3]	- time_point_Run3_[4]	).count()
                                 //                                                                            			<<"\n InitializeBondsCUDA()		= "		<<	( time_point_Run3_[4]	- time_point_Run3_[5]	).count()
                                 //                                                                            			<<"\n = "		<<	( time_point_Run3_[5]	- time_point_Run3_[6]	).count()
                                 //                                                                            			<<"\n = "		<<	( time_point_Run3_[6]	- time_point_Run3_[7]	).count()
                                 //                                                                            			<<"\n = "		<<	( time_point_Run3_[7]	- time_point_Run3_[8]	).count()
                                 //                                                                            			<<"\n Run2InnerPhysicalLoop()= "		<<	( time_point_Run3_[8]	- time_point_Run3_[9]	).count()
                                 //                                                                            			<<"\n Run2GeneAction()			= "		<<	( time_point_Run3_[9]	- time_point_Run3_[10]	).count()
                                 //                                                                            			<<"\n Run2Remodelling			= "		<<	( time_point_Run3_[10]	- time_point_Run3_[11]	).count()
                                 //                                                                            			<<"\n = "		<<	( time_point_Run3_[11]	- time_point_Run3_[12]	).count()
                                 //                                                                            			<<"\n Run2PhysicalSort()		= "		<<	( time_point_Run3_[12]	- time_point_Run3_[13]	).count()
                                 //                                                                            			<<"\n ZeroVelCUDA ()			= "		<<	( time_point_Run3_[13]	- time_point_Run3_[14]	).count()
                                 //                                                                            			<<"\n = "		<<	( time_point_Run3_[14]	- time_point_Run3_[15]	).count()
                                 //                                                                            			<<"\n setFreeze(false)			= "		<<	( time_point_Run3_[15]	- time_point_Run3_[16]	).count()
                                 //                                                                            			<<"\n = "		<<	( time_point_Run3_[16]	- time_point_Run3_[17]	).count()
                                 //                                                                            			<<"\n = "		<<	( time_point_Run3_[17]	- time_point_Run3_[18]	).count()
                                 //                                                                            			<<"\n Run2InnerPhysicalLoop()	= "		<<	( time_point_Run3_[18]	- time_point_Run3_[19]	).count()
                                 //                                                                            			<<"\n Run2GeneAction()			= "		<<	( time_point_Run3_[19]	- time_point_Run3_[20]	).count()
                                 //                                                                            			<<"\n Run2Remodelling 			= "		<<	( time_point_Run3_[20]	- time_point_Run3_[21]	).count()
                                 //                                                                            			<<"\n Run2PhysicalSort()		= "		<<	( time_point_Run3_[21]	- time_point_Run3_[22]	).count()
                                 //                                                                            			<<"\n = "		<<	( time_point_Run3_[22]	- time_point_Run3_[23]	).count()
                                 //                                                                                        <<std::flush;
                                 //
                                 //                                                                    // timing Run2PhysicalSort
                                 //                                                                            std::cout	<<"\n\ntiming Run2PhysicalSort, nanoseconds"
                                 //                                                                            			<<"\n InsertParticlesCUDA 		= "		<<	( time_point_Run2PhysicalSort[0]	- time_point_Run2PhysicalSort[1]	).count()
                                 //                                                                            			<<"\n cuCtxSynchronize()		= "		<<	( time_point_Run2PhysicalSort[1]	- time_point_Run2PhysicalSort[2]	).count()
                                 //                                                                            			<<"\n PrefixSumCellsCUDA ( 1 )	= "		<<	( time_point_Run2PhysicalSort[2]	- time_point_Run2PhysicalSort[3]	).count()
                                 //                                                                            			<<"\n cuCtxSynchronize()		= "		<<	( time_point_Run2PhysicalSort[3]	- time_point_Run2PhysicalSort[4]	).count()
                                 //                                                                            			<<"\n CountingSortFullCUDA 		= "		<<	( time_point_Run2PhysicalSort[4]	- time_point_Run2PhysicalSort[5]	).count()
                                 //                                                                            			<<"\n cuCtxSynchronize()		= "		<<	( time_point_Run2PhysicalSort[6]	- time_point_Run2PhysicalSort[6]	).count()
                                 //                                                                                        <<std::flush;
                                 //
                                 //                                                                    // timing Run2InnerPhysicalLoop
                                 //                                                                            std::cout	<<"\n\ntiming Run2InnerPhysicalLoop, nanoseconds"
                                 //                                                                            			<<"\n InitializeBondsCUDA ()	= "		<<	( time_point_Run2InnerPhysicalLoop[0]	- time_point_Run2InnerPhysicalLoop[1]	).count()
                                 //                                                                            			<<"\n ComputePressureCUDA()		= "		<<	( time_point_Run2InnerPhysicalLoop[1]	- time_point_Run2InnerPhysicalLoop[2]	).count()
                                 //                                                                            			<<"\n ComputeForceCUDA ()		= "		<<	( time_point_Run2InnerPhysicalLoop[2]	- time_point_Run2InnerPhysicalLoop[3]	).count()
                                 //                                                                            			<<"\n TransferPosVelVeval ()	= "		<<	( time_point_Run2InnerPhysicalLoop[3]	- time_point_Run2InnerPhysicalLoop[4]	).count()
                                 //                                                                            			<<"\n AdvanceCUDA				= "		<<	( time_point_Run2InnerPhysicalLoop[4]	- time_point_Run2InnerPhysicalLoop[5]	).count()
                                 //                                                                            			<<"\n SpecialParticlesCUDA		= "		<<	( time_point_Run2InnerPhysicalLoop[5]	- time_point_Run2InnerPhysicalLoop[6]	).count()
                                 //                                                                            			<<"\n TransferPosVelVevalFromTemp = "		<<	( time_point_Run2InnerPhysicalLoop[6]	- time_point_Run2InnerPhysicalLoop[7]	).count()
                                 //                                                                            			<<"\n = "		<<	( time_point_Run2InnerPhysicalLoop[7]	- time_point_Run2InnerPhysicalLoop[8]	).count()
                                 //                                                                            			<<"\n AdvanceTime ()			= "		<<	( time_point_Run2InnerPhysicalLoop[8]	- time_point_Run2InnerPhysicalLoop[9]	).count()
                                 //                                                                                        <<std::flush;
                                 //
                                 //                                                                    // timing Run2GeneAction
                                 //                                                                            std::cout	<<"\n\ntiming Run2GeneAction, nanoseconds"
                                 //                                                                            			<<"\n ComputeDiffusionCUDA()	= "		<<	( time_point_Run2GeneAction[0]	- time_point_Run2GeneAction[1]	).count()
                                 //                                                                            			<<"\n ComputeGenesCUDA()		= "		<<	( time_point_Run2GeneAction[0]	- time_point_Run2GeneAction[1]	).count()
                                 //                                                                                        <<std::flush;
                                 //
                                 //                                                                    // timing Run2Remodelling
                                 //                                                                            std::cout	<<"\n\ntiming Run2Remodelling, nanoseconds"
                                 //                                                                            			<<"\n AssembleFibresCUDA ()		= "		<<	( time_point_Run2Remodelling[0]	- time_point_Run2Remodelling[1]	).count()
                                 //                                                                            			<<"\n ComputeBondChangesCUDA	= "		<<	( time_point_Run2Remodelling[1]	- time_point_Run2Remodelling[2]	).count()
                                 //                                                                            			<<"\n PrefixSumChangesCUDA ( 1 )= "		<<	( time_point_Run2Remodelling[2]	- time_point_Run2Remodelling[3]	).count()
                                 //                                                                            			<<"\n CountingSortChangesCUDA	= "		<<	( time_point_Run2Remodelling[3]	- time_point_Run2Remodelling[4]	).count()
                                 //                                                                            			<<"\n ComputeParticleChangesCUDA= "		<<	( time_point_Run2Remodelling[4]	- time_point_Run2Remodelling[5]	).count()
                                 //                                                                            			<<"\n = "		<<	( time_point_Run2Remodelling[5]	- time_point_Run2Remodelling[6]	).count()
                                 //                                                                                        <<std::flush;
*/
/*
                                                                                                    // timing InsertParticlesCUDA
                                                                                                            std::cout	<<"\n\n###############################################################"
                                                                                                            			<<"\n\ntiming InsertParticlesCUDA, nanoseconds"
                                                                                                            			<<"\n cuMemsetD8 FGRIDCNT FGRIDOFF					= "		<<	( time_point_InsertParticlesCUDA[0]	- time_point_InsertParticlesCUDA[1]	).count()
                                                                                                            			<<"\n cuMemsetD8 FGRIDCNT FGRIDOFF ACTIVE_GENES		= "		<<	( time_point_InsertParticlesCUDA[1]	- time_point_InsertParticlesCUDA[2]	).count()
                                                                                                            			<<"\n computeNumBlocks								= "		<<	( time_point_InsertParticlesCUDA[2]	- time_point_InsertParticlesCUDA[3]	).count()
                                                                                                            			<<"\n cuLaunchKernel(m_Func[FUNC_INSERT]			= "		<<	( time_point_InsertParticlesCUDA[3]	- time_point_InsertParticlesCUDA[4]	).count()
                                                                                                            			<<"\n = "		<<	( time_point_InsertParticlesCUDA[4]	- time_point_InsertParticlesCUDA[5]	).count()
                                                                                                            			<<"\n cuMemcpyDtoH FGCELL FGNDX FGRIDCNT			= "		<<	( time_point_InsertParticlesCUDA[5]	- time_point_InsertParticlesCUDA[6]	).count()
                                                                                                            			<<"\n = "		<<	( time_point_InsertParticlesCUDA[6]	- time_point_InsertParticlesCUDA[7]	).count()
                                                                                                                        <<std::flush;

                                                                                                    // timing PrefixSumCellsCUDA
                                                                                                            std::cout	<<"\n\ntiming PrefixSumCellsCUDA, nanoseconds"
                                                                                                            			<<"\n = "		<<	( time_point_PrefixSumCellsCUDA[0]	- time_point_PrefixSumCellsCUDA[1]	).count()
                                                                                                            			<<"\n cuLaunchKernel ( 	m_Func[FUNC_FPREFIXSUM] 	numElem2 	= "		<<	( time_point_PrefixSumCellsCUDA[1]	- time_point_PrefixSumCellsCUDA[2]	).count()
                                                                                                            			<<"\n cuLaunchKernel ( 	m_Func[FUNC_FPREFIXSUM] 	numElem3 	= "		<<	( time_point_PrefixSumCellsCUDA[2]	- time_point_PrefixSumCellsCUDA[3]	).count()
                                                                                                            			<<"\n cuLaunchKernel (	m_Func[FUNC_FPREFIXSUM] 	1		 	= "		<<	( time_point_PrefixSumCellsCUDA[3]	- time_point_PrefixSumCellsCUDA[4]	).count()
                                                                                                            			<<"\n cuLaunchKernel (	m_Func[FUNC_FPREFIXFIXUP], 	numElem3	= "		<<	( time_point_PrefixSumCellsCUDA[4]	- time_point_PrefixSumCellsCUDA[5]	).count()
                                                                                                            			<<"\n cuLaunchKernel ( 	m_Func[FUNC_FPREFIXFIXUP], 	numElem2	= "		<<	( time_point_PrefixSumCellsCUDA[5]	- time_point_PrefixSumCellsCUDA[6]	).count()
                                                                                                            			<<"\n = "		<<	( time_point_PrefixSumCellsCUDA[6]	- time_point_PrefixSumCellsCUDA[7]	).count()
                                                                                                            			<<"\n = "		<<	( time_point_PrefixSumCellsCUDA[7]	- time_point_PrefixSumCellsCUDA[8]	).count()
                                                                                                            			<<"\n cuMemsetD8 ( scan1 array2 scan2 array3 scan3				= "		<<	( time_point_PrefixSumCellsCUDA[8]	- time_point_PrefixSumCellsCUDA[9]	).count()
                                                                                                            			<<"\n cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], 	numElem2	= "		<<	( time_point_PrefixSumCellsCUDA[9]	- time_point_PrefixSumCellsCUDA[10]	).count()
                                                                                                                        <<"\n cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], 	numElem3	= "		<<	( time_point_PrefixSumCellsCUDA[10]	- time_point_PrefixSumCellsCUDA[11]	).count()
                                                                                                            			<<"\n cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], 		1		= "		<<	( time_point_PrefixSumCellsCUDA[11]	- time_point_PrefixSumCellsCUDA[12]	).count()
                                                                                                            			<<"\n cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], 	numElem3	= "		<<	( time_point_PrefixSumCellsCUDA[12]	- time_point_PrefixSumCellsCUDA[13]	).count()
                                                                                                            			<<"\n cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], 	numElem2	= "		<<	( time_point_PrefixSumCellsCUDA[13]	- time_point_PrefixSumCellsCUDA[14]	).count()

                                                                                                                        <<"\n cuLaunchKernel ( m_Func[FUNC_TALLYLISTS], 	NUM_GENES	= "		<<	( time_point_PrefixSumCellsCUDA[14]	- time_point_PrefixSumCellsCUDA[15]	).count()

                                                                                                                        <<"\n AllocateBufferDenseLists( gene, 							= "		<<	( time_point_PrefixSumCellsCUDA[15]	- time_point_PrefixSumCellsCUDA[16]	).count()
                                                                                                            			<<"\n cuMemcpyHtoD(m_Fluid.gpu(FDENSE_LISTS) FDENSE_BUF_LENGTHS	= "		<<	( time_point_PrefixSumCellsCUDA[16]	- time_point_PrefixSumCellsCUDA[17]	).count()
                                                                                                            			<<"\n = "		<<	( time_point_PrefixSumCellsCUDA[17]	- time_point_PrefixSumCellsCUDA[18]	).count()
                                                                                                                        <<std::flush;


                                                                                                    // timing CountingSortFullCUDA
                                                                                                            std::cout	<<"\n\ntiming CountingSortFullCUDA, nanoseconds"
                                                                                                            			<<"\n = "		<<	( time_point_CountingSortFullCUDA[0]	- time_point_CountingSortFullCUDA[1]	).count()



                                                                                                                        <<std::flush;


                                                                                                    // timing PrefixSumChangesCUDA
                                                                                                            std::cout	<<"\n\ntiming PrefixSumChangesCUDA, nanoseconds"
                                                                                                            			<<"\n = "		<<	( time_point_PrefixSumChangesCUDA[0]	- time_point_PrefixSumChangesCUDA[1]	).count()



                                                                                                                        <<std::flush;


                                                                                                    // timing ComputeGenesCUDA
                                                                                                            std::cout	<<"\n\ntiming ComputeGenesCUDA, nanoseconds"
                                                                                                            			<<"\n = "		<<	( time_point_ComputeGenesCUDA[0]	- time_point_ComputeGenesCUDA[1]	).count()



                                                                                                                        <<std::flush;


                                                                                                    // timing SpecialParticlesCUDA
                                                                                                            std::cout	<<"\n\ntiming SpecialParticlesCUDA, nanoseconds"
                                                                                                            			<<"\n = "		<<	( time_point_SpecialParticlesCUDA[0]	- time_point_SpecialParticlesCUDA[1]	).count()


                                                                                                                        <<std::flush;
*/

        //if (mActivePoints < 500 ){std::cout<<"\n(mActivePoints < 500) stopping. chk why I am loosing particles?"<<std::flush;  Exit();}        // temp chk for why I am loosing particles.
    }
    //launchParams.file_num++;
     																				/*if(launchParams.debug>0)*/ std::cerr << "\nWriting final files \n" << std::endl;
     																				/*if(launchParams.debug>0)*/ std::cout << "\nWriting final files \n" << std::endl;

    TransferFromCUDA ();
   // m_FParams.debug = 2; //	### temporary
    SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+99);
    SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+99);   // save "end condition", even if not saving the series.


    WriteSimParams ( launchParams.outPath );
    WriteGenome( launchParams.outPath );
    WriteSpecificationFile_fromLaunchParams( launchParams.outPath );
																					/*if(launchParams.debug>0)*/ std::cerr << "\nFluidSystem::run3() finished \n" << std::endl;
																					/*if(launchParams.debug>0)*/ std::cout << "\nFluidSystem::run3() finished \n" << std::endl;
}
