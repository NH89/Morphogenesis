#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include <chrono>
#include <sys/types.h>
#include <sys/stat.h>
#include "fluid_system.h"
typedef	unsigned int    uint;

int mk_subdir(char* path){
    std::cout<<"\nmk_subdir: chk 0 path="<<path<<"\n"<<std::flush;
    struct stat st;
    if((stat(path,&st) == 0) && ((st.st_mode & S_IFDIR) != 0)  ){ 
        std::cout<<"\tmk_subdir: chk 1  directory \""<<path<<"\" exists.\n"<<std::flush;
    }else if((stat(path,&st) == 0) && ((st.st_mode & S_IFDIR) == 0) ){
        std::cout<<"\tmk_subdir: chk 1  cannot create directory \""<<path<<"\", file exists.\n"<<std::flush;
        return 1;
    }else{
        std::cout<<"\tmk_subdir: chk 1  creating directory \""<<path<<"\".\n"<<std::flush;
        int check = mkdir(path,0777);
        if (check ){
            printf("\nUnable to create sub-directory: %s\n", path); 
            return 1;
        }
    }
    std::cout<<"\tmk_subdir: chk 2 success\n"<<std::flush;  
    return 0;
}

int main ( int argc, const char** argv ) 
{
    char folder_path[256];
    char spec_file[256];
    if ( argc != 2 ) {
        printf ( "usage: SpecfileBatchGenerator <path_to_folder_containing specification_file.txt> \n" );
        return 0;
    } else {
        sprintf ( spec_file, "%s", argv[1] );
        printf ( "specification_file = %s\n", spec_file );
    }
    
    // Initialize
    cuInit ( 0 );
    int deviceCount = 0;
    cuDeviceGetCount ( &deviceCount );
    if ( deviceCount == 0 ) {
        printf ( "There is no device supporting CUDA.\n" );
        exit ( 0 );
    }
    CUdevice cuDevice;
    cuDeviceGet ( &cuDevice, 0 );
    CUcontext cuContext;
    cuCtxCreate ( &cuContext, 0, cuDevice );
    
    FluidSystem fluid;
    fluid.InitializeCuda ();
    
    std::cout<<"\nSpecfileBatchGenerator chk0, spec_file = "<< spec_file << std::flush;
    fluid.ReadSpecificationFile ( spec_file );
    fluid.SetDebug(2);// NB sets m_FParams.debug for this code, separate from fluid.launchParams.debug for specfile.txt .
    std::string sourceGenomePath = fluid.launchParams.genomePath;
    
    std::cout<<"\nSpecfileBatchGenerator chk1, fluid.launchParams.debug="<<fluid.launchParams.debug<<", fluid.launchParams.paramsPath=" <<fluid.launchParams.paramsPath <<std::flush; //## corect the  .paramsPath !!!
    
    std::string spec_file_str = spec_file;
    spec_file_str.erase(remove(spec_file_str.begin(), spec_file_str.end(), '.'), spec_file_str.end());
    spec_file_str.erase(remove(spec_file_str.begin(), spec_file_str.end(), '/'), spec_file_str.end());
    std::cout<<"\nSpecfileBatchGenerator chk2, spec_file_str = "<< spec_file_str/*spec_file*/ << std::flush;
    
    // Params to loop around:
    sprintf(folder_path, "%s_batch", spec_file_str.c_str() );                             // make sub dir
    if(mk_subdir(folder_path))fluid.Exit();                                               // failed to make directory
    
    char Specification_file_path[256];
    std::cout<<"\n m_FParams.debug = "<< fluid.GetDebug() << std::flush;
    //initstiff
    float default_initstiff = fluid.launchParams.intstiff;
    for (int i=1; i<=32; i*=2){
        float intstiff = default_initstiff * 0.125 * (float)i;                                     // set parameter
        fluid.launchParams.intstiff=i;
        sprintf ( Specification_file_path, "%s/%s_intstiff%f", folder_path, spec_file_str.c_str(), intstiff);
        mk_subdir(Specification_file_path);
        fluid.WriteSpecificationFile_fromLaunchParams(Specification_file_path);                             // write spec file
    }
    //visc
    float default_visc = fluid.launchParams.visc;
    fluid.ReadSpecificationFile ( spec_file );
    fluid.SetDebug(2);// NB sets m_FParams.debug for this code, separate from fluid.launchParams.debug for specfile.txt .
    for (int i=1; i<=32; i*=2){
        float visc = default_visc * 0.125 * (float)i;                                                               // set parameter
        std::string txt_flt=std::to_string(visc);
        fluid.launchParams.visc=visc;
        sprintf ( Specification_file_path, "%s/%s_visc%s", folder_path, spec_file_str.c_str(), txt_flt.c_str()/*visc*/);
        mk_subdir(Specification_file_path);
        fluid.WriteSpecificationFile_fromLaunchParams(Specification_file_path);                             // write spec file
    }
    //surface_tension
    float default_s_t = fluid.launchParams.surface_tension;
    fluid.ReadSpecificationFile ( spec_file );
    fluid.SetDebug(2);// NB sets m_FParams.debug for this code, separate from fluid.launchParams.debug for specfile.txt .
    for (int i=1; i<=32; i*=2){
        float surface_tension = default_s_t * 0.125 * (float)i;                                    // set parameter
        fluid.launchParams.surface_tension=surface_tension;
        sprintf ( Specification_file_path, "%s/%s_surface_tension%f", folder_path, spec_file_str.c_str(), surface_tension);
        mk_subdir(Specification_file_path);
        fluid.WriteSpecificationFile_fromLaunchParams(Specification_file_path);                           // write spec file
    }
    //mass
    float default_mass = fluid.launchParams.mass;
    fluid.ReadSpecificationFile ( spec_file );
    fluid.SetDebug(2);// NB sets m_FParams.debug for this code, separate from fluid.launchParams.debug for specfile.txt .
    for (int i=1; i<=32; i*=2){
        float mass = default_mass * 0.125 * (float)i;                                              // set parameter
        fluid.launchParams.mass=mass;
        sprintf ( Specification_file_path, "%s/%s_mass%f", folder_path, spec_file_str.c_str(), mass);
        mk_subdir(Specification_file_path);
        fluid.WriteSpecificationFile_fromLaunchParams(Specification_file_path);                             // write spec file
    }
    //////Genome parameters: set in genome, per tissue.
    
    //default_modulus
    {
    fluid.ReadGenome(sourceGenomePath.c_str());          //spec_file                          // NB must reset genome to template
    FGenome tempGenome = fluid.GetGenome();
    int d_mod = tempGenome.default_modulus;
    int elastin = tempGenome.elastin;
    float default_modulus = tempGenome.param[elastin][d_mod];
    fluid.ReadSpecificationFile ( spec_file );
    fluid.SetDebug(2);// NB sets m_FParams.debug for this code, separate from fluid.launchParams.debug for specfile.txt .
    for (int i=-4; i<=4; i++){
        float modulus = default_modulus * pow(10,i);                    // set parameter
        tempGenome.param[elastin][d_mod]=modulus;
        
        sprintf ( Specification_file_path, "%s/%s_elastin_modulus%f", folder_path, spec_file_str.c_str(), modulus);
        mk_subdir(Specification_file_path);
        fluid.launchParams.read_genome='y';
        
        fluid.SetGenome(tempGenome);
        std::cout<<"\n m_FParams.debug = "<< fluid.GetDebug() << std::flush;
        fluid.SetDebug(2);
        fluid.WriteGenome(Specification_file_path);
        sprintf ( fluid.launchParams.genomePath, "%s/genome.csv", Specification_file_path );
        fluid.WriteSpecificationFile_fromLaunchParams(Specification_file_path);                             // write spec file
    }
    }
    
    //default_damping
    {
    fluid.ReadGenome(sourceGenomePath.c_str());          //spec_file                          // NB must reset genome to template
    FGenome tempGenome = fluid.GetGenome();
    int d_mod = tempGenome.default_damping;
    int elastin = tempGenome.elastin;
    float default_damping = tempGenome.param[elastin][d_mod] ;
    fluid.ReadSpecificationFile ( spec_file );
    fluid.SetDebug(2);// NB sets m_FParams.debug for this code, separate from fluid.launchParams.debug for specfile.txt .
    for (int i=-4; i<=4; i++){
        float damping = default_damping * pow(10,i);                      // set parameter
        tempGenome.param[elastin][d_mod]=damping;
        
        sprintf ( Specification_file_path, "%s/%s_elastin_damping%f", folder_path, spec_file_str.c_str(), damping);
        mk_subdir(Specification_file_path);
        fluid.launchParams.read_genome='y';
        
        fluid.SetGenome(tempGenome);
        fluid.WriteGenome(Specification_file_path);
        sprintf ( fluid.launchParams.genomePath, "%s/genome.csv", Specification_file_path );
        fluid.WriteSpecificationFile_fromLaunchParams(Specification_file_path);                             // write spec file
    }
    }
    
    //elastLim
    {
    fluid.ReadGenome(sourceGenomePath.c_str());          //spec_file                          // NB must reset genome to template
    FGenome tempGenome = fluid.GetGenome();
    int d_mod = tempGenome.elastLim;
    int elastin = tempGenome.elastin;
    float default_elastLim = tempGenome.param[elastin][d_mod];
    fluid.ReadSpecificationFile ( spec_file );
    fluid.SetDebug(2);// NB sets m_FParams.debug for this code, separate from fluid.launchParams.debug for specfile.txt .
    for (int i=1; i<=32; i*=2){
        float elastLim = default_elastLim * 0.125 * (float)i;                            // set parameter
        tempGenome.param[elastin][d_mod]=elastLim;
        
        sprintf ( Specification_file_path, "%s/%s_elastin_elastLim%f", folder_path, spec_file_str.c_str(), elastLim);
        mk_subdir(Specification_file_path);
        fluid.launchParams.read_genome='y';
        
        fluid.SetGenome(tempGenome);
        fluid.WriteGenome(Specification_file_path);
        sprintf ( fluid.launchParams.genomePath, "%s/genome.csv", Specification_file_path );
        fluid.WriteSpecificationFile_fromLaunchParams(Specification_file_path);                             // write spec file
    }
    }
    
    //default_rest_length
    {
    fluid.ReadGenome(sourceGenomePath.c_str());          //spec_file                          // NB must reset genome to template
    FGenome tempGenome = fluid.GetGenome();
    int d_mod = tempGenome.default_rest_length;
    int elastin = tempGenome.elastin;
    float default_rest_length = tempGenome.param[elastin][d_mod];
    fluid.ReadSpecificationFile ( spec_file );
    fluid.SetDebug(2);// NB sets m_FParams.debug for this code, separate from fluid.launchParams.debug for specfile.txt .
    for (int i=1; i<=32; i*=2){
        float rest_length = default_rest_length * 0.125 * (float)i;                  // set parameter
        tempGenome.param[elastin][d_mod]=rest_length;
        
        sprintf ( Specification_file_path, "%s/%s_elastin_rest_length%f", folder_path, spec_file_str.c_str(), rest_length);
        mk_subdir(Specification_file_path);
        fluid.launchParams.read_genome='y';
        
        fluid.SetGenome(tempGenome);
        fluid.WriteGenome(Specification_file_path);
        sprintf ( fluid.launchParams.genomePath, "%s/genome.csv", Specification_file_path );
        fluid.WriteSpecificationFile_fromLaunchParams(Specification_file_path);                             // write spec file
    }
    }
    
    
    /* From genome.csv
    Remodelling parameters, rows : elastin,collagen,apatite
collumns : /_*triggering bond parameter changes*_/, /_*triggering particle changes*_/, /_*initial values for new bonds*_/
elongation_threshold,	elongation_factor,	strength_threshold,	strengthening_factor,		    max_rest_length,	min_rest_length,	max_modulus,	min_modulus,			    elastLim,	default_rest_length,	default_modulus,	default_damping

    struct FGenome{   // ## currently using fixed size genome for efficiency. NB Particle data size depends on genome size.
        uint mutability[NUM_GENES];
        uint delay[NUM_GENES];
        uint sensitivity[NUM_GENES][NUM_GENES];     // for each gene, its sensitivity to each TF or morphogen
        uint tf_diffusability[NUM_TF];              // for each transcription_factor, the diffusion and breakdown rates of its TF.
        uint tf_breakdown_rate[NUM_TF];
                                                    // sparse lists final entry = num elem, other entries (elem_num, param)
        int secrete[NUM_GENES][2*NUM_TF+1];         // -ve secretion => active breakdown. Can be useful for pattern formation.
        int activate[NUM_GENES][2*NUM_GENES+1];
        //uint *function[NUM_GENES];                // cancelled// Hard code a case-switch that calls each gene's function iff the gene is active.
        enum {elastin,collagen,apatite};
                                                    //FBondParams fbondparams[3];   // 0=elastin, 1=collagen, 2=apatite
        
        enum params{  /_*triggering bond parameter changes*_/ elongation_threshold,   elongation_factor,      strength_threshold,     strengthening_factor,
                      /_*triggering particle changes*_/       max_rest_length,        min_rest_length,        max_modulus,            min_modulus,
                      /_*initial values for new bonds*_/      elastLim,               default_rest_length,    default_modulus,        default_damping
        };
        float param[3][12];
    };
    */
    /*
    if (launchParams.read_genome !='y'){SetupExampleGenome();}// NB default initialization is launchParams.read_genome='n', => used by make_demo.cpp.
    else {ReadGenome(launchParams.genomePath);}               // make_demo2.cpp reads Specification_File.txt, which may give launchParams.read_genome='y', to use an edited genome.
    */
    std::cout<<"\n\nSpecfileBatchGenerator chk3 "<<std::flush;
    
    // clean up and exit
    fluid.Exit ();
    
    size_t   free1, free2, total;
    cudaMemGetInfo(&free1, &total);
    printf("\n\nCuda Memory, before cuCtxDestroy(cuContext): free=%lu, total=%lu.\t",free1,total);
    
    CUresult cuResult = cuCtxDestroy ( cuContext ) ;
    if ( cuResult!=0 ) {printf ( "error closing, cuResult = %i \n",cuResult );}
    
    cudaMemGetInfo(&free2, &total);
    printf("\nAfter cuCtxDestroy(cuContext): free=%lu, total=%lu, released=%lu.\n",free2,total,(free2-free1) );
    
    printf ( "\nClosed SpecfileBatchGenerator.\n" );
    return 0;
}
