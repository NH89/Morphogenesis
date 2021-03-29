

#include "fluid_system.h"

void FluidSystem::ReadGenome( const char * relativePath){
    // NB currently GPU allocation is by Allocate particles, called by ReadPointsCSV.
    const char * genes_file_path = relativePath;
    if (m_FParams.debug>1)printf("\n## opening file %s \n", genes_file_path);
    FILE * genes_file = fopen(genes_file_path, "rb");
    if (genes_file == NULL) {
        if (m_FParams.debug>1)std::cout << "\nvoid FluidSystem::ReadGenome( const char * relativePath, int gpu_mode, int cpu_mode)  Could not read file "<< genes_file_path <<"\n"<< std::flush;
        assert(0);
    }
    int num_genes1, num_tf, ret=0;
    ret += std::fscanf(genes_file, "num genes = %i,\tnum transcription factors = %i \n", &num_genes1, &num_tf);
    
    if (m_FParams.debug>1 && ((num_genes1 != NUM_GENES) || (num_tf != NUM_TF ))){
        std::cout << "\n! Miss-match of parameters ! ((num_genes1 != num_genes2) || (num_genes1 != NUM_GENES) || (num_tf != NUM_TF ) )\n";
        std::cout << "num_genes = " << num_genes1 <<"\t NUM_GENES = "<<NUM_GENES<<"\tnum_tf = "<<num_tf<<"\n";
    }
    ret += std::fscanf(genes_file, "mutability, delay,\tsensitivity[NUM_GENES]");
    for(int i=0; i<NUM_GENES; i++)ret += std::fscanf(genes_file, ",\t");
    
    ret += std::fscanf(genes_file, "secrete[NUM_TF][2]");
    for(int i=0; i<NUM_TF; i++)ret += std::fscanf(genes_file, ",,\t");
    
    ret += std::fscanf(genes_file, "secrete[2*NUM_TF], ");
    
    ret += std::fscanf(genes_file, "activate[NUM_GENES][2]");
    for(int i=0; i<NUM_GENES; i++)ret += std::fscanf(genes_file, ",,\t");
    
    ret += std::fscanf(genes_file, "activate[2*NUM_GENES]");
    
    ret=0;
    int i, j;
    for (i=0; i<num_genes1; i++ ) {
        ret = std::fscanf(genes_file,"%i,%i,",&m_FGenome.mutability[i],&m_FGenome.delay[i] );
        for(int j=0; j<NUM_GENES; j++)  ret += std::fscanf(genes_file,"%i,", &m_FGenome.sensitivity[i][j] );
        for(int j=0; j<NUM_TF; j++) ret += std::fscanf(genes_file, "%i,%i,\t", &m_FGenome.secrete[i][j*2], &m_FGenome.secrete[i][j*2 + 1] );//(elemID, secretion_rate)
        ret += std::fscanf(genes_file, "%i,\t\t", &m_FGenome.secrete[i][2*NUM_TF] );            //num active elements, NB sparse list, kernel will only read active elems.
        for(int j=0; j<NUM_GENES; j++) ret += std::fscanf(genes_file, "%i,%i,\t", &m_FGenome.activate[i][j*2], &m_FGenome.activate[i][j*2 + 1] );//(elemID, other_geneID)
        ret += std::fscanf(genes_file, "%i,\t\t \n", &m_FGenome.activate[i][2*NUM_GENES] );        //num active elements,
        if (ret != (2 + NUM_GENES + NUM_TF*2 + 1 + NUM_GENES*2 + 1) ) {
            if (m_FParams.debug>1)std::cout << "\nvoid FluidSystem::ReadGenome, read failure !  gene number = " << i << ", ret = "<< ret <<"\n " << std::flush;
            fclose(genes_file);
            return;
        }
        if (m_FParams.debug>1) for(int j=0; j<NUM_GENES; j++)  std::cout << m_FGenome.sensitivity[i][j] <<",";
        if (m_FParams.debug>1)std::cout <<"\n";
    }
    if (m_FParams.debug>1)std::cout << "\n" << i << " genes read.\n" << std::flush;
    
    ret=0;
    ret += std::fscanf(genes_file,"\nTranscription Factors (tf_difusibility,tf_breakdown_rate) \n" );
    for(i=0; i<num_tf; i++) { 
        ret += std::fscanf(genes_file,"\t%u,\t",&m_FGenome.tf_diffusability[i] );
        ret += std::fscanf(genes_file,"%u,\t",&m_FGenome.tf_breakdown_rate[i] );
        if (m_FParams.debug>1)std::cout <<"\t("<< m_FGenome.tf_diffusability[i] <<"," << m_FGenome.tf_breakdown_rate[i] <<"),"<< std::flush;
    }
    if (ret != NUM_TF*2)if (m_FParams.debug>1) std::cout << "\nvoid FluidSystem::ReadGenome, Transcription Factor read failure !  gene number = " << i << ", ret = "<< ret <<"\n " << std::flush;
    if (m_FParams.debug>1)std::cout << "\n" << i << " transcription factors read.\n" << std::flush;
    
    ret=0;
    ret += std::fscanf(genes_file, "\nRemodelling parameters, rows : elastin,collagen,apatite\n" );
    ret += std::fscanf(genes_file, "\ncollumns : /*triggering bond parameter changes*/, /*triggering particle changes*/, /*initial values for new bonds*/" );
    ret += std::fscanf(genes_file, "\nelongation_threshold,\telongation_factor,\tstrength_threshold,\tstrengthening_factor,\t\t\
    max_rest_length,\tmin_rest_length,\tmax_modulus,\tmin_modulus,\t\t\t\
    elastLim,\tdefault_rest_length,\tdefault_modulus,\tdefault_damping\n");
    
    for(i=0; i<3; i++){
        for(j=0; j<12;j++) ret += std::fscanf(genes_file, "\t%f,\t", &m_FGenome.param[i][j] ); 
        ret += std::fscanf(genes_file, "\n");
    }
    if (m_FParams.debug>1)std::cout << "\n" << i <<"*"<< j << " remodelling parameters read. ret = "<< ret <<"\n" << std::flush;
    
    fclose(genes_file);
}

void FluidSystem::WriteGenome( const char * relativePath){
    if (m_FParams.debug>1)std::cout << "\n  FluidSystem::WriteGenome( const char * relativePath)  started \n" << std::flush;
    char buf[256];
    sprintf ( buf, "%s/genome.csv", relativePath );
    FILE* fp = fopen ( buf, "w" );
    if (fp == NULL) {
        if (m_FParams.debug>1)std::cout << "\nvoid FluidSystem::WriteGenome( const char * relativePath)  Could not open file "<< fp <<"\n"<< std::flush;
        assert(0);
    }
    fprintf(fp, "num genes = %i,\tnum transcription factors = %i \n", NUM_GENES, NUM_TF );
    
    fprintf(fp, "mutability, delay,\tsensitivity[NUM_GENES]");
    for(int i=0; i<NUM_GENES; i++)fprintf(fp, ",\t");
    
    fprintf(fp, "secrete[NUM_TF][2]");
    for(int i=0; i<NUM_TF; i++)fprintf(fp, ",,\t");
    
    fprintf(fp, "secrete[2*NUM_TF], ");
    
    fprintf(fp, "activate[NUM_GENES][2]");
    for(int i=0; i<NUM_GENES; i++)fprintf(fp, ",,\t");
    
    fprintf(fp, "activate[2*NUM_GENES] \n" );
    
    for(int i=0; i<NUM_GENES; i++) {
        fprintf(fp, "%i,\t", m_FGenome.mutability[i] );
        fprintf(fp, "%i,\t\t", m_FGenome.delay[i] );
        
        for(int j=0; j<NUM_GENES; j++) fprintf(fp, "%i,\t", m_FGenome.sensitivity[i][j] );
        fprintf(fp, " \t\t" );
        for(int j=0; j<NUM_TF; j++) fprintf(fp, "%i,%i,\t", m_FGenome.secrete[i][j*2], m_FGenome.secrete[i][j*2 + 1] );//secretion_rate
        
        fprintf(fp, "\t\t%i,\t\t", m_FGenome.secrete[i][2*NUM_TF] );    //num active elements, NB sparse list, kernel will only read active elems.
        
        for(int j=0; j<NUM_GENES; j++) fprintf(fp, "%i,%i,\t", m_FGenome.activate[i][j*2], m_FGenome.activate[i][j*2 + 1] );
        
        fprintf(fp, "\t\t%i,\t\t", m_FGenome.activate[i][2*NUM_GENES] );  
        fprintf(fp, " \n" );
    }
    fprintf(fp, "\nTranscription Factors (tf_difusibility,tf_breakdown_rate) \n" );
    for(int i=0; i<NUM_TF; i++) {    
        fprintf(fp, "\t%i,\t", m_FGenome.tf_diffusability[i] );
        fprintf(fp, "%i,\t", m_FGenome.tf_breakdown_rate[i] );
    }
    fprintf(fp, "\n\nRemodelling parameters, rows : elastin,collagen,apatite" );
    fprintf(fp, "\ncollumns : /*triggering bond parameter changes*/, /*triggering particle changes*/, /*initial values for new bonds*/" );
    fprintf(fp, "\nelongation_threshold,\telongation_factor,\tstrength_threshold,\tstrengthening_factor,\t\t\
    max_rest_length,\tmin_rest_length,\tmax_modulus,\tmin_modulus,\t\t\t\
    elastLim,\tdefault_rest_length,\tdefault_modulus,\tdefault_damping\n");
    
    for(int i=0; i<3; i++){
        for(int j=0; j<12;j++) fprintf(fp, "\t%f,\t", m_FGenome.param[i][j]);
        fprintf(fp, "\n");
    }
    
    fprintf(fp, "\n\n\
    struct FGenome{   // ## currently using fixed size genome for efficiency. NB Particle data size depends on genome size.\n\
        uint mutability[NUM_GENES];\n\
        uint delay[NUM_GENES];\n\
        uint sensitivity[NUM_GENES][NUM_GENES];     // for each gene, its sensitivity to each TF or morphogen\n\
        uint tf_diffusability[NUM_TF];              // for each transcription_factor, the diffusion and breakdown rates of its TF.\n\
        uint tf_breakdown_rate[NUM_TF];\n\
                                                    // sparse lists final entry = num elem, other entries (elem_num, param)\n\
        int secrete[NUM_GENES][2*NUM_TF+1];         // -ve secretion => active breakdown. Can be useful for pattern formation.\n\
        int activate[NUM_GENES][2*NUM_GENES+1];\n\
        //uint *function[NUM_GENES];                // cancelled// Hard code a case-switch that calls each gene's function iff the gene is active.\n\
        enum {elastin,collagen,apatite};\n\
                                                    //FBondParams fbondparams[3];   // 0=elastin, 1=collagen, 2=apatite\n\
        \n\
        enum params{  /*triggering bond parameter changes*/ elongation_threshold,   elongation_factor,      strength_threshold,     strengthening_factor,\n\
                      /*triggering particle changes*/       max_rest_length,        min_rest_length,        max_modulus,            min_modulus,\n\
                      /*initial values for new bonds*/      elastLim,               default_rest_length,    default_modulus,        default_damping\n\
        };\n\
        float param[3][12];\n\
    };\n\
    ");
    
    fclose(fp);
}

void FluidSystem::SavePointsVTP2 ( const char * relativePath, int frame ){// uses vtk library to write binary vtp files
    // based on VtpWrite(....)demo at https://vtk.org/Wiki/Write_a_VTP_file  (30 April 2009)
    // and on https://lorensen.github.io/VTKExamples/site/Cxx/IO/WriteVTP/   (post vtk-8.90.9)

    // Header information:  ?? how can this be added ??
    //  A) fparams & fgenome
    //  B) header of the.csv file, giving sizes of arrays.
    
    // points, vertices & lines
    // points & vertices = FPOS 3df
    vtkSmartPointer<vtkPoints> points3D = vtkSmartPointer<vtkPoints>::New();                           // Points3D
	vtkSmartPointer<vtkCellArray> Vertices = vtkSmartPointer<vtkCellArray>::New();                     // Vertices
    uint num_active_points = 0;
    for ( unsigned int i = 0; i < mMaxPoints; ++i )
	{	
        if( *getParticle_ID(i)<=mMaxPoints ){ // if(active_particle)
            vtkIdType pid[1];
            //Point P = Model.Points[i];
            Vector3DF* Pos = getPos(i); 
            pid[0] = points3D->InsertNextPoint(Pos->x, Pos->y, Pos->z);
            Vertices->InsertNextCell(1,pid);
            num_active_points++;
        }else break;
	}
	
	// Inset vertices for sim volume
	{
        vtkIdType pid[1];
        Vector3DF  pos = {0,0,0};
        Vector3DF* Pos =&pos;
        for(int corner=0; corner<8; corner++){
            if(corner&1) Pos->x = m_Vec[ PVOLMAX ].x; // bitmask to select axes to swap to PVOLMAX
            else Pos->x = m_Vec[ PVOLMIN ].x;
            if(corner&2) Pos->y = m_Vec[ PVOLMAX ].y;
            else Pos->y = m_Vec[ PVOLMIN ].y;
            if(corner&4) Pos->z = m_Vec[ PVOLMAX ].z; 
            else Pos->z = m_Vec[ PVOLMIN ].z;
            pid[0] = points3D->InsertNextPoint(Pos->x, Pos->y, Pos->z);
            Vertices->InsertNextCell(1,pid);
        }
    }
	
    // edges = FELASTIDX [0]current index uint                                                         // Lines
    vtkSmartPointer<vtkCellArray> Lines = vtkSmartPointer<vtkCellArray>::New();
    uint *ElastIdx;
    float *ElastIdxPtr;
    for ( unsigned int i = 0; i < num_active_points; ++i )
	{	
        ElastIdx = getElastIdx(i);
        //ElastIdxPtr = (float*)ElastIdx;
        for(int j=0; j<(BONDS_PER_PARTICLE ); j++) { 
            uint secondParticle = ElastIdx[j * DATA_PER_BOND];
            uint bond = ElastIdx[j * DATA_PER_BOND +2];          // NB [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index 
            if (bond==0 || bond==UINT_MAX || secondParticle==UINT_MAX) secondParticle = i;                    // i.e. if [2]restlength, then bond is broken, therefore bond to self.
            vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
            line->GetPointIds()->SetId(0,i);
            line->GetPointIds()->SetId(1,secondParticle);
            Lines->InsertNextCell(line);
        }
	}
	// Sim volume boundry lines
	for(int corner=0; corner<8; corner++){
    int firstParticle = corner + num_active_points;
    vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
    if(corner&1){ // Pos->x = m_Vec[ PVOLMAX ].x; // bitmask to select axes to swap to PVOLMAX
        int secondParticle = corner -1 + num_active_points;
        line->GetPointIds()->SetId(0,firstParticle);
        line->GetPointIds()->SetId(1,secondParticle);
        Lines->InsertNextCell(line);
    }
    if(corner&2){ // Pos->y = m_Vec[ PVOLMAX ].y;
        int secondParticle = corner -2 + num_active_points;
        line->GetPointIds()->SetId(0,firstParticle);
        line->GetPointIds()->SetId(1,secondParticle);
        Lines->InsertNextCell(line);
    }
    if(corner&4){ // Pos->z = m_Vec[ PVOLMAX ].z; 
        int secondParticle = corner -4 + num_active_points;
        line->GetPointIds()->SetId(0,firstParticle);
        line->GetPointIds()->SetId(1,secondParticle);
        Lines->InsertNextCell(line);
    }
    //pid[0] = points3D->InsertNextPoint(Pos->x, Pos->y, Pos->z);
    //Vertices->InsertNextCell(1,pid);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////// Particle Data 
    
    // FELASTIDX bond data, float and uint vtkDataArrays, stored in particles
    vtkSmartPointer<vtkUnsignedIntArray> BondsUIntData = vtkSmartPointer<vtkUnsignedIntArray>::New();
    BondsUIntData->SetNumberOfComponents(3);
	BondsUIntData->SetName("curr_idx, particle ID, bond index");
    
    vtkSmartPointer<vtkFloatArray> BondsFloatData = vtkSmartPointer<vtkFloatArray>::New();
    BondsFloatData->SetNumberOfComponents(6);
	BondsFloatData->SetName("elastic limit, restlength, modulus, damping coeff, stress integrator");
    
    
    for ( unsigned int i = 0; i < num_active_points; ++i )
	{
        ElastIdx = getElastIdx(i);                     // FELASTIDX[BONDS_PER_PARTICLE]  [0]current index uint, [5]particle ID uint, [6]bond index uint
        ElastIdxPtr = (float*)ElastIdx;                // FELASTIDX[BONDS_PER_PARTICLE]  [1]elastic limit float, [2]restlength float, [3]modulus float, [4]damping coeff float,
        for(int j=0; j<(BOND_DATA); j+=DATA_PER_BOND) { 
            BondsUIntData->InsertNextTuple3(ElastIdx[j], ElastIdx[j+5], ElastIdx[j+6]);
            //BondsFloatData->InsertNextTuple6(ElastIdxPtr[j+1], ElastIdxPtr[j+2], ElastIdxPtr[j+3], ElastIdxPtr[j+4], ElastIdxPtr[j+7], 0);
            BondsFloatData->InsertNextTuple6(ElastIdx[j+1], ElastIdx[j+2], ElastIdx[j+3], ElastIdx[j+4], ElastIdx[j+7], 0);
        }
    }
    for(int corner=0; corner<8; corner++){
        for(int j=0; j<(BOND_DATA); j+=DATA_PER_BOND) { 
            BondsUIntData->InsertNextTuple3(0,0,0);
            BondsFloatData->InsertNextTuple6(0,0,0,0,0,0);
        }
    }
    //BondsUIntData->SetNumberOfComponents(BONDS_PER_PARTICLE *3);
    //BondsFloatData->SetNumberOfComponents(BONDS_PER_PARTICLE *4); 
    

    // FVEL 3df, 
    Vector3DF* Vel;
    vtkSmartPointer<vtkFloatArray> fvel = vtkSmartPointer<vtkFloatArray>::New();
    fvel->SetNumberOfComponents(3);
	fvel->SetName("FVEL");
    for(unsigned int i=0;i<num_active_points;i++){
        Vel = getVel(i);
        fvel->InsertNextTuple3(Vel->x,Vel->y,Vel->z);
    }
    for(int corner=0; corner<8; corner++){
        fvel->InsertNextTuple3(0,0,0);
    }
    fvel->SetNumberOfComponents(BONDS_PER_PARTICLE *3);
    
    
/*    // FVEVAL 3df, 
    Vector3DF* Veval;
    vtkSmartPointer<vtkFloatArray> fveval = vtkSmartPointer<vtkFloatArray>::New();
    fvel->SetNumberOfComponents(3);
	fvel->SetName("FVEVAL");
    for(unsigned int i=0;i<num_active_points;i++){
        Veval = getVeval(i);
        fveval->InsertNextTuple3(Veval->x,Veval->y,Veval->z);
    }
    fveval->SetNumberOfComponents(BONDS_PER_PARTICLE *3);
*/
    
/*    // FFORCE 3df, 
    Vector3DF* Force;
    vtkSmartPointer<vtkFloatArray> fforce = vtkSmartPointer<vtkFloatArray>::New();
    fforce->SetNumberOfComponents(3);
	fforce->SetName("FFORCE");
    for(unsigned int i=0;i<num_active_points;i++){
        Force = getForce(i);
        fforce->InsertNextTuple3(Force->x,Force->y,Force->z);
    }
    fforce->SetNumberOfComponents(BONDS_PER_PARTICLE *3);
*/
    
    
/*    // FPRESS f,
    float* Pres;
    vtkSmartPointer<vtkFloatArray> fpres = vtkSmartPointer<vtkFloatArray>::New();
    fpres->SetNumberOfComponents(1);
	fpres->SetName("FPRESS");
    for(unsigned int i=0;i<num_active_points;i++){
        Pres = getPres(i);
        fpres->InsertNextTuple(Pres);
    }
*/     
    
/*    // FDENSITY f, 
    float* Dens;
    vtkSmartPointer<vtkFloatArray> fdens = vtkSmartPointer<vtkFloatArray>::New();
    fdens->SetNumberOfComponents(1);
	fdens->SetName("FDENSITY");
    for(unsigned int i=0;i<num_active_points;i++){
        Dens = getDensity(i);
        fdens->InsertNextTuple(Dens);
    }
*/
    
    // FAGE ushort, 
    unsigned int* age = getAge(0);
    vtkSmartPointer<vtkUnsignedIntArray> fage = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fage->SetNumberOfComponents(1);
	fage->SetName("FAGE");
    for(unsigned int i=0;i<num_active_points;i++){
        fage->InsertNextValue(age[i]);
    }
    for(int corner=0; corner<8; corner++){
        fage->InsertNextValue(0);
    }
    
    // FCLR uint, 
    unsigned int* color = getClr(0);
    vtkSmartPointer<vtkUnsignedIntArray> fcolor = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fcolor->SetNumberOfComponents(1);
	fcolor->SetName("FCLR");
    for(unsigned int i=0;i<num_active_points;i++){
        fcolor->InsertNextValue(color[i]);
    }
    for(int corner=0; corner<8; corner++){
        fcolor->InsertNextValue(0);
    }
    
    // FGCELL	uint, 
    
    // FPARTICLEIDX uint[BONDS_PER_PARTICLE *2],  

    
    // FPARTICLE_ID  uint, 
    unsigned int* pid = getParticle_ID(0);
    vtkSmartPointer<vtkUnsignedIntArray> fpid = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fpid->SetNumberOfComponents(1);
	fpid->SetName("FPARTICLE_ID");
    for(unsigned int i=0;i<num_active_points;i++){
        fpid->InsertNextValue(pid[i]);
    }
    for(int corner=0; corner<8; corner++){
        fpid->InsertNextValue(0);
    }
    
    // FMASS_RADIUS uint (holding modulus 16bit and limit 16bit.),    
    unsigned int* Mass_Radius = getMass_Radius(0);
    uint mass, radius;
    vtkSmartPointer<vtkUnsignedIntArray> fmass_radius = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fmass_radius->SetNumberOfComponents(2);
	fmass_radius->SetName("FMASS_RADIUS");
    for(unsigned int i=0;i<num_active_points;i++){
        if(Mass_Radius[i]==0){   mass = 0; }else{  mass = Mass_Radius[i]; } 
        radius = mass >> 16;
        mass = mass & TWO_POW_16_MINUS_1;
        fmass_radius->InsertNextTuple2(mass,radius);
    }
    for(int corner=0; corner<8; corner++){
        fmass_radius->InsertNextTuple2(0,0);
    }
    
    // FNERVEIDX uint, 
    unsigned int* nidx = getNerveIdx(0);
    vtkSmartPointer<vtkUnsignedIntArray> fnidx = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fnidx->SetNumberOfComponents(1);
	fnidx->SetName("FNERVEIDX");
    for(unsigned int i=0;i<num_active_points;i++){
        fnidx->InsertNextValue(nidx[i]);
    }
    for(int corner=0; corner<8; corner++){
        fnidx->InsertNextValue(0);
    }
    
    // FCONC float[NUM_TF].                                                                                     // commented out until Matt's edit FCONC uint->foat is merged
    vtkSmartPointer<vtkFloatArray> fconc[NUM_TF];
    char buf_conc[256];
    for (int a=0; a<NUM_GENES; a++){ 
        fconc[a] = vtkSmartPointer<vtkFloatArray>::New();
        fconc[a]->SetNumberOfComponents(1);
        sprintf ( buf_conc, "FCONC_%i",a);
        fconc[a]->SetName(buf_conc);
    }
    float *conc;
    for ( unsigned int i = 0; i < NUM_GENES; ++i ){
        conc = getConc(i);                   
        for(int j=0; j<num_active_points; j++){
            fconc[i]->InsertNextValue(conc[j]);                              // now have one array for each column of fepigen
        }
        for(int corner=0; corner<8; corner++){
            fconc[i]->InsertNextValue(0);
        }
    }
    
    // FEPIGEN uint[NUM_GENES] ... make an array of arrays
    vtkSmartPointer<vtkUnsignedIntArray> fepigen[NUM_GENES];
    char buf_epigen[256];
    for (int a=0; a<NUM_GENES; a++){ 
        fepigen[a] = vtkSmartPointer<vtkUnsignedIntArray>::New();
        fepigen[a]->SetNumberOfComponents(1);
        sprintf ( buf_epigen, "FEPIGEN_%i",a);
        fepigen[a]->SetName(buf_epigen);
    }
    unsigned int *epigen;
    for ( unsigned int i = 0; i < NUM_GENES; ++i ){
        epigen = getEpiGen(i);                   
        for(int j=0; j<num_active_points; j++){    
            fepigen[i]->InsertNextValue(epigen[j]);                              // now have one array for each column of fepigen
        }
        for(int corner=0; corner<8; corner++){
            fepigen[i]->InsertNextValue(0);
        }
    }

    
    // F_TISSUE_TYPE  uint, 
    unsigned int tissueType;
    vtkSmartPointer<vtkUnsignedIntArray> ftissue = vtkSmartPointer<vtkUnsignedIntArray>::New();
    ftissue->SetNumberOfComponents(1);
	ftissue->SetName("F_TISSUE_TYPE");
    unsigned int *epigen_[NUM_GENES];
    for ( unsigned int i = 0; i < NUM_GENES; ++i )  epigen_[i] = getEpiGen(i);
    
    for(unsigned int i=0;i<num_active_points;i++){
        if      (epigen_[9][i] >0/*bone*/)      tissueType =9;                                                   
        else if (epigen_[6][i] >0/*tendon*/)    tissueType =6;
        else if (epigen_[7][i] >0/*muscle*/)    tissueType =7;
        else if (epigen_[10][i]>0/*elast lig*/) tissueType =10;
        else if (epigen_[8][i] >0/*cartilage*/) tissueType =8;
        else                                    tissueType =0;
        ftissue->InsertNextValue(tissueType);
    }
    for(int corner=0; corner<8; corner++){
        ftissue->InsertNextValue(0);
    }
    
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // POLYDATA
	vtkSmartPointer<vtkPolyData> polydata = vtkPolyData::New();                                        // polydata
	polydata->SetPoints(points3D);
	//polydata->SetVerts(Vertices);
    polydata->SetLines(Lines);
    
    
    //if (m_FParams.debug>1)cout << "\nStarting writing bond data to polydata\n" << std::flush;
    polydata->GetCellData()->AddArray(BondsUIntData);
    polydata->GetCellData()->AddArray(BondsFloatData);
    //polydata->GetPointData()->AddArray(BondsUIntData);
    //polydata->GetPointData()->AddArray(BondsFloatData);
    polydata->GetPointData()->AddArray(fage);
    polydata->GetPointData()->AddArray(fcolor);
    polydata->GetPointData()->AddArray(fpid);
    polydata->GetPointData()->AddArray(fmass_radius);
    polydata->GetPointData()->AddArray(fnidx);
    
    for(int i=0;i<NUM_TF; i++)      polydata->GetPointData()->AddArray(fconc[i]);
    for(int i=0;i<NUM_GENES; i++)   polydata->GetPointData()->AddArray(fepigen[i]);
    
    polydata->GetPointData()->AddArray(ftissue);
    
    //if (m_FParams.debug>1)cout << "\nFinished writing bond data to polydata\n" << std::flush;
    
    // WRITER  
	vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();       // writer
    char buf[256];
    frame += 100000;                                                                                              // ensures numerical and alphabetic order match of filenames
    sprintf ( buf, "%s/particles_pos_vel_color%04d.vtp", relativePath, frame );
	writer->SetFileName(buf);
	writer->SetInputData(polydata);
    writer->SetDataModeToAscii();   
    //writer->SetDataModeToAppended();    // prefered, produces a human readable header followed by a binary blob.
    //writer->SetDataModeToBinary();
	writer->Write();
    
	//if (m_FParams.debug>1)cout << "\nFinished writing vtp file " << buf << "." << endl;
	//if (m_FParams.debug>1)cout << "\tnum_active_points: " << num_active_points << endl;
}

void FluidSystem::SavePointsCSV2 ( const char * relativePath, int frame ){
    if (m_FParams.debug>1) std::cout << "\n  SavePointsCSV2 ( const char * relativePath = "<< relativePath << ", int frame = "<< frame << " );  started \n" << std::flush;
    char buf[256];
    frame += 100000;    // ensures numerical and alphabetic order match
    sprintf ( buf, "%s/particles_pos_vel_color%04d.csv", relativePath, frame );
    FILE* fp = fopen ( buf, "w" );
    if (fp == NULL) {
        if (m_FParams.debug>1) std::cout << "\nvoid FluidSystem::SavePointsCSV ( const char * relativePath, int frame )  Could not open file "<< fp <<"\n"<< std::flush;
        assert(0);
    }
    int numpnt = mMaxPoints;//NumPoints();
    Vector3DF* Pos;
    Vector3DF* Vel;
    float *Conc;
    uint* Age, *Clr, *NerveIdx, *ElastIdx, *Particle_Idx, *Particle_ID, *Mass_Radius, *EpiGen;                  // Q: why are these pointers? A: they get dereferenced below.
    uint mass, radius;
    float *ElastIdxPtr;
    
    fprintf(fp, "i,, x coord, y coord, z coord\t\t x vel, y vel, z vel\t\t age,  color\t\t FELASTIDX[%u*%u]", BONDS_PER_PARTICLE, DATA_PER_BOND);  // This system inserts commas to align header with csv data
    for (int i=0; i<BONDS_PER_PARTICLE; i++)fprintf(fp, ",(%u)[0]curIdx, [1]elastLim, [2]restLn, [3]modulus, [4]damping, [5]partID, [6]bond index, [7]stress integrator, [8]change-type,,  ",i);
    fprintf(fp, "\t"); 
    fprintf(fp, "\tParticle_ID, mass, radius, FNERVEIDX,\t\t Particle_Idx[%u*2]", BONDS_PER_PARTICLE);    
    for (int i=0; i<BONDS_PER_PARTICLE; i++)fprintf(fp, "%u,,, ",i);
    fprintf(fp, "\t\tFCONC[%u] ", NUM_TF);
    for (int i=0; i<NUM_TF; i++)fprintf(fp, "%u, ",i);
    fprintf(fp, "\t\tFEPIGEN[%u] ", NUM_GENES);
    for (int i=0; i<NUM_GENES; i++)fprintf(fp, "%u, ",i);
    fprintf(fp, "\n");

    for(int i=0; i<numpnt; i++) {       // nb need get..() accessors for private data.
        Pos = getPos(i);                // e.g.  Vector3DF* getPos ( int n )	{ return &m_Fluid.bufV3(FPOS)[n]; }
        Vel = getVel(i);
        Age = getAge(i);
        Clr = getClr(i);
        ElastIdx = getElastIdx(i);      // NB [BONDS_PER_PARTICLE]
      //if (m_FParams.debug>1)printf("\t%u,",ElastIdx[0]);
        ElastIdxPtr = (float*)ElastIdx; // #############packing floats and uints into the same array - should replace with a struct.#################
        Particle_Idx = getParticle_Idx(i);
        Particle_ID = getParticle_ID(i);//# uint  original pnum, used for bonds between particles. 32bit, track upto 4Bn particles.
        if(*Particle_ID==0){
         if (m_FParams.debug>1) std::cout << "SavePointsCSV2: Particle_ID = pointer not assigned. i="<<i<<". \t" << std::flush;
         return;
        }
        // ? should I be splitting mass_radius with bitshift etc  OR just use two uit arrays .... where are/will these used anyway ?
        Mass_Radius = getMass_Radius(i);//# uint holding modulus 16bit and limit 16bit.
        if(*Mass_Radius==0){   mass = 0; }else{  mass = *Mass_Radius; }    // modulus          // '&' bitwise AND is bit masking. ;
        radius = mass >> 16;
        mass = mass & TWO_POW_16_MINUS_1;
        
        NerveIdx = getNerveIdx(i);      //# uint
        //Conc = getConc(i);              //# float[NUM_TF]        NUM_TF = num transcription factors & morphogens
        //EpiGen = getEpiGen(i);          //# uint[NUM_GENES]  see below.
        
        fprintf(fp, "%u,,%f,%f,%f,\t%f,%f,%f,\t %u, %u,, \t", i, Pos->x, Pos->y,Pos->z, Vel->x,Vel->y,Vel->z, *Age, *Clr );
        //if (m_FParams.debug>1) std::cout<<"\t"<<Pos->z<<std::flush;
        for(int j=0; j<(BOND_DATA); j+=DATA_PER_BOND) { 
            fprintf(fp, "%u, %f, %f, %f, %f, %u, %u, %f, %u, ", ElastIdx[j], ElastIdxPtr[j+1], ElastIdxPtr[j+2], ElastIdxPtr[j+3], ElastIdxPtr[j+4], ElastIdx[j+5], ElastIdx[j+6], ElastIdxPtr[j+7], ElastIdx[j+8] );
            
           /*
            // if ((j%DATA_PER_BOND==0)||((j+1)%DATA_PER_BOND==0))  fprintf(fp, "%u, ",  ElastIdx[j] );  // print as int   [0]current index, [5]particle ID, [6]bond index 
           // else  fprintf(fp, "%f, ",  ElastIdxPtr[j] );                                               // print as float [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, 
           //  if((j+1)%DATA_PER_BOND==0)  
            */
            fprintf(fp, "\t\t");
        }
        fprintf(fp, " \t%u, %u, %u, %u, \t\t", *Particle_ID, mass, radius, *NerveIdx );
        for(int j=0; j<(BONDS_PER_PARTICLE*2); j+=2)   { fprintf(fp, "%u, %u,, ",  Particle_Idx[j], Particle_Idx[j+1] );}  fprintf(fp, "\t\t"); // NB index of other particle AND other particle's index of the bond
        
        for(int j=0; j<(NUM_TF); j++)               { 
            Conc = getConc(j);
            fprintf(fp, "%f, ",  Conc[i] ); 
        }fprintf(fp, "\t\t");
        
        for(int j=0; j<(NUM_GENES); j++)            { 
            EpiGen = getEpiGen(j);
            fprintf(fp, "%u, ",  EpiGen[i] );   // NB FEPIGEN[gene][particle], for memory efficiency on the device. ? Need to test.
        }fprintf(fp, " \n");
    }
    fclose ( fp );
    fflush ( fp );
}

void FluidSystem::ReadPointsCSV2 ( const char * relativePath, int gpu_mode, int cpu_mode){ // NB allocates buffers as well.
    //if (m_FParams.debug>1) std::cout << "\n  ReadPointsCSV2 ( const char * relativePath, int gpu_mode, int cpu_mode);  started \n" << std::flush;
    const char * points_file_path = relativePath;
    if (m_FParams.debug>1)printf("\n## opening file %s ", points_file_path);
    FILE * points_file = fopen(points_file_path, "rb");
    if (points_file == NULL) {
        assert(0);
    }
    // find number of lines = number of particles
    int ch, number_of_lines = 0;
    while (EOF != (ch=getc(points_file)))   if ('\n' == ch)  ++number_of_lines;

    // Allocate buffers for points
    m_Param [PNUM] = number_of_lines -1;                                    // NB there is a line of text above the particles, hence -1.
    mMaxPoints = m_Param [PNUM];
    m_Param [PGRIDSIZE] = 2*m_Param[PSMOOTHRADIUS] / m_Param[PGRID_DENSITY];

    SetupSPH_Kernels ();
    SetupSpacing ();
    //std::cout<<"\n\nReadPointsCSV2: SetupGrid ( "<<m_Vec[PVOLMIN].x <<", "<<m_Vec[PVOLMIN].y<<", "<<m_Vec[PVOLMIN].z<<", "<<m_Vec[PVOLMAX].x<<", "<<m_Vec[PVOLMAX].y<<", "<<m_Vec[PVOLMAX].z<<", "<<m_Param[PSIMSCALE]<<", "<<m_Param[PGRIDSIZE]<<")\n"<<std::flush;
    
    SetupGrid ( m_Vec[PVOLMIN]/*bottom corner*/, m_Vec[PVOLMAX]/*top corner*/, m_Param[PSIMSCALE], m_Param[PGRIDSIZE]);
    if (gpu_mode != GPU_OFF) {     // create CUDA instance etc.. 
        FluidSetupCUDA ( mMaxPoints, m_GridSrch, *(int3*)& m_GridRes, *(float3*)& m_GridSize, *(float3*)& m_GridDelta, *(float3*)& m_GridMin, *(float3*)& m_GridMax, m_GridTotal, 0 );
        UpdateParams();            //  sends simulation params to device.
        UpdateGenome();            //  sends genome to device.              // NB need to initialize genome from file, or something.
    }
    
    //cout<<"ReadPointsCSV2: AllocateParticles ( "<<mMaxPoints<<", "<<gpu_mode<<", "<<cpu_mode<<" )"<<std::flush;
    AllocateParticles ( mMaxPoints, gpu_mode, cpu_mode );  // allocates only cpu buffer for particles
    AllocateGrid(gpu_mode, cpu_mode);
    //////////////////////////////////////// 
    uint Clr, Age;
    Vector3DF Pos, Vel, PosMin, PosMax;
    uint  ElastIdxU[BOND_DATA];
    float ElastIdxF[BOND_DATA];
    uint Particle_Idx[BONDS_PER_PARTICLE * 2];
    uint Particle_ID, mass, radius, Mass_Radius, NerveIdx, discard_uint;
    float Conc[NUM_TF];
    uint EpiGen[NUM_GENES];
    
    float vel_lim = GetParam ( PVEL_LIMIT );
    PosMin = GetVec ( PBOUNDMIN );  //PBOUNDMIN  // PVOLMIN
    PosMax = GetVec ( PBOUNDMAX );  //PBOUNDMAX  // PVOLMAX

    if (m_FParams.debug>1) cout<<"\n\nPosMin = PBOUNDMIN=("<<m_Vec[PBOUNDMIN].x <<","<<m_Vec[PBOUNDMIN].y <<","<<m_Vec[PBOUNDMIN].z 
        <<"),  PosMax = PBOUNDMAX=("<<m_Vec[PBOUNDMAX].x <<","<<m_Vec[PBOUNDMAX].y <<","<< m_Vec[PBOUNDMAX].z
        <<"),  PVOLMIN=("<<m_Vec[PVOLMIN].x <<","<<m_Vec[PVOLMIN].y <<","<<m_Vec[PVOLMIN].z 
        <<"),  PVOLMAX=("<<m_Vec[PVOLMAX].x <<","<<m_Vec[PVOLMAX].y <<","<<m_Vec[PVOLMAX].z 
        <<")\n"<<std::flush;
    
    std::fseek(points_file, 0, SEEK_SET);
    uint bond_data=999, data_per_bond=999, bonds_per_particle=999, num_TF=999, num_genes=999;
    int result=-2;
    result = std::fscanf(points_file, "i,, x coord, y coord, z coord\t\t x vel, y vel, z vel\t\t age,  color\t\t FELASTIDX[%u*%u]", &bond_data, &data_per_bond);
//if (m_FParams.debug>1) std::cout<<"\n\n ReadPointsCSV2() line 1241: scanf result="<<result<<"\n"<<std::flush; 
    for (int i=0; i<data_per_bond; i++) result+=std::fscanf(points_file, ",(%u)[0]curIdx, [1]elastLim, [2]restLn, [3]modulus, [4]damping, [5]partID, [6]bond index, [7]stress integrator, [8]change-type,,  ",&discard_uint);
    bond_data = bond_data * data_per_bond;
    result += fscanf(points_file, "\t");
//if (m_FParams.debug>1) std::cout<<"\n ReadPointsCSV2() line 1246: scanf result="<<result<<"\n"<<std::flush; 
    result = std::fscanf(points_file, "\tParticle_ID, mass, radius, FNERVEIDX,\t\t Particle_Idx[%u*2]", &bonds_per_particle);
    for (int i=0; i<BONDS_PER_PARTICLE; i++) result+=fscanf(points_file, "%u,,, ",&discard_uint);
//if (m_FParams.debug>1) std::cout<<"\n ReadPointsCSV2() line 1249: scanf result="<<result<<"\n"<<std::flush;     
    result = std::fscanf(points_file, "\t\tFCONC[%u]",&num_TF);
    for (int i=0; i<NUM_TF; i++)result += fscanf(points_file, "%u, ",&discard_uint);
//if (m_FParams.debug>1) std::cout<<"\n ReadPointsCSV2() line 1252: scanf result="<<result<<"\n"<<std::flush;     
    result = std::fscanf(points_file, "\t\tFEPIGEN[%u] ", &num_genes );
    for (int i=0; i<NUM_GENES; i++)result += fscanf(points_file, "%u, ",&discard_uint);
    result += fscanf(points_file, "\n");
//if (m_FParams.debug>1) std::cout<<"\n ReadPointsCSV2() line 1254: scanf result="<<result<<"\n"<<std::flush;     
    
if (m_FParams.debug>1) std::cout<<"\n\n ReadPointsCSV2() starting loop: number_of_lines="<<number_of_lines<<"\n"<<std::flush;
    ////////////////////
    int i, index, ret;
    for (i=1; i<number_of_lines; i++ ) {
        // transcribe particle data from file to Pos, Vel and Clr
        ret=0;
        ret += std::fscanf(points_file, "%u,,%f,%f,%f,\t%f,%f,%f,\t %u, %u,, \t",&index, &Pos.x, &Pos.y, &Pos.z, &Vel.x, &Vel.y, &Vel.z, &Age, &Clr );
//if (m_FParams.debug>1) std::cout<<"\n ReadPointsCSV2() row="<< i <<", (line 1259, ret="<<ret<<"),\t"<<std::flush;
        for(int j=0; j<BOND_DATA; j+=DATA_PER_BOND) {// BONDS_PER_PARTICLE * DATA_PER_BOND
            ret += std::fscanf(points_file, "%u, %f, %f, %f, %f, %u, %u, %f, %u, ", &ElastIdxU[j+0], &ElastIdxF[j+1], &ElastIdxF[j+2], &ElastIdxF[j+3], &ElastIdxF[j+4], &ElastIdxU[j+5], &ElastIdxU[j+6], &ElastIdxF[j+7], &ElastIdxU[j+8] );
        }
      //if (m_FParams.debug>1)printf("\t%u\t",ElastIdxU[0]);
//if (m_FParams.debug>1) std::cout<<"(line 1263, ret="<<ret<<")\t"<<std::flush;
        ret += std::fscanf(points_file, " \t%u, %u, %u, %u, \t\t", &Particle_ID, &mass, &radius, &NerveIdx);
        Mass_Radius = mass + (radius << 16);                                    // pack two 16bit uint  into one 32bit uint.
//if (m_FParams.debug>1) std::cout<<"(ReadPointsCSV2() line 1266, ret="<<ret<<"),\t"<<std::flush;
        for(int j=0; j<(BONDS_PER_PARTICLE*2); j+=2) {
            ret += std::fscanf(points_file, "%u, %u,, ",  &Particle_Idx[j], &Particle_Idx[j+1] );
        }
//if (m_FParams.debug>1) std::cout<<"(ReadPointsCSV2() line 1270, ret="<<ret<<"),\t"<<std::flush;        
        for(int j=0; j<(NUM_TF); j++)       {    ret += std::fscanf(points_file, "%f, ",  &Conc[j] );   } ret += std::fscanf(points_file, "\t");
//if (m_FParams.debug>1) std::cout<<"(ReadPointsCSV2() line 1272, ret="<<ret<<"),\t"<<std::flush;
        for(int j=0; j<(NUM_GENES); j++)    {    ret += std::fscanf(points_file, "%u, ",  &EpiGen[j] ); } ret += std::fscanf(points_file, " \n");
//if (m_FParams.debug>1) std::cout<<"(ReadPointsCSV2() line 1274, ret="<<ret<<"),\t"<<std::flush;

if (ret != (9 + BOND_DATA + 4 + BONDS_PER_PARTICLE*2 + NUM_TF + NUM_GENES) ) {  // 9 + 6*9 + 4 + 6*2 + 16 + 16 = 111
            if (m_FParams.debug>1) std::cout<<"\n ReadPointsCSV2() fail line 1276, ret="<<ret<<"\n"<<std::flush;// ret=39
            fclose(points_file);
            return;
        } // ret=8 ret=32 ret=36 ret=48 ret=64 ret=80 

        // check particle is within simulation bounds
        if (Pos.x < PosMin.x || Pos.y < PosMin.y || Pos.z < PosMin.z
                || Pos.x > PosMax.x   || Pos.y > PosMax.y || Pos.z > PosMax.z
                || (Vel.x * Vel.x + Vel.y * Vel.y + Vel.z * Vel.z) > vel_lim * vel_lim )
        {
            //if (m_FParams.debug>1) {
            //  std::cout << "\n void FluidSystem::ReadPointsCSV, out of bounds !  particle number = " << i;
            //  std::cout << "\n Pos.x = " << Pos.x << "  Pos.y = " << Pos.y << "  Pos.z = " << Pos.z;
            //  std::cout << "\n PosMax.x = " << PosMax.x << "  PosMax.y = " << PosMax.y << "  PosMax.z = " << PosMax.z;
            //  std::cout << "\n PosMin.x = " << PosMin.x << "  PosMin.y = " << PosMin.y << "  PosMin.z = " << PosMin.z;
            //  std::cout << "\n velocity = " << sqrt(Vel.x * Vel.x + Vel.y * Vel.y + Vel.z * Vel.z) << "   vel_lim = " << vel_lim;
            //  std::cout << "\n " << std::flush;
            //}
            if (m_FParams.debug>1)printf("\nParticle out of bounds, i=%u\t Pos=(%f,%f,%f), PosMin=(%f,%f,%f), PosMax=(%f,%f,%f), Closing file and exiting.",
                   i, Pos.x, Pos.y, Pos.z, PosMin.x, PosMin.y, PosMin.z, PosMax.x, PosMax.y, PosMax.z );
            fclose(points_file);
            Exit();
        }
        AddParticleMorphogenesis2 (&Pos, &Vel, Age, Clr, ElastIdxU, ElastIdxF, Particle_Idx, Particle_ID, Mass_Radius,  NerveIdx, Conc, EpiGen );
    }
    if (m_FParams.debug>1) std::cout<<"\n ReadPointsCSV2() finished reading points. i="<<i<<", NumPoints()="<<NumPoints()<<"\n"<<std::flush;
    fclose(points_file);
    AddNullPoints ();                                   // add null particles up to mMaxPoints // should be redundant here as mMaxPoints = number_of_lines-1
    if (gpu_mode != GPU_OFF) TransferToCUDA ();         // Initial transfer
  //if (m_FParams.debug>1)printf("\n m_Fluid.gpu(FGRIDOFF_ACTIVE_GENES)=%llu, \t m_Fluid.gpu(FGRIDOFF_CHANGES)=%llu, \t m_Fluid.gpu(FGRIDCNT_CHANGES)=%llu   \n",m_Fluid.gpu(FGRIDOFF_ACTIVE_GENES), m_Fluid.gpu(FGRIDOFF_CHANGES) , m_Fluid.gpu(FGRIDCNT_CHANGES)   );
    if (m_FParams.debug>1) std::cout<<"\n ReadPointsCSV2() finished extra functions. NumPoints()="<<NumPoints()<<"\n"<<std::flush;
}

void FluidSystem::ReadSimParams ( const char * relativePath ) { // transcribe SimParams from file to fluid_system object.
    const char * SimParams_file_path = relativePath;
    if (m_FParams.debug>1)printf ( "\n## opening file %s ", SimParams_file_path );
    FILE * SimParams_file = fopen ( SimParams_file_path, "rb" );
    if ( SimParams_file == NULL ) {
        if (m_FParams.debug>1) std::cout << "\nvoid FluidSystem::ReadSimParams (const char * relativePath )  Could not read file "<< SimParams_file_path <<"\n"<< std::flush;
        assert ( 0 );
    }
    // find number of lines
    int ch, number_of_lines = 0;
    while ( EOF != ( ch=getc ( SimParams_file ) ) )   if ( '\n' == ch )  ++number_of_lines; // chk num lines
    if (m_FParams.debug>1) std::cout << "\nNumber of lines in SimParams_file = " << number_of_lines << std::flush;

    Vector3DF pplane_grav_dir, pvolmin, pvolmax, pinitmin, pinitmax;

    std::fseek(SimParams_file, 0, SEEK_SET);
    int ret = std::fscanf ( SimParams_file, " m_Time = %f\n ", &m_Time );
    ret += std::fscanf ( SimParams_file, "m_DT = %f\n ", &m_DT );
    ret += std::fscanf ( SimParams_file, "m_Param [ PSIMSCALE ] = %f\n ", &m_Param [ PSIMSCALE ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PGRID_DENSITY ] = %f\n ", &m_Param [ PGRID_DENSITY ] ); // added
    ret += std::fscanf ( SimParams_file, "m_Param [ PVISC ] = %f\n ", &m_Param [ PVISC ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PSURFACE_TENSION ] = %f\n ", &m_Param [ PSURFACE_TENSION ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PRESTDENSITY ] = %f\n ", &m_Param [ PRESTDENSITY ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PSPACING ] = %f\n ", &m_Param [ PSPACING ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PMASS ] = %f\n ", &m_Param [ PMASS ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PRADIUS ] = %f\n ", &m_Param [ PRADIUS ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PDIST ] = %f\n ", &m_Param [ PDIST ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PSMOOTHRADIUS ] = %f\n ", &m_Param [ PSMOOTHRADIUS ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PINTSTIFF ] = %f\n ", &m_Param [ PINTSTIFF ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PEXTSTIFF ] = %f\n ", &m_Param [ PEXTSTIFF ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PEXTDAMP ] = %f\n ", &m_Param [ PEXTDAMP ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PACCEL_LIMIT ] = %f\n ", &m_Param [ PACCEL_LIMIT ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PVEL_LIMIT ] = %f\n ", &m_Param [ PVEL_LIMIT ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PGRAV ] = %f\n ", &m_Param [ PGRAV ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PGROUND_SLOPE ] = %f\n ", &m_Param [ PGROUND_SLOPE ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PFORCE_MIN ] = %f\n ", &m_Param [ PFORCE_MIN ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PFORCE_MAX ] = %f\n ", &m_Param [ PFORCE_MAX ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PFORCE_FREQ ] = %f\n ", &m_Param [ PFORCE_FREQ ] );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PPLANE_GRAV_DIR ].Set ( %f, %f, %f )\n ", &pplane_grav_dir.x, &pplane_grav_dir.y, &pplane_grav_dir.z );
    ret += std::fscanf ( SimParams_file, "// Default sim config\n ");
    ret += std::fscanf ( SimParams_file, "m_Param [ PGRIDSIZE ] = %f\n ", &m_Param [ PGRIDSIZE ] );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PVOLMIN ].Set ( %f, %f, %f )\n ", &pvolmin.x, &pvolmin.y, &pvolmin.z );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PVOLMAX ].Set ( %f, %f, %f )\n ", &pvolmax.x, &pvolmax.y, &pvolmax.z );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PINITMIN ].Set ( %f, %f, %f )\n ", &pinitmin.x, &pinitmin.y, &pinitmin.z );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PINITMAX ].Set ( %f, %f, %f )\n ", &pinitmax.x, &pinitmax.y, &pinitmax.z );
    ret += std::fscanf ( SimParams_file, "m_Param [ PFORCE_MIN ] = %f\n ", &m_Param [ PFORCE_MIN ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PFORCE_FREQ ] = %f\n ", &m_Param [ PFORCE_FREQ ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PGROUND_SLOPE ] = %f\n ", &m_Param [ PGROUND_SLOPE ] );

    if ( ret != 41 ) {
        if (m_FParams.debug>1) std::cout << "\nvoid FluidSystem::ReadSimParams(..), read failure ! ret = " << ret << std::flush;
        fclose ( SimParams_file );
        Exit();
    }
    m_Vec [ PPLANE_GRAV_DIR ].Set ( pplane_grav_dir.x, pplane_grav_dir.y, pplane_grav_dir.z );
    // Default sim config
    m_Vec [ PVOLMIN ].Set ( pvolmin.x, pvolmin.y, pvolmin.z );
    m_Vec [ PVOLMAX ].Set ( pvolmax.x, pvolmax.y, pvolmax.z );
    m_Vec [ PINITMIN ].Set ( pinitmin.x, pinitmin.y, pinitmin.z );
    m_Vec [ PINITMAX ].Set ( pinitmax.x, pinitmax.y, pinitmax.z );
    
    if (m_FParams.debug>1) std::cout << "\nvoid FluidSystem::ReadSimParams(..), read success !  ret = " << ret << "\n" << std::flush;
    fclose ( SimParams_file );
    return;
}

void FluidSystem::WriteSimParams ( const char * relativePath ){
    Vector3DF /*point_grav_pos,*/ pplane_grav_dir, /* pemit_pos, pemit_rate, pemit_ang, pemit_dang,*/ pvolmin, pvolmax, pinitmin, pinitmax;

    //int pwrapx, pwall_barrier, plevy_barrier, pdrain_barrier, prun;

    //point_grav_pos = m_Vec [ PPOINT_GRAV_POS ];
    pplane_grav_dir = m_Vec [ PPLANE_GRAV_DIR ];
    //pemit_pos = m_Vec [ PEMIT_POS ];
    //pemit_rate = m_Vec [ PEMIT_RATE ];
    //pemit_ang = m_Vec [ PEMIT_ANG ];
    //pemit_dang = m_Vec [ PEMIT_DANG ];
    pvolmin = m_Vec [ PVOLMIN ];
    pvolmax = m_Vec [ PVOLMAX ];
    pinitmin = m_Vec [ PINITMIN ];
    pinitmax = m_Vec [ PINITMAX ];
/*
    pwrapx = m_Toggle [ PWRAP_X ] ;
    pwall_barrier =  m_Toggle [ PWALL_BARRIER ];
    plevy_barrier = m_Toggle [ PLEVY_BARRIER ];
    pdrain_barrier = m_Toggle [ PDRAIN_BARRIER ];
    prun = m_Toggle [ PRUN ];
*/
    std::cout<<"\nWriteSimParams chk1 "<<std::flush;
    
    // open file to write SimParams to
    char SimParams_file_path[256];
    sprintf ( SimParams_file_path, "%s/SimParams.txt", relativePath );
    if (m_FParams.debug>1)printf("\n## opening file %s ", SimParams_file_path);
    FILE* SimParams_file = fopen ( SimParams_file_path, "w" );
    if (SimParams_file == NULL) {
        if (m_FParams.debug>1) std::cout << "\nvoid FluidSystem::WriteSimParams (const char * relativePath )  Could not open file "<< SimParams_file_path <<"\n"<< std::flush;
        assert(0);
    }
    std::cout<<"\nWriteSimParams chk2,  SimParams_file_path="<<SimParams_file_path<<"\n"<<std::flush;
    /*m_Toggle [ PWRAP_X ] = %i\n m_Toggle [ PWALL_BARRIER ] = %i\n m_Toggle [ PLEVY_BARRIER ] = %i\n m_Toggle [ PDRAIN_BARRIER ] = %i\n */
    /*
                           pwrapx, pwall_barrier, plevy_barrier, pdrain_barrier,*/
    /*  m_Toggle [ PRUN ] = %i\n */
    /*  m_Vec [ PPOINT_GRAV_POS ].Set ( %f, %f, %f )\n */
    /*  m_Param [ PMAX_FRAC ] = %f\n */
    /*  m_Param [ PSTAT_NBRMAX ] = %f\n */
    /*  m_Param [ PSTAT_SRCHMAX ] = %f\n */
    /*  m_Vec [ PEMIT_POS ].Set ( %f, %f, %f )\n */
    /*  m_Vec [ PEMIT_RATE ].Set ( %f, %f, %f )\n m_Vec [ PEMIT_ANG ].Set ( %f, %f, %f )\n m_Vec [ PEMIT_DANG ].Set ( %f, %f, %f )\n */
    
    int ret = std::fprintf(SimParams_file,
                           " m_Time = %f\n m_DT = %f\n m_Param [ PSIMSCALE ] = %f\n m_Param [ PGRID_DENSITY ] = %f\n m_Param [ PVISC ] = %f\n m_Param [ PSURFACE_TENSION ] = %f\n m_Param [ PRESTDENSITY ] = %f\n m_Param [ PSPACING ] = %f\n m_Param [ PMASS ] = %f\n m_Param [ PRADIUS ] = %f\n m_Param [ PDIST ] = %f\n m_Param [ PSMOOTHRADIUS ] = %f\n m_Param [ PINTSTIFF ] = %f\n m_Param [ PEXTSTIFF ] = %f\n m_Param [ PEXTDAMP ] = %f\n m_Param [ PACCEL_LIMIT ] = %f\n m_Param [ PVEL_LIMIT ] = %f\n m_Param [ PGRAV ] = %f\n m_Param [ PGROUND_SLOPE ] = %f\n m_Param [ PFORCE_MIN ] = %f\n m_Param [ PFORCE_MAX ] = %f\n m_Param [ PFORCE_FREQ ] = %f\n m_Vec [ PPLANE_GRAV_DIR ].Set ( %f, %f, %f )\n // Default sim config\n m_Param [ PGRIDSIZE ] = %f\n m_Vec [ PVOLMIN ].Set ( %f, %f, %f )\n m_Vec [ PVOLMAX ].Set ( %f, %f, %f )\n m_Vec [ PINITMIN ].Set ( %f, %f, %f )\n m_Vec [ PINITMAX ].Set ( %f, %f, %f )\n m_Param [ PFORCE_MIN ] = %f\n m_Param [ PFORCE_FREQ ] = %f\n m_Param [ PGROUND_SLOPE ] = %f\n ",
                           m_Time,
                           m_DT,
                           m_Param [ PSIMSCALE ],
                           m_Param [ PGRID_DENSITY ],
                           m_Param [ PVISC ],
                           m_Param [ PSURFACE_TENSION ],
                           m_Param [ PRESTDENSITY ],
                           m_Param [ PSPACING ],
                           m_Param [ PMASS ],
                           m_Param [ PRADIUS ],
                           m_Param [ PDIST ],
                           m_Param [ PSMOOTHRADIUS ],
                           m_Param [ PINTSTIFF ],
                           m_Param [ PEXTSTIFF ],
                           m_Param [ PEXTDAMP ],
                           m_Param [ PACCEL_LIMIT ],
                           m_Param [ PVEL_LIMIT ],
                           //m_Param [ PMAX_FRAC ],
                           m_Param [ PGRAV ],
                           m_Param [ PGROUND_SLOPE ],
                           m_Param [ PFORCE_MIN ],
                           m_Param [ PFORCE_MAX ],
                           m_Param [ PFORCE_FREQ ],
                           //m_Param [ PSTAT_NBRMAX ],
                           //m_Param [ PSTAT_SRCHMAX ],
                           //point_grav_pos.x, point_grav_pos.y, point_grav_pos.z,
                           pplane_grav_dir.x, pplane_grav_dir.y, pplane_grav_dir.z,
                           //pemit_pos.x, pemit_pos.y, pemit_pos.z,
                           //pemit_rate.x, pemit_rate.y, pemit_rate.z,
                           //pemit_ang.x, pemit_ang.y, pemit_ang.z,
                           //pemit_dang.x, pemit_dang.y, pemit_dang.z,
                           // Default sim config
                           //prun,
                           m_Param [ PGRIDSIZE ],
                           pvolmin.x, pvolmin.y, pvolmin.z,
                           pvolmax.x, pvolmax.y, pvolmax.z,
                           pinitmin.x, pinitmin.y, pinitmin.z,
                           pinitmax.x, pinitmax.y, pinitmax.z,
                           m_Param [ PFORCE_MIN ],
                           m_Param [ PFORCE_FREQ ],
                           m_Param [ PGROUND_SLOPE ]
                          );
    std::cout<<"\nWriteSimParams chk3 "<<std::flush;
    
    if (m_FParams.debug>1) std::cout << "\nvoid FluidSystem::WriteSimParams (const char * relativePath ) wrote file "<< SimParams_file_path <<"\t"<<
              "ret = " << ret << "\n" << std::flush;
    fclose(SimParams_file);
    return;
}

void FluidSystem::WriteDemoSimParams ( const char * relativePath, int gpu_mode, int cpu_mode, uint num_particles, float spacing, float x_dim, float y_dim, float z_dim, uint demoType, uint simSpace, uint debug){
    m_FParams.debug=debug;
    
    std::cout<<"\nWriteDemoSimParams chk1, num_particles="<<num_particles<<", m_FParams.debug="<<m_FParams.debug<<std::flush;
    
    m_Param[PEXAMPLE] = simSpace;          // simSpace==2 : wave pool example.
    m_Param[PGRID_DENSITY] = 2.0;
    m_Param[PNUM] = num_particles;  // 1000000;    //1000 = minimal simulation, 1000000 = large simulation
    AllocateBuffer ( FPARAMS, sizeof(FParams), 1,0, GPU_OFF, CPU_YES ); 
    m_Time = 0;
    mNumPoints = 0;			        // reset count
    SetupDefaultParams();           // set up the standard demo
    SetupExampleParams(spacing);
    SetupExampleGenome();
    SetupSimulation(gpu_mode, cpu_mode);
    /*
    mMaxPoints = m_Param[PNUM];    
    m_Param[PGRIDSIZE] = 2*m_Param[PSMOOTHRADIUS] / m_Param[PGRID_DENSITY];
    SetupSPH_Kernels();
    SetupSpacing();
    SetupGrid ( m_Vec[PVOLMIN], m_Vec[PVOLMAX], m_Param[PSIMSCALE], m_Param[PGRIDSIZE], 1.0f );  if (m_FParams.debug>1) std::cout << " chk1.0 " << std::flush ;
    AllocateParticles ( mMaxPoints, GPU_OFF, CPU_YES );                                          if (m_FParams.debug>1) std::cout << " chk1.1 " << std::flush ;
    AllocateGrid(GPU_OFF, CPU_YES);                                                              if (m_FParams.debug>1) std::cout << " chk1.2 " << std::flush ;
    std::cout<<"\nWriteDemoSimParams chk2 "<<std::flush;
    */
    m_Vec[PBOUNDMIN].x= m_Vec[PVOLMIN].x + 2*(m_Param[PGRIDSIZE]/m_Param[PSIMSCALE]);
    m_Vec[PBOUNDMIN].y= m_Vec[PVOLMIN].y + 2*(m_Param[PGRIDSIZE]/m_Param[PSIMSCALE]);
    m_Vec[PBOUNDMIN].z= m_Vec[PVOLMIN].z + 2*(m_Param[PGRIDSIZE]/m_Param[PSIMSCALE]);
    m_Vec[PINITMIN].x = std::max(m_Vec[PINITMIN].x , m_Vec[PBOUNDMIN].x+1 );
    m_Vec[PINITMIN].y = std::max(m_Vec[PINITMIN].y , m_Vec[PBOUNDMIN].y+1 );
    m_Vec[PINITMIN].z = std::max(m_Vec[PINITMIN].z , m_Vec[PBOUNDMIN].z+1 );
    
    Vector3DF pinit_max = {x_dim,y_dim,z_dim};
    pinit_max += m_Vec[PINITMIN];
    
    m_Vec[PBOUNDMAX].x= m_Vec[PVOLMAX].x - 2*(m_Param[PGRIDSIZE]/m_Param[PSIMSCALE]);
    m_Vec[PBOUNDMAX].y= m_Vec[PVOLMAX].y - 2*(m_Param[PGRIDSIZE]/m_Param[PSIMSCALE]);
    m_Vec[PBOUNDMAX].z= m_Vec[PVOLMAX].z - 2*(m_Param[PGRIDSIZE]/m_Param[PSIMSCALE]);
    pinit_max.x = std::min(pinit_max.x , m_Vec[PBOUNDMAX].x-1 );
    pinit_max.y = std::min(pinit_max.y , m_Vec[PBOUNDMAX].y-1 );
    pinit_max.z = std::min(pinit_max.z , m_Vec[PBOUNDMAX].z-1 );
    
    if (m_FParams.debug>1) {
    cout<<"\n\nPGRIDSIZE=("<<m_Vec[PGRIDSIZE].x<<","<<m_Vec[PGRIDSIZE].y<<","<<m_Vec[PGRIDSIZE].z
        <<"),  PSIMSCALE=("<<m_Vec[PSIMSCALE].x<<","<<m_Vec[PSIMSCALE].y<<","<<m_Vec[PSIMSCALE].z
        <<"),  m_Param[PGRIDSIZE]="<<m_Param[PGRIDSIZE]<<",  m_Param[PSIMSCALE]="<<m_Param[PSIMSCALE]
        <<"\n"<<std::flush;
    
    cout<<"\n\nPBOUNDMIN=("<<m_Vec[PBOUNDMIN].x <<","<<m_Vec[PBOUNDMIN].y <<","<<m_Vec[PBOUNDMIN].z 
        <<"), PBOUNDMAX=("<<m_Vec[PBOUNDMAX].x <<","<<m_Vec[PBOUNDMAX].y <<","<< m_Vec[PBOUNDMAX].z
        <<"),  PVOLMIN=("<<m_Vec[PVOLMIN].x <<","<<m_Vec[PVOLMIN].y <<","<<m_Vec[PVOLMIN].z 
        <<"),   PVOLMAX=("<<m_Vec[PVOLMAX].x <<","<<m_Vec[PVOLMAX].y <<","<<m_Vec[PVOLMAX].z 
        <<")\n"<<std::flush;
    
    cout<<"\n\n### WriteDemoSimParams: SetupAddVolumeMorphogenesis2"
        <<"(  min=("<<m_Vec[PINITMIN].x<<","<<m_Vec[PINITMIN].y<<","<<m_Vec[PINITMIN].z
        <<"), max=("<<pinit_max.x<<","<<pinit_max.y<<","<<pinit_max.z
        <<"), spacing="<<spacing<<", 0.1f, demoType="<<demoType
        <<")\n"<<std::flush;
    }
    std::cout<<"\nWriteDemoSimParams chk3 "<<std::flush;
    
    SetupAddVolumeMorphogenesis2(m_Vec[PINITMIN], pinit_max, spacing, 0.1f, demoType);
    std::cout<<"\nWriteDemoSimParams chk4, relativePath="<<relativePath<<"\n "<<std::flush;
    
    WriteSimParams ( relativePath );    if (m_FParams.debug>1) std::cout << "\n WriteSimParams ( relativePath );  completed \n" << std::flush ;  // write data to file
    std::cout<<"\nWriteDemoSimParams chk5 "<<std::flush;
    
    WriteGenome ( relativePath);        if (m_FParams.debug>1) std::cout << "\n WriteGenome ( relativePath );  completed \n" << std::flush ;
    std::cout<<"\nWriteDemoSimParams chk6 "<<std::flush;
    
    SavePointsCSV2 ( relativePath, 1 ); if (m_FParams.debug>1) std::cout << "\n SavePointsCSV ( relativePath, 1 );  completed \n" << std::flush ;
    std::cout<<"\nWriteDemoSimParams chk7 "<<std::flush;
    
}

void FluidSystem::ReadSpecificationFile ( const char * relativePath ){
    char SimParams_file_path[256];
    sprintf ( SimParams_file_path, "%s/SpecificationFile.txt", relativePath );
    
    FILE * SpecFile = fopen(SimParams_file_path, "rb");
    if(SpecFile == NULL)    { std::cout<<"\nCould not read file: "<<relativePath<<".\n"<<std::flush; }
    
    // find number of lines
    int ch, number_of_lines = 0;
    while ( EOF != ( ch=getc ( SpecFile ) ) )   if ( '\n' == ch )  ++number_of_lines;
    std::cout << "\nSpecFile = " << SimParams_file_path << std::flush;
    std::cout << "\nNumber of lines in SpecFile = " << number_of_lines << std::flush;

    // read file
    std::fseek(SpecFile, 0, SEEK_SET);
    int ret =0;
    
    ret += std::fscanf ( SpecFile, "num_particles = %u\n ", &launchParams.num_particles );
    ret += std::fscanf ( SpecFile, "demoType = %u\n ", &launchParams.demoType );
    ret += std::fscanf ( SpecFile, "simSpace = %u\n ", &launchParams.simSpace );
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "m_Time = %f\n ", &launchParams.m_Time );
    ret += std::fscanf ( SpecFile, "m_DT = %f\n ", &launchParams.m_DT );
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "gridsize = %f\n ", &launchParams.gridsize);
    ret += std::fscanf ( SpecFile, "spacing = %f\n ", &launchParams.spacing);
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "simscale = %f\n ", &launchParams.simscale);
    ret += std::fscanf ( SpecFile, "smooth_radius = %f\n ", &launchParams.smoothradius);
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "visc = %f\n ", &launchParams.visc);
    ret += std::fscanf ( SpecFile, "surface_t = %f\n ", &launchParams.surface_tension);
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "mass = %f\n ", &launchParams.mass);
    ret += std::fscanf ( SpecFile, "radius = %f\n ", &launchParams.radius);
    /*ret += std::fscanf ( SpecFile, "dist = %f\n ", &launchParams.dist);*/
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "int_stiff = %f\n ", &launchParams.intstiff);
    ret += std::fscanf ( SpecFile, "ext_stiff = %f\n ", &launchParams.extstiff);
    ret += std::fscanf ( SpecFile, "ext_damp = %f\n ", &launchParams.extdamp);
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "accel_limit = %f\n ", &launchParams.accel_limit);
    ret += std::fscanf ( SpecFile, "vel_limit = %f\n ", &launchParams.vel_limit);
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "grav = %f\n ", &launchParams.grav);
    ret += std::fscanf ( SpecFile, "slope = %f\n ", &launchParams.ground_slope);
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "force_min = %f\n ", &launchParams.force_min);
    ret += std::fscanf ( SpecFile, "force_max = %f\n ", &launchParams.force_max);
    ret += std::fscanf ( SpecFile, "force_freq = %f\n ", &launchParams.force_freq);
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "x_dim = %f\n ", &launchParams.x_dim );
    ret += std::fscanf ( SpecFile, "y_dim = %f\n ", &launchParams.y_dim );
    ret += std::fscanf ( SpecFile, "z_dim = %f\n ", &launchParams.z_dim );
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "pos_x = %f\n ", &launchParams.pos_x );
    ret += std::fscanf ( SpecFile, "pos_y = %f\n ", &launchParams.pos_y );
    ret += std::fscanf ( SpecFile, "pos_z = %f\n ", &launchParams.pos_z );
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "volmin_x = %f\n ", &launchParams.volmin.x );
    ret += std::fscanf ( SpecFile, "volmin_y = %f\n ", &launchParams.volmin.y );
    ret += std::fscanf ( SpecFile, "volmin_z = %f\n ", &launchParams.volmin.z );
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "volmax_x = %f\n ", &launchParams.volmax.x );
    ret += std::fscanf ( SpecFile, "volmax_y = %f\n ", &launchParams.volmax.y );
    ret += std::fscanf ( SpecFile, "volmax_z = %f\n ", &launchParams.volmax.z );
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "initmin_x = %f\n ", &launchParams.initmin.x );
    ret += std::fscanf ( SpecFile, "initmin_y = %f\n ", &launchParams.initmin.y );
    ret += std::fscanf ( SpecFile, "initmin_z = %f\n ", &launchParams.initmin.z );
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "initmax_x = %f\n ", &launchParams.initmax.x );
    ret += std::fscanf ( SpecFile, "initmax_y = %f\n ", &launchParams.initmax.y );
    ret += std::fscanf ( SpecFile, "initmax_z = %f\n ", &launchParams.initmax.z );
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "paramsPath = %s\n ", launchParams.paramsPath );
    ret += std::fscanf ( SpecFile, "pointsPath = %s\n ", launchParams.pointsPath );
    ret += std::fscanf ( SpecFile, "genomePath = %s\n ", launchParams.genomePath );
    ret += std::fscanf ( SpecFile, "outPath = %s\n ", launchParams.outPath );
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "num_files = %u\n ", &launchParams.num_files );
    ret += std::fscanf ( SpecFile, "steps_per_InnerPhysicalLoop = %u\n ", &launchParams.steps_per_InnerPhysicalLoop );
    ret += std::fscanf ( SpecFile, "steps_per_file = %u\n ", &launchParams.steps_per_file );
    ret += std::fscanf ( SpecFile, "freeze_steps = %u\n ", &launchParams.freeze_steps );
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "debug = %u\n ", &launchParams.debug );
    ret += std::fscanf ( SpecFile, "file_num = %u\n ", &launchParams.file_num );
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "save_ply = %c\n ", &launchParams.save_ply );
    ret += std::fscanf ( SpecFile, "save_csv = %c\n ", &launchParams.save_csv );
    ret += std::fscanf ( SpecFile, "save_vtp = %c\n ", &launchParams.save_vtp );
    ret += std::fscanf ( SpecFile, "\n");
    
    ret += std::fscanf ( SpecFile, "gene_activity = %c\n ", &launchParams.gene_activity );
    ret += std::fscanf ( SpecFile, "remodelling = %c\n ", &launchParams.remodelling );
    ret += std::fscanf ( SpecFile, "\n");
 
    m_FParams.debug = launchParams.debug ;
    std::cout<<"\n\n   launchParams.debug="<<launchParams.debug<<",  m_FParams.debug="<<m_FParams.debug <<" .\t"<<std::flush;
    if (m_FParams.debug>1) std::cout << "\nvoid FluidSystem::ReadSpecificationFile(..),  ret = " << ret << std::flush;
    fclose ( SpecFile );
    return;
}

void FluidSystem::WriteExampleSpecificationFile ( const char * relativePath ){ // writes a default version 
    char SimParams_file_path[256];
    sprintf ( SimParams_file_path, "%s/SpecificationFile.txt", relativePath );
    //const char * SimParams_file_path = relativePath;
    FILE * SpecFile = fopen ( SimParams_file_path, "w" );
    int ret =0;
    
    
    ret += std::fprintf ( SpecFile, "num_particles = %u\n ", launchParams.num_particles );
    ret += std::fprintf ( SpecFile, "demoType = %u\n ", launchParams.demoType );
    ret += std::fprintf ( SpecFile, "simSpace = %u\n ", launchParams.simSpace );
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "m_Time = %f\n ", m_Time );
    ret += std::fprintf ( SpecFile, "m_DT = %f\n ", m_DT );
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "gridsize = %f\n ", m_Param [ PGRIDSIZE ]);
    ret += std::fprintf ( SpecFile, "spacing = %f\n ", m_Param [ PSPACING ]);
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "simscale = %f\n ", m_Param [ PSIMSCALE ]);
    ret += std::fprintf ( SpecFile, "smooth_radius = %f\n ", m_Param [ PSMOOTHRADIUS ]);
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "visc = %f\n ", m_Param [ PVISC ]);
    ret += std::fprintf ( SpecFile, "surface_t = %f\n ", m_Param [ PSURFACE_TENSION ]);
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "mass = %f\n ", m_Param [ PMASS ]);
    ret += std::fprintf ( SpecFile, "radius = %f\n ", m_Param [ PRADIUS ]);
    /*ret += std::fprintf ( SpecFile, "dist = %f\n ", m_Param [ PDIST ]);*/
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "int_stiff = %f\n ", m_Param [ PINTSTIFF ]);
    ret += std::fprintf ( SpecFile, "ext_stiff = %f\n ", m_Param [ PEXTSTIFF ]);
    ret += std::fprintf ( SpecFile, "ext_damp = %f\n ", m_Param [ PEXTDAMP ]);
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "accel_limit = %f\n ", m_Param [ PACCEL_LIMIT ]);
    ret += std::fprintf ( SpecFile, "vel_limit = %f\n ", m_Param [ PVEL_LIMIT ]);
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "grav = %f\n ", m_Param [ PGRAV ]);
    ret += std::fprintf ( SpecFile, "slope = %f\n ", m_Param [ PGROUND_SLOPE ]);
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "force_min = %f\n ", m_Param [ PFORCE_MIN ]);
    ret += std::fprintf ( SpecFile, "force_max = %f\n ", m_Param [ PFORCE_MAX ]);
    ret += std::fprintf ( SpecFile, "force_freq = %f\n ", m_Param [ PFORCE_FREQ ]);
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "x_dim = %f\n ", launchParams.x_dim );
    ret += std::fprintf ( SpecFile, "y_dim = %f\n ", launchParams.y_dim );
    ret += std::fprintf ( SpecFile, "z_dim = %f\n ", launchParams.z_dim );
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "pos_x = %f\n ", launchParams.pos_x );
    ret += std::fprintf ( SpecFile, "pos_y = %f\n ", launchParams.pos_y );
    ret += std::fprintf ( SpecFile, "pos_z = %f\n ", launchParams.pos_z );
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "volmin_x = %f\n ", m_Vec [ PVOLMIN ].x);
    ret += std::fprintf ( SpecFile, "volmin_y = %f\n ", m_Vec [ PVOLMIN ].y );
    ret += std::fprintf ( SpecFile, "volmin_z = %f\n ", m_Vec [ PVOLMIN ].z );
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "volmax_x = %f\n ", m_Vec [ PVOLMAX ].x );
    ret += std::fprintf ( SpecFile, "volmax_y = %f\n ", m_Vec [ PVOLMAX ].y );
    ret += std::fprintf ( SpecFile, "volmax_z = %f\n ", m_Vec [ PVOLMAX ].z );
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "initmin_x = %f\n ", m_Vec [ PINITMIN ].x );
    ret += std::fprintf ( SpecFile, "initmin_y = %f\n ", m_Vec [ PINITMIN ].y );
    ret += std::fprintf ( SpecFile, "initmin_z = %f\n ", m_Vec [ PINITMIN ].z );
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "initmax_x = %f\n ", m_Vec [ PINITMAX ].x );
    ret += std::fprintf ( SpecFile, "initmax_y = %f\n ", m_Vec [ PINITMAX ].y );
    ret += std::fprintf ( SpecFile, "initmax_z = %f\n ", m_Vec [ PINITMAX ].z );
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "paramsPath = %s\n ", launchParams.paramsPath );
    ret += std::fprintf ( SpecFile, "pointsPath = %s\n ", launchParams.pointsPath );
    ret += std::fprintf ( SpecFile, "genomePath = %s\n ", launchParams.genomePath );
    ret += std::fprintf ( SpecFile, "outPath = %s\n ", launchParams.outPath );
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "num_files = %u\n ", launchParams.num_files );
    ret += std::fprintf ( SpecFile, "steps_per_InnerPhysicalLoop = %u\n ", launchParams.steps_per_InnerPhysicalLoop );
    ret += std::fprintf ( SpecFile, "steps_per_file = %u\n ", launchParams.steps_per_file );
    ret += std::fprintf ( SpecFile, "freeze_steps = %u\n ", launchParams.freeze_steps );
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "debug = %u\n ", launchParams.debug );
    ret += std::fprintf ( SpecFile, "file_num = %u\n ", launchParams.file_num );
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "save_ply = %c\n ", launchParams.save_ply );
    ret += std::fprintf ( SpecFile, "save_csv = %c\n ", launchParams.save_csv );
    ret += std::fprintf ( SpecFile, "save_vtp = %c\n ", launchParams.save_vtp );
    ret += std::fprintf ( SpecFile, "\n");
    
    ret += std::fprintf ( SpecFile, "gene_activity = %c\n ", launchParams.gene_activity );
    ret += std::fprintf ( SpecFile, "remodelling = %c\n ", launchParams.remodelling );
    ret += std::fprintf ( SpecFile, "\n");
    
    fclose ( SpecFile );
    return;
}

void FluidSystem::SaveUintArray( uint* array, int numElem1, const char * relativePath ){ /// Used to save an array to .csv for debugging.
   FILE* fp = fopen ( relativePath, "w" );
    if (fp == NULL) {
        if (m_FParams.debug>1) std::cout << "\nvoid FluidSystem::SaveUintArray ( const char * relativePath, int frame )  Could not open file "<< fp <<"\n"<< std::flush;
        assert(0);
    } 
    for(uint i=0; i<numElem1; i+=100){ 
        fprintf(fp, "\n"); 
        for(uint j=0;j<100 && i+j<numElem1;j++) fprintf(fp, ",%u,%u,",i+j,array[i+j]); 
    }
    fclose ( fp );
    fflush ( fp );
    std::string s;
    std::stringstream ss;
    ss << relativePath;
    ss >> s;  
    if (m_FParams.debug>1) cout<<"\nSaved UintArray file: "<< s <<"\n"<<std::flush;
}

void FluidSystem::SaveUintArray_2Columns( uint* array, int numElem1, int buff_len, const char * relativePath ){ /// Used to save DESNSE_LIST_CHANGES (particle,bondIdx) arrays to .csv for debugging.
   FILE* fp = fopen ( relativePath, "w" );
    if (fp == NULL) {
        if (m_FParams.debug>1) std::cout << "\nvoid FluidSystem::SaveUintArray_2Collumns ( const char * relativePath, int frame )  Could not open file "<< fp <<"\n"<< std::flush;
        assert(0);
    } 
    for(uint i=0; i<numElem1; i+=100){ 
        fprintf(fp, "\n"); 
        for(uint j=0;j<100 && i+j<numElem1;j++) fprintf(fp, ",%u,%u,%u,",i+j,array[i+j],array[i+j+buff_len]); // i.e. listIdx, ParticleIdx, BondIdx.
    }
    fclose ( fp );
    fflush ( fp );
    std::string s;
    std::stringstream ss;
    ss << relativePath;
    ss >> s;  
    if (m_FParams.debug>1) cout<<"\nSaved UintArray_2Columns file: "<< s <<"\n"<<std::flush;
}

void FluidSystem::SaveUintArray_2D( uint* array, int numElem1, int numElem2, const char * relativePath ){ /// Used to save DESNSE_LIST_CHANGES (particle,bondIdx) arrays to .csv for debugging.
   FILE* fp = fopen ( relativePath, "w" );
    if (fp == NULL) {
        if (m_FParams.debug>1) std::cout << "\nvoid FluidSystem::SaveUintArray_2D ( const char * relativePath, int frame )  Could not open file "<< fp <<"\n"<< std::flush;
        assert(0);
    } 
    for(uint i=0; i<numElem1; i++){ 
        fprintf(fp, "\n"); 
        for(uint j=0; j<numElem2; j++) fprintf(fp, ",%u,",array[j*numElem1+i]); // i.e. listIdx, ParticleIdx, BondIdx.
    }
    fclose ( fp );
    fflush ( fp );
    std::string s;
    std::stringstream ss;
    ss << relativePath;
    ss >> s;  
    if (m_FParams.debug>1) cout<<"\nSaved UintArray_2D file: "<< s <<"\n"<<std::flush;
}

