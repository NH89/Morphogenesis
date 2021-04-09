
#### Memory usage

    Factors affecting memory requrement for Morphogenesis:
    
* 0/ Available GPU memory depends on the device + other sofftware using the GPU, especially yuor desktop, GUIs, graphics and video.
To see your current GPU memory useage and total GPU memory, call "nvidia-smi" from the commandline.

NB When running a GUI onthe GPU it is advised to use models with fewer than 100k bins and particles.
    
* 1/ The number of bins in the simulation volume. i.e. volume/particle_radius^3
Dense arrays of bins are maintained for sorting particles. genes, remodelling.
These arrays are use to profuce the dense lists of particles for each kernel, for efficient computation.

* 2/ The number of particles in the simulation

* 3/ The number of genes in the simulation will increase the size of the bin arrays(1), and the data per particle(2).

* 4/ The number of active genes among the particles will increase the size of the dense lists of particles for each active gene (1).

* 5/ The number of particles subject to remodelling per time step, will incease the size of the dense lists for each remodelling kernel.

NB the simulation volume and particle radius are set in SimParams.txt by :
        m_Param[PSIMSCALE], m_Param[PGRID_DENSITY], m_Param[PSMOOTHRADIUS],  
        m_Vec[PVOLMIN], m_Vec[PVOLMAX].
        
Where: 
        m_GridRes = (PVOLMAX-PVOLMIN)*PGRID_DENSITY*PSIMSCALE / (2*PSMOOTHRADIUS)
        
        Total number of bins = m_GridRes.x * m_GridRes.y * m_GridRes.z
        
For default PGRID_DENSITY=2, PSIMSCALE=0.005, PSMOOTHRADIUS=0.015,
This gives  m_GridRes = (PVOLMAX-PVOLMIN)/3
So 
        10x10x10 vol -> 333 bins
        30x30x30 vol -> 9,000 bins
        60x60x60 vol -> 72,000 bins
        100x100x100 vol -> 333,3333 bins
        150x150x150 vol -> 1,125,000 bins
        1000x1000x1000 vol -> 333,333,333 bins. Too big for most GPUs, would need multi-GPU on cluster.
    
    
#### Processing speed

Factors affecting prcoessing speed include:
    
* 0/ Available GPU blocks and cores per block, after space taken by other GPU contxts for other programs.
    
* 1/ The number of bins. These have to be processesd every time step to sort the particle data and generate the dense lists.
    
* 2/ The number of particles. 
    NB there is the possiblity of "solid" and "living" particles into separate lists (using genes), to avoid running elastic and gene kernels on fluid and non-living tissue.
    
* 3/ The length of the "active gene" and "remodelling" particle lists per time step.
    NB the system of dense lists, provides faster running for models with few active genes per particle, and few particles remodelling per time step.

CPU-only test program to generate example **"SimParams.txt"** and **"particles_pos_vel_color100001.csv"** files for specifying models, and a **"particles_pos100001.ply"** for viewing a model in e.g. Meshlab.


### Profiling the code

If you have Cuda-toolkit installed, enter "nsight-sys" at the commandline to open "Nvidia Nsight Systems".
    
On Linux, first set the 'paranoid level'

    sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'

Call the Nsight Systems GUI

    nsight-sys
    
See 
    https://developer.nvidia.com/nsight-systems , https://docs.nvidia.com/nsight-systems/index.html
    
Follow the nsight-systems InstallationGuide.
NB especially wrt setting /proc/sys/kernel/perf_event_paranoid    

    https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#linux-requirements
    
For a an introduction to how to use Nsight Systems, see "Blue Waters Webinar: Introduction to NVIDIA Nsight Systems"   https://www.youtube.com/watch?v=WA8C48FJi3c 
    
Note, setting the "debug" value (0-5) when launching "load_sim" will have a big effect on the memory transfers, flies saved and console output, and therefore the performance.
