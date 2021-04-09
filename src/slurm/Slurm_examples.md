

### Accessing GPUs and Xeon Phi cards on bracewell using the SLURM job manager
Accessing GPUs
Identifying which GPUs your job was allocated
Should I request an entire node and/or all the GPUs?
Setting the compute access mode
Accessing Xeon Phi cards
Identifying which Xeon Phi cards your job was allocated
How can I make use of Phi cards once I have been allocated them?

The purpose of this page is to explain the process of requesting accelerator resources on the bracewell cluster. If you are new to using a Linux cluster, you should read the quick start guide first. In particular, it is essential to know how to run batch jobs on a cluster.
Accessing GPUs and Xeon Phi cards on bracewell using the SLURM job manager
Each Bracewell node possesses one of two distinct types of accelerator resources – either Nvidia GPUs, or Intel Xeon Phi cards. 128 nodes are classified as GPU nodes; each of these holds three Nvidia GPUs. The remaining 16 nodes are Xeon Phi nodes, each containing two SE7120P Xeon Phi cards.

Due to their nature, accelerators are not available in isolation; to access an accelerator, at least one core must be requested on the node that it belongs to. SLURM uses a 'generic resource' module that allows specification of a --gres=gpu:X or --gres=mic:Y type syntax for requesting nodes that have accelerators.

#### Accessing GPUs
The nodes containing GPUs are named g001, g002, and so on. The nodes are accessible to jobs that require only CPUs, and by jobs that require GPUs. A possible SLURM request for allocation of some number of GPUs   (per node) and cores (per node) on some number of nodes is as follows:

    #SBATCH --nodes=i
    #SBATCH --tasks-per-node=j  (or --ntasks-per-node=j)
    #SBATCH --cpus-per-task=k
    #SBATCH --gres=gpu:m
This syntax can be understood as saying "I need access to i nodes. On each of those i nodes, I will run j tasks and each of those tasks to be given access to k CPU cores (with a maximum of 16 total per node i.e j x k <= 16) and there will be m GPUs (with a maximum of 3) per node."

Note that --cpus-per-task will give a task the exclusive use of the k allocated CPUs, however the  --gres=gpu:m request will give a task non-exclusive access to all m GPUs requested. Importantly, it should be noted that by default, GPUs have have been set so that they can be accessed by a single process (task) only, so code/workflow logic must be present so that a particular GPU is only used by a particular task. This 'exclusive access' setup also alows a single task to access multiple GPUs (just not vice-versa).

So, for example to request a single GPU and a single CPU on a single node:

    #SBATCH --nodes=1
    #SBATCH --tasks-per-node=1
    #SBATCH --cpus-per-task=1
    #SBATCH --gres=gpu:1
The above request could omit the --nodes= paramater and it would imply that the number of nodes requested is 1.

Other examples include:

    # 1 node, 8 cores and access to 3 GPUs.
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=8
    #SBATCH --gres=gpu:3
 
    # 3 nodes, 12 cores in total and 1 GPU per node.
    #SBATCH --nodes=3
    #SBATCH --ntasks-per-node=4
    #SBATCH --gres=gpu:1
 
    # 5 nodes, 60 cores in total and 3 GPUs per node.
    #SBATCH --nodes=5
    #SBATCH --ntasks-per-node=12
    #SBATCH --gres=gpu:3
 
Note that if you request more than one node, access to those GPUs is possible through the use of MPI processes or the srun command. These are advanced techniques – please ensure you understand them before you attempt to use them.

Specifying the  --tasks-per-node=k request will set an environment variable named  SLURM_NTASKS_PER_NODE  in each nodes environment and this can be used  within a program to delegate available GPUs to tasks. Please note that a variable named SLURM_TASKS_PER_NODE can also be made available, however it is not readily useable for task delegation due to its irregular formatting.

If you need to use one or more GPUs on a node, make sure that you explicitly request them.

#### Identifying which GPUs your job was allocated
The SLURM scheduler makes available a list of GPU resources that have been allocated to a job by setting the  CUDA_VISIBLE_DEVICES  variable to a zero based, comma separated list of GPUs allocated. The value of the GPU allocated is not important as CUDA and OpenCL programs using the allocated device will have any avilable devices numbered from zero.

    # Inside a sbatch job created with the resource request:
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --gres=gpu:2
 
    # You will have access to the allocated requested resources
    # Run a job to look for the environment on the allocated resources using srun:
 
>srun env | grep CUDA_VISIBLE_DEVICES
 
could return any of the following:
CUDA_VISIBLE_DEVICES=0,1
CUDA_VISIBLE_DEVICES=0,2
CUDA_VISIBLE_DEVICES=1,2
If your code is not requesting all the GPUs available on a node, your job script could use some mechanism of parsing the value of the CUDA_VISIBLE_DEVICES variable and  count the number of values present in this  variable

For multi-node jobs, such as those using MPI, this approach is insufficient. Please contact the SC helpdesk for assistance in this case.

#### Should I request an entire node and/or all the GPUs?
It is quite common for GPU-enabled codes to not make use of multiple CPU cores, but to make use of multiple GPUs. The obvious resource request for a job that uses only one core but three GPUs is:

    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --gres=gpu:3
This is good from the perspective of the batch system, as the remaining fifteen CPU cores on that node can be allocated to other jobs in the queue that do not need access to GPUs. However, it is possible that other jobs running on the same node will contend with yours for access to the filesystems and to memory, which may degrade performance. In particular, if you wish to benchmark the performance of a code, you should request all the CPU cores on a node (even if you don't need them). You should only do this if you have a good reason.

Also, consider the possibility that you only need a single GPU:

    #SBATCH --gres=gpu:1
While such a job is running, it is possible that other GPU-using jobs will be concurrently executed on the same node and there can be contention on the shared PCIe bus. If this will adversely affect performance you may want to request all GPUs to prevent it. Again, you should only do this if you know for sure that you need to (as it limits access to others).

#### Setting the compute access mode
We are currently requesting advice from our SLURM support team on how users may change the GPU access mode. The default compute mode has been set for each device as exclusive process.

The nVidia GPUs support the following modes:

0 – Default - Shared mode available for multiple processes
1 – Exclusive Thread - Only one host thread is allowed to access the GPU for compute
2 – Prohibited - No host threads / processes are allowed to access the GPU for compute
3 – Exclusive Process - Only one host process is allowed to access the GPU for compute

###########################################################################################################

### Sample Slurm job scripts

This page contains a number of sample job scripts that outline how to request resources for different type of workloads.

A single core job
A 10-core MPI job on one compute node
A 40-core MPI job on two compute nodes
A 16-core multi-threading job on one node
A MPI/OpenMP hybrid application 20 threads per node and 12 nodes
A 1-core, 1-GPU job
A 12-core, 3-GPU MPI job

#### A single core job
This job, named "HelloWorld", requests one hour of wall time, one core and 500 MB of memory.

    #!/bin/bash
 
    #SBATCH --job-name=HelloWorld
    #SBATCH --time=01:00:00
    #SBATCH --ntasks=1
    #SBATCH --mem=500m

    # Application specific commands:
    ./helloworld

#### A 10-core MPI job on one compute node
This job, named "MPI-test", requests 24 hours of wall time, 10 cores on one single compute node and 2 GB of memory.

    #!/bin/bash
    #SBATCH --job-name=MPI-test
    #SBATCH --time=24:00:00
    #SBATCH --ntasks-per-node=10
    #SBATCH --nodes=1
    #SBATCH --mem=2g

    # Application specific commands:
    module load openmpi
    cd project
    mpirun -np 10 ./MPI-test

#### A 40-core MPI job on two compute nodes
This job, named "Large-MPI-test", requests 2 days of wall time, 40 cores in total on two compute nodes and 100 GB of memory per node.

    #!/bin/bash
    #SBATCH --job-name=Large-MPI-test
    #SBATCH --time=48:00:00
    #SBATCH --ntasks-per-node=20
    #SBATCH --nodes=2
    #SBATCH --mem=100g

    # Application specific commands:
    module load openmpi
    cd project
    mpirun -np 40 ./MPI-test2

#### A 16-core multi-threading job on one node
This job, named "ThreadedJob", requests 6 hours of wall time, 8 GB of memory and one single process that needs 16 cores to handle.

    #!/bin/bash
    #SBATCH --job-name=ThreadedJob
    #SBATCH --time=6:00:00
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=16
    #SBATCH --nodes=1
    #SBATCH --mem=8g

    # Application specific commands:
    cd project
    ./thread-test

#### A MPI/OpenMP hybrid application 20 threads per node and 12 nodes
This job, named "hybrid", requests 72 hours of wall time, 125 GB of memory per node and 240 cores in total. In particular, the resource request also indicates that there will be one MPI process on each node and each MPI process will run 20 threads.

    #!/bin/bash
    #SBATCH --job-name=hybrid
    #SBATCH --time=72:00:00
    #SBATCH --nodes=12
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=20
    #SBATCH --mem=125g

    # Application specific commands:
    cd project
    mpirun -np 12 ./run-hybrid-job

#### A 1-core, 1-GPU job
This job, named "GPUtest", requests 10 hours of wall time, 4 GB of memory, one single core and one GPU.

    #!/bin/bash
    #SBATCH --job-name=GPUtest
    #SBATCH --time=10:00:00
    #SBATCH --ntasks-per-node=1
    #SBATCH --gres=gpu:1
    #SBATCH --mem=4g

    # Application specific commands:
    module load cuda
    cd project
    ./gputest

#### A 12-core, 3-GPU MPI job
This job, named "gpu-mpi-test", requests 12 hours of wall time, 12 GB of memory, 12 cores and 3 GPUs.

    #!/bin/bash
    #SBATCH --job-name=gpu-mpi-test
    #SBATCH --time=12:00:00
    #SBATCH --ntasks-per-node=12
    #SBATCH --gres=gpu:3
    #SBATCH --mem=12g

    # Application specific commands:
    module load cuda openmpi
    cd project
    mpirun -np 12 ./gpu-mpi-test




###################################################

### Running jobs in parallel:

* Running jobs in parallel is what a batch system does.
* What if I have a lot of jobs?
* Submit jobs using a loop.
* Submit parameterized jobs using a loop and a single job script.
* Submit array jobs with a single job script.
* Generating a set of job scripts.


#### Running jobs in parallel is what a batch system does.
You can prepare a small number of scripts by hand (one for each job) and submit them one-by-one to the batch system. eg.:

    sbatch Rjob.sh; sbatch MatlabJ.sh; sbatch AnotherRjob.sh
    
These independent jobs will be run when the resources become available (possibly/probably in parallel).

#### What if I have a lot of jobs?
If there are a lots of scripts/jobs with a pattern, it will save a lot of effort and be less error-prone if you automate creating the files and submitting the jobs. This is particularly suited to 'ensemble' jobs or 'parameter sweeps' where many very similar jobs are run to generate results over a defined parameter space.

This article concentrates on submitting jobs and on writing jobscripts so one script can be submitted many times. You can use a scripting framework of your choice to automate creating script files if you want to generate and keep separate scripts. Managing a lot of scripts in separate files can be awkward and error prone (though it has some benefits). The basic ideas are common. Here are a few examples:

#### Submit jobs using a loop.
If you have a script for each job to run, you can submit jobs in a loop. This example uses a bash shell loop and matches a particular pattern for script names:

    for SCRIPT in *.q; do sbatch $SCRIPT; done
    
where *.q refers to all the files in the current working directory with a .q extension.

#### Submit parameterized jobs using a loop and a single job script.
This is another bash loop, which sets a variable that is passed into the job using the -v argument to sbatch. A single job script is used for all of the jobs. You would use the variable within the script to make each job work on a different parameter:

for X in $(seq 2.1 0.3 3.4); do sbatch --export X=$X myjob.q; done
The job script might contain:

    #SBATCH --output <path>
    cd $SLURM_SUBMIT_DIR
    ./my_exe $X > $X.out
    
You can further extend the example to have nested loops:

    #Slurm
    for X in $(seq 100); do
        for Y in $(seq 100); do
            sbatch --export X=$X,Y=$Y myXYjob.q
        done
    done
 
You can loop over things that are not numbers - say files instead:

    #Slurm
    for FILE in *.in; do sbatch --export IN=$FILE myjob.q; done

Or lines in a file - submit myjob.q with LINE set to each line in paramset.in:

    #Slurm
    for L in $(cat paramset.in); do sbatch --export LINE="$L" myjob.q; done

This last case is pretty flexible as you can pull apart the file line however you like in the script, but you might be better off passing a line number (say as the index of an array job) and grabbing the line from the file within your script (maybe with awk). That way if the job fails for a given line it is easy to submit a replacement job.

#### Submit array jobs with a single job script.
Array jobs use the sbatch -a option. For example, submit myjob.q. as an array job.

    sbatch -a 1-20 myjob.q

Array jobs offer some advantages in terms of managing the jobs as a group and reduced load on the server when you submit many jobs (different load anyway). Array jobs will submit more quickly as separate connections to the server for each job are not needed.

Within each job the array index will be available in the variable SLURM_ARRAY_TASK_ID to make decisions about what work to do.

#### Generating a set of job scripts.
You may prefer to have a set of separate scripts to track which ones have been run successfully. In this case you would loop over your parameters and write them into a set of scripts based on a template. There are a large variety of ways of doing this so only a simple example will be given. Care must be taken to make sure variables get expanded/interpreted in the right context. It might be troublesome to try and use a shell script loop to write shell scripts. Python or perl might be better scripting languages for writing the scripts in this case but in a simple case bash is OK.

The following example reads lines from a file and puts each of them into a separate batch script, named after the line number with padding (seq001.q, seq002.q, ...):

    #Slurm
    nlines=$(wc -l < command_file)
    for i in $(seq $nlines); do
        padi=$(printf '%03d\n' $i)
        file=seq${padi}.q
        echo '#!/bin/bash' > $file
        echo '#SBATCH --output path' >> $file
        head -$i command_file | tail -1 >> $file
    done

qpool can be used to manage a very large set of jobs.

########################################################################

### Requesting resources in Slurm:

To submit batch job scripts in Slurm, the sbatch command is used. In most cases it is essential to specify what resources your job needs. There are modest defaults set so very simple batch jobs will work OK.

Batch jobs on SC shared systems are allocated dedicated portions of the system. The key sbatch options for resources are time (--time), virtual memory (--mem), and nodes/cpus (--ntask-per-node and --cpus-per-tasks).

* sbatch resource request syntax
* Time
* Memory
* Nodes and CPUs
* Serial Jobs
* Distributed Memory Parallel Jobs
* Shared Memory / Multi-threading Parallel Jobs
* Queues
* Finding out resource and configuration information for each node
* Further information

For a quick summary of the SLURM commands, options and available environment variables, see the SLURM cheat sheet.

For job script examples, see Sample Slurm Job Scripts.

sbatch resource request syntax
When submitting a job to the cluster with sbatch it is necessary to include a resource request.
This can be done on the command line with with a lower case L argument

sbatch <resource options> script
or as part of the job script with

    #SBATCH <resource options>

The resource options are documented in the sbatch man page, but the most common ones we recommend are mentioned here.

#### Time
    --time=<minutes>
The total wall time limit for the job. Time can be expressed in minutess as an integer, or in the following formats: "hours:minutes:seconds","days-hours", "days-hours:minutes"

#### Memory
 
    ruby UV3000 memory

As ruby is a NUMA system it is critical to co-locate cores and local memory. As such the batch system has a default memory per core enabled (12.8GB) and you should not request --mem

    --mem=<MB>
The total memory limit (per node) for the job - can be specified with units of "MB" or "GB" but only integer values can be given. As an alternative you can specify the memory per cpu.

    --mem-per-cpu=<MB>
Specifying memory is especially important for allowing jobs to co-exist on nodes and for jobs requiring large memory to be scheduled on appropriate nodes. There is a small default value so you must explicitly specify --mem in most cases as jobs will otherwise fail as they exceed the default limit.

 

#### Request the memory you need

Your job will only run if there is sufficient free memory so making a sensible memory request will allow your jobs to run sooner. A little trial and error may be required to find how much memory your jobs are using. scontrol show job <jobid> lists the actual usage of a job. Since not all nodes have the same memory size, it is important to get this right to get the right types of nodes. See Estimating Memory Requirements for more guidance on making more accurate memory estimates.


#### Nodes and CPUs
To request tasks (which map to CPU cores) use; --ntasks=T or --ntasks-per-node=T

To request Nodes use; --nodes=N ( (warning) remembering on ruby there is only one big node)

To request four cores (assuming nodes have at least 4 cores) you could use:

    sbatch --ntasks-per-node=4
To request 2 nodes and 40 cpus (assuming a node has 20 cpus)

    sbatch --nodes=2 --ntasks-per-node=20
It is best to co-locate cores on as few nodes as possible. This increases your opportunity to have faster communication between cores and decreases the likelihood of your jobs having contention with others'.

#### ruby UV3000 specifics

ruby has a single node so please do not specify --nodes. It is important to place jobs on as few whole sockets as possible so we recommend using a multiple of 10 cores (or fewer than 10).

for larger jobs use:

    --ntasks-per-node=N --cores-per-socket=10

for smaller jobs use:

    --ntasks-per-node=N --cores-per-socket=N

If the number of processors your job needs cannot be evenly distributed onto multiple nodes (e.g. 2 whole nodes plus half a node), then you can use something like this:

    sbatch -ntasks=50
This will request 50 tasks spread across nodes.

In general, We do not recommend routine use of ntasks because it can cause and perpetuate fragmentation of the available resources, resulting in inefficient resources use. Please use --nodes=N --ntasks-per-node=M for most jobs.

If you request multiple cores using --ntasks=T the scheduler may be able to start your job sooner than if you were using the other syntax. However cores requested with --ntasks=T are likely to be spread in 'gaps' across the cluster and cause or perpetuate fragmentation of the available resource. Therefore you should only use --ntasks for short jobs where turnaround is critical, e.g. during code development.

(warning) On ruby using --ntasks may actively spread your job across the sockets causing poor performance and disrupting others. We are likely to need to kill or prevent such jobs on ruby.

The next section tells you how to find out the number of nodes, number of processors and amount of memory on each system.

#### Serial Jobs
Serial jobs (using one core only) do not need to specify --ntasks=T .

Distributed Memory Parallel Jobs
A parallel job that can use distributed memory (usually MPI) can use the form, but will usually perform more consistently with the --nodes=N --ntasks-per-node=M format and should usually use whole nodes.

Jobs requiring a hybrid of distributed and shared memory (such as hybrid OpenMP/MPI) will need to use the --ntasks-per-node=M format .

The number of cores on each node determines the maximum number you can request for --ntasks-per-node. This is different for each cluster. You can query for information about the nodes on a given cluster with scontrol show node <node>

Shared Memory / Multi-threading Parallel Jobs
A shared memory (eg. OpenMP) job should use --cpus-per-task=N where N is the number of threads a single task will launch and --nodes=1 and --ntasks-per-node=1 should also be specified.

#### Queues
The clusters (pearcey and bragg) have an queues for specific tasks - eg extended queues, io queues etc. See: Job Queues

 

#### Which nodes and cores are allocated?

Once your batch job has started, within the batch environment there is a variable $SLURM_JOB_NODELIST which contains the names of the nodes allocated to you, in a compact form. You can expend the list with 'scontrol show hostnames $SLURM_JOB_NODELIST'. This can be useful if you are crafting a custom parallel job but in the most common case (using MPI) the assigned nodes and cores will be used automatically. The srun command or mpirun from openmpi can be used to launch processes on the allocated nodes.


Finding out resource and configuration information for each node
Running the below command will give you information about the node configuration and whether of not they are in use:

    scontrol show node <node>
An example of the output:

    pearcey-login:~> scontrol show node c015
    NodeName=c015 Arch=x86_64 CoresPerSocket=10
        CPUAlloc=20 CPUErr=0 CPUTot=20 CPULoad=20.04 Features=(null)
        Gres=(null)
        NodeAddr=c015 NodeHostName=c015 Version=14.03.0
        OS=Linux RealMemory=129151 AllocMem=10240 Sockets=2 Boards=1
        State=ALLOCATED ThreadsPerCore=1 TmpDisk=201586 Weight=1
        BootTime=2014-10-31T09:51:14 SlurmdStartTime=2014-11-17T10:51:13
        CurrentWatts=0 LowestJoules=0 ConsumedJoules=0
        ExtSensorsJoules=n/s ExtSensorsWatts=0 ExtSensorsTemp=n/s
In the above example, you will see that the node (on Pearcey) has 2 sockets with 10 cores per socket, or 20 cpus in total. this node has 128GB of memory.

#### Further information
* see "man sbatch".

* How to estimate the required walltime
