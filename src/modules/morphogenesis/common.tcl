# BEGIN CONFIGURATION #########################################################
# Environment Settings
set             morphogenesis_home $::env(HOME)/apps/morphogenesis/0.1.1
setenv          MORPHOGENESIS_HOME $morphogenesis_home 
prepend-path	LD_RUN_PATH        $morphogenesis_home/lib64
prepend-path	LD_LIBRARY_PATH    $morphogenesis_home/lib64
prepend-path	PATH               $morphogenesis_home/bin


# END #########################################################################

# STOP HERE !!! ###############################################################
# DO NOT CHANGE THE BELOW #####################################################
global m_app_name
# Only set the app name if it hasn't been set
# i.e. a module file can specify a name that is different from the default
if { ! [info exists m_app_name] } {
  set m_app_name "load_sim"
}

# What users will see when running "module help <module>"
proc ModulesHelp { } {
  set desc "Morphogenesis is a biolgical morphogenesis simulator that can also be used for generation and evoluion of soft robots.It is a Cuda based GPU code that combines SPH viscous fuild with springs between particles for fibrous elasticity, mmorphogen diffusion between particles and epigenetic regulation of gene networks. It depends on VTK-9.x or higher for file out put for visualization in Paraview. It also outputs .csv files of data. For details see src/README.md"
  set link "https://github.com/NH89/Morphogenesis"
  puts stderr "${desc}\n\nSee more at ${link}.\n"
}

