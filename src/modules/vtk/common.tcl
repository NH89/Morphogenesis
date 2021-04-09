# BEGIN CONFIGURATION #########################################################
# Environment Settings
setenv          VTK_HOME           $::env(HOME)/apps/vtk/9.0.1
prepend-path    LD_RUN_PATH        $::env(VTK_HOME)/lib64
prepend-path    LD_LIBRARY_PATH    $::env(VTK_HOME)/lib64
prepend-path    LIBRARY_PATH       $::env(VTK_HOME)/lib64
prepend-path    CPATH              $::env(VTK_HOME)/include
prepend-path    PATH               $::env(VTK_HOME)/bin

# END #########################################################################

# STOP HERE !!! ###############################################################
# DO NOT CHANGE THE BELOW #####################################################
global m_app_name
# Only set the app name if it hasn't been set
# i.e. a module file can specify a name that is different from the default
if { ! [info exists m_app_name] } {
  set m_app_name "vtk"
}

# What users will see when running "module help <module>"
proc ModulesHelp { } {
  set desc "VTK-9.0.1 is a pre-requisite for Morphogenesis. See Morphogenesis README, and https://vtk.org"
  puts stderr "${desc}\n\nSee more at ${link}.\n"
}

