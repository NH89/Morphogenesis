# Cuda patch

default Cuda 13.1 on Ubuntu 26.04 LTS requires a patch to work.

##The bug

Compilation with CUDA fails: execption specification is incompatible with that of previous function "rsqrt"



##The Cause

tl;dr: You'll need to patch the CUDA headers :-(

[Why is this happening?](https://stackoverflow.com/questions/79938413/compilation-with-cuda-fails-execption-specification-is-incompatible-with-that-o)

Recent versions of glibc (yours is 2.42) provide more and more math functions were not available in the past; and some of those are also provided as CUDA __device__ functions, declared in the CUDA headers. Last year it was sinpi(), cospi() and friends, this year it's rsqrt() and rsqrtf(). The basic signatures are identical, but - the functions may be declared as throwing or as non-throwing, depending on... stuff, and the CUDA distribution has to use a matching declaration.

Such a match can be achieved for all GCC versions by some macro work, but - NVIDIA have done their macro work only about declaration conflicts they already know about. For the additional functions, we must do that work ourselves - via a patch.

Credit goes to @mohangk and @morgwai666 on the NVIDIA developer forums, who have provided two alternative patches.

Both patches must be applied when your current directory is /usr/local/cuda/include/crt (or whatever your CUDA include directory is, in the crt/ subdirectory)

A patch for CUDA 13.0 and later
Newer versions of CUDA can be more flexible regarding different versions of GCC and their regimes of throwing exceptions from various functions. So, this patch is for newer versions of CUDA, and it users a macro-based mechanism for controlling whether or not such math functions are to be defined to throw exceptions:




##The patch

[forums.developer.nvidia.com/](https://forums.developer.nvidia.com/t/fedora-43-and-nvcc-cuda13-1-error-exception-specification-is-incompatible-rsqrt-rsqrtf/354510/5)

morgwai666  paul.mairo Mar 7

Newer toolkit versions (I think since 13) now have a special macro called __NV_IEC_60559_FUNCS_EXCEPTION_SPECIFIER to deal with this. You can try if maybe it works on clang with the following patch:

```
--- math_functions.h.orig	2026-03-07 12:36:19.092019479 +0100
+++ math_functions.h	2026-03-07 17:15:33.322027903 +0100
@@ -604,6 +604,11 @@ extern __DEVICE_FUNCTIONS_DECL__ __devic
 #if defined(__QNX__) && !defined(_LIBCPP_VERSION)
 } /* std */
 #endif
+#if __NV_GLIBC_PROVIDES_IEC_60559_FUNCS
+#define __NV_IEC_60559_FUNCS_EXCEPTION_SPECIFIER __THROW
+#else
+#define __NV_IEC_60559_FUNCS_EXCEPTION_SPECIFIER
+#endif
 /**
  * \ingroup CUDA_MATH_DOUBLE
  * \brief Calculate the reciprocal of the square root of the input argument.
@@ -626,7 +631,7 @@ extern __DEVICE_FUNCTIONS_DECL__ __devic
  *
  * \note_accuracy_double
  */
-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x) __NV_IEC_60559_FUNCS_EXCEPTION_SPECIFIER;
 
 /**
  * \ingroup CUDA_MATH_SINGLE
@@ -650,7 +655,8 @@ extern __DEVICE_FUNCTIONS_DECL__ __devic
  *
  * \note_accuracy_single
  */
-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x) __NV_IEC_60559_FUNCS_EXCEPTION_SPECIFIER;
+#undef __NV_IEC_60559_FUNCS_EXCEPTION_SPECIFIER
 
 #if defined(__QNX__) && !defined(_LIBCPP_VERSION)
 namespace std {
```
This is the same mechanism that is now in use for sinpi / cospi and their float variants that used to cause [similar problems](https://forums.developer.nvidia.com/t/error-exception-specification-is-incompatible-for-cospi-sinpi-cospif-sinpif-with-glibc-2-41/323591)



##  How to apply a .patch to source files

#### See: man patch

SYNOPSIS         top
       patch [options] [originalfile [patchfile]]

       but usually just

       patch -pnum <patchfile




####How to apply a patch to source files. I recently kludged my way through this and found no information in the puppy forums about how to actually apply patches.

https://forum.puppylinux.com/viewtopic.php?t=6011
Post by dogcat » Fri May 20, 2022 9:05 pm

--
Sometimes libraries and or program sources need to be patched and recompiled to fix issues.
Patches will have the file extension .patch, for example your-patch-file.patch

The following procedure is how I accomplished applying a .patch file in Puppy Linux.

1. Download the source files that need patching. Make certain they are the correct files to patch or patching will fail. Usually there is an MD5, SHA1, or SHA256 associated with an archive that you can verify the source archive with.

2. Locate the .patch (usually downloaded from somewhere but can come as additional sources on a CD for drivers, etc.)

3. Extract (uncompress) the sources if they are an archive. You should now have a directory containing the source files.

4. Copy your .patch file into the directory with the source files to be patched.

5. Load the DEVX (the DEVX will usually need to be loaded because that is where the patch utility is located)

6. Using your file manager (ROX in Puppy Linux) navigate to the extracted source directory that should now also contain the .patch file.

7. Open a terminal window in that directory (using ROX or whatever file manager or method you normally use to do that. Opening the terminal in the directory is the easiest way I know of)

8. It is now time to patch. There are a lot of command line options for patch, this command is what I found that works for me. In the terminal window use the following patch command:

Code:

```patch -Np1 -i your-patch-file.patch```

that will run the patch "your-patch-file.patch" against the proper files in the directory. Your actual .patch file name will be different.

P.S. After patching, make sure the patched files do not have a time in the future if you are compiling your patched files, that will prevent compiling loops.

