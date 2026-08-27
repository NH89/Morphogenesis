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
