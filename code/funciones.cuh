#ifndef FUNCIONES
#define FUNCIONES

// Inclusion de librerias
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2\opencv.hpp>
#include <cuComplex.h>

#define PI 3.141592653589793

/* maneja los errores que pudieran aparecer en el codigo */
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
	#if defined(DEBUG) || defined(_DEBUG)
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
    #endif
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__device__ __managed__ float mDI[] = {0.0, 0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0, 0.0};

/* 
Convierte un valor entero @n a una cadena binaria 
con una longitud maxima de @lbits bits y se almacena de @mbits (previamente con memoria asignada)
*/
__host__ __device__ void obtener_cadena_binaria(unsigned int lbits,unsigned long long n, short* mbits);

/*
Accion: @a es una matriz en notacion de array a la cual se le calcula su inversa
y se almacena en ainv. Unicamente para matrices de 3x3
*/
__host__ __device__ void obtener_inversa_matriz3x3(float a[], float ainv[]);

// Las siguientes funciones son utilizadas para la obtencion de los valores propios de una
// matriz general no simetrica de 3x3 

__host__ __device__ int IMAX(int, int);
__host__ __device__ float SQR(float);


/* limite de operaciones de tipo double */
__host__ __device__ double machine_eps_dbl();

/* operaciones con complejos */

// crear un valor complejo
__host__ __device__ cuFloatComplex Complexn(float,float);

// conjugado de un valor complejo
__host__ __device__ cuFloatComplex Conjg(cuFloatComplex);

// division con complejos
__host__ __device__ cuFloatComplex Cdiv(cuFloatComplex, cuFloatComplex);

/*Balanceo de una matriz*/
__host__ __device__ void balanc(double[], int, float []);

/* proceso para convertir matriz en hes */
__host__ __device__ void elmhes(double[], int, int[]);

/* proceso para ...*/
__host__ __device__ void eltran(int, double[], double[], int[]);

/*proceso para ...*/
__host__ __device__ void balbak(int, double[], float[]);

/* proceso para encontrar las raices y vectores propios*/
__host__ __device__ void hqr2(double[], int, cuFloatComplex[], double[]);

/* ordenar los vectores propios*/
__host__ __device__ void sortvecs(cuFloatComplex[], double []);

/* funcion principal para llamar cronologicamente  a la obtencion de los eigen*/
__host__ __device__ void eig(double[], double[], double[]);


#endif