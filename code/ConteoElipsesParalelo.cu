/* bibliotecas CUDA */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

/* bibliotecas de C y C++ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <ctime>

/* bibliotecas de OpenCV */ 
#include <opencv2\opencv.hpp>

/* bibliotecas personales */
#include "funciones.cuh"
#include "helpers.h"

/* constantes definidas con la macro #define */
#define EPSILON_DIF 2 // diferencia minima de aceptacion para ecuaciones de combinaciones
#define MAX_THREADS 262144 // debe ser un valor basado en 2^(N) -> 2^18 = 262144 
#define MAX_THREADS_PER_BLOCK 256 // maximo de hilos por bloque
#define MAX_NUM_BLOCK (MAX_THREADS/MAX_THREADS_PER_BLOCK) //maximo numero de bloques
#define MAX_NPA 2500 //maximo de memoria para caja ajuste
#define LIM_INF_DAT_COMB 30 // limite inferior para tamaño de datos combinatorios

/* utilizaremos los espacios de nombres */
using namespace std;
using namespace cv;

__device__ int dev_indice_csel;

/* obtener parametros de la elipse base y retornar la imagen binaria*/
Mat proceso_imagen_base(double a[], vector<Point2f> &cbase, Mat imagen_base, const ushort opt = _BW_GLOBAL_);

/* copia valores de vectores de C++ en memoria creada con malloc de C */
void copiar_vector_a_memoriac(vector<Point2f> coords, vector<unsigned int> index_arco, vector<unsigned int> longitud_arco, int *x, int *y, int *lon_arcs, int total_puntos_contorno, int NPS);

/* dibuja las elipses obtenidas en el proceso combinatorio */
void dibujar_resultados(Mat img, int elipses, vector<double> params_elipses, double tiempo);

/* ajusta la mejor elipse a un conjunto de puntos datos */
__host__ __device__ void FitEllipse(double a[], float *X, float *Y, long int npuntos);

/* Funcion donde se realiza el proceso combinatorio entre los arcos del conglomerado simulando el proceso paralelo a utilizar */
unsigned int proceso_combinatorio_paralelo(int *man_x, int *man_y, int *man_lon_arcs, int NPSt, int total_puntos_contorno, double a[], vector<double> &params_elipses,int limite_superior_datos_combinacion);
/****************************************************************/
/************** INICIA DEFINICION DE KERNELS ********************/
/****************************************************************/

__global__ void kernel_obtener_lista_evaluacion(int *LISTA_CSEL, int NPS, int *lon_arcs, int indexItProcesadas, short *str_bins, int limite_superior_datos_combinacion, long long maxIter)
{
	// identificador del hilo
	const unsigned int id = (blockIdx.x*blockDim.x)+threadIdx.x;

	// combinacion a ser analizada
	unsigned long long combinacion = id + indexItProcesadas;

	// si la combinacion es mayor a maxIter, entonces automaticamente se
	// vuelve combinacion*0, lo que implica que no se analizara
	// en caso contrario, simplemente es una asignacion combinacion*1
	combinacion = combinacion*((int)(combinacion<maxIter)*((int)((dev_indice_csel+1)<MAX_THREADS)));

	// obtenemos la cadena binaria asociada a la combinacion
	short *str_bin = str_bins+(id*NPS);
	obtener_cadena_binaria(NPS, combinacion, str_bin);

	// contamos cuantos puntos tiene la combinacion
	int total_puntos = 0;
	for (int j = 0; j < NPS; j++)
		total_puntos += (lon_arcs[2*j+1]*str_bin[j]);

	if(combinacion>0 && total_puntos < limite_superior_datos_combinacion){
		// Guardamos el resultado
		__syncthreads();
		int nindx_guardado = atomicAdd(&dev_indice_csel,1);
		LISTA_CSEL[nindx_guardado] = combinacion;		
		
		
		//__syncthreads();
	}

}

__global__ void kernel_combinaciones_selectas(int *x, int *y, int *lon_arcs, int NPS, int total_puntos_contorno, long long maxIter, double *eqEllipses, int limite_superior_datos_combinacion, float *espacio_x, float *espacio_y, short *str_bins, int *LISTA_CSEL, double rMax, double rMin)
{
	const unsigned int id = (blockIdx.x*blockDim.x)+threadIdx.x;
	long long combinacion = LISTA_CSEL[id];
	if(combinacion>0){
		
		// espacio para la posible solucion a esta combinacion
		double tmp_a[] = {0.0, 0.0, 0.0, 0.0, 0.0};
		
		//obtenemos la cadena binaria asociada al numero de combinacion
		short *str_bin = str_bins+(id*NPS);
		obtener_cadena_binaria(NPS, combinacion, str_bin);

		// obtenemos el total de puntos en esta combinacion
		unsigned int total_puntos = 0;
		for (int j = 0; j < NPS; j++)
			total_puntos += (lon_arcs[2 * j + 1] * str_bin[j]);

		// preparamos el espacio de memoria
		float *s_x = espacio_x + (id*limite_superior_datos_combinacion); // le asignamos un espacio de memoria ya separada en el host
		float *s_y = espacio_y + (id*limite_superior_datos_combinacion);

		// obtenemos las coordenadas
		int l,m,o,k = 0;
		for (int j = 0; j < NPS & k < limite_superior_datos_combinacion; j++) {
			if (str_bin[j] == 1 & k < limite_superior_datos_combinacion) {
				l = lon_arcs[2 * j];
				m = lon_arcs[2 * j + 1] + l;
				for (o = l; o <= m && k < limite_superior_datos_combinacion; o++, k++){
					s_x[k] = x[o];
					s_y[k] = y[o];
				}
			}
		}

		// ajustamos la mejor elipse
		FitEllipse(tmp_a, s_x, s_y, (total_puntos < limite_superior_datos_combinacion) ? total_puntos : limite_superior_datos_combinacion);

		// aplicamos el ********** PRIMER CRITERIO DE SELECCION *******
		double rMaxtmp = tmp_a[2] > tmp_a[3] ? tmp_a[2] : tmp_a[3];
		double rMintmp = tmp_a[2] < tmp_a[3] ? tmp_a[2] : tmp_a[3];

		// si cumple la restriccion entonces los valores son guardados
		if (fabs(rMax - rMaxtmp) < EPSILON_DIF && fabs(rMin - rMintmp) < EPSILON_DIF) {
			eqEllipses[6 * id] = tmp_a[0];
			eqEllipses[6 * id + 1] = tmp_a[1];
			eqEllipses[6 * id + 2] = tmp_a[2];
			eqEllipses[6 * id + 3] = tmp_a[3];
			eqEllipses[6 * id + 4] = tmp_a[4];
			eqEllipses[6 * id + 5] = (double)combinacion;
		}
	}
}

/***************************************************************/
/************** TERMINA DEFINICION DE KERNELS ******************/
/***************************************************************/


std::string generate_filename(string path) {
	int l = path.length();
	int index_separator=0;
	for (unsigned int i = l - 1; i >= 0; i--) {
		if (path.at(i) == '/' || path.at(i) == '\\') {
			index_separator = i;
			break;
		}
	}

	if (index_separator == 0) { // no hay ruta, solo archivo
		return string("RES_") + path;
	}
	string filename = path.substr(index_separator + 1, string::npos);
	string only_path = path.substr(0, index_separator + 1);
	string str_new_filename = only_path + string("RES_") + filename;
	return str_new_filename;
}

Mat obtener_rect_base(Mat img, int w, int h){

	int width = (w%2==0)?w:(w-1);
	int heigth = (h%2==0)?h:(h-1);
	Mat a2 = img.clone();
	Rect area(0,0,width,heigth);
	Mat base = a2(area);
	return base;
}


/*
Linea de ejecucion
ce.exe imagen tipo param1, param2

@tipo 1->(param1 => x , param2 => y), 2->(param1 => coord_x, param2 => coord_y)

*/
int main(int argc, char *argv[]){


	if (argc == 5) {

		string conglomerado = argv[1];
		int tipo = atoi(argv[2]);
		string opt_eq = "+e";
		
		// 1. Primero leemos las imagenes para asegurarnos que existen ambas
		Mat img_original = imread(conglomerado);
		// si no se encuentra la base terminamos el programa
		if (!img_original.data) {
			cout << "-1, Imagenes no encontradas" << endl;
			return -1;
		}

		double rx, ry;
		rx = atof(argv[3]);
		ry = atof(argv[4]);

		// espacio de memoria para la ecuacion de la imagen base
		double a[] = { 0.0, 0.0, 0.0, 0.0, 0.0 }; 
		int limite_superior_datos_combinacion = 0;
		vector<Point2f> cbase;
		Mat imagen_base_binaria_contorno;


		if(tipo==1){

			// significa que rx y ry representan a los radios de la elipse a detectar
			a[2] = rx;
			a[3] = ry;
			limite_superior_datos_combinacion = PI*(3*(a[2]+a[3])-sqrt((3*a[2]+a[3])*(a[2]+3*a[3])));

		}else if(tipo == 2){

			// debemos obtener el cuadrante superior encerrado por el punto rx,ry
			Mat imagen_base = obtener_rect_base(img_original, (int)rx, (int) ry);
			imagen_base_binaria_contorno = proceso_imagen_base(a, cbase, imagen_base, _BW_GLOBAL_);
			limite_superior_datos_combinacion = cbase.size();

		}

		// 2. Proceso con el conglomerado
		// Objetos Mat para guardar los datos de las diferentes imagenes
		Mat img_binaria, img_contorno, img_ps;
		img_binaria = binarizar(img_original);
		img_contorno = generar_contorno(img_binaria);

		imwrite(conglomerado+"binaria.jpg",img_binaria);
		imwrite(conglomerado+"contornos.jpg",img_contorno);
		

		vector<Point2f> coords;
		vector<unsigned int> index_arco;
		vector<unsigned int> longitud_arco;
		obtener_coordenadas_contorno(img_contorno, coords);
		
		img_ps = obtener_ps_arco(img_contorno, coords, index_arco, longitud_arco,cbase, imagen_base_binaria_contorno,true);
		imwrite(conglomerado+"ps.jpg",img_ps);
		unsigned int NPS = index_arco.size();
		int total_puntos_contorno = coords.size();
		printf("total ps: %d\n", total_puntos_contorno);

		// Memoria CUDA C para procesamiento principal combinatorio
		int *x, *y, *lon_arcs;
		cudaMallocManaged(&x, sizeof(int)*total_puntos_contorno);
		cudaMallocManaged(&y, sizeof(int)*total_puntos_contorno);
		cudaMallocManaged(&lon_arcs, sizeof(int)*2*NPS);
		
		// copiamos los datos en memoria teniendo en cuenta el inicio de indices en 0
		copiar_vector_a_memoriac(coords, index_arco, longitud_arco, x, y, lon_arcs, total_puntos_contorno,NPS);
		
		unsigned int total_elipses = 0;
		vector<double> params_elipses; // se almacenan las elipses obtenidas
		clock_t tiempo_inicio, tiempo_fin;
		tiempo_inicio = clock();
		total_elipses = proceso_combinatorio_paralelo(x, y, lon_arcs, NPS, total_puntos_contorno, a, params_elipses,limite_superior_datos_combinacion);
		tiempo_fin = clock();
		double tiempo_total = double(tiempo_fin - tiempo_inicio) / CLOCKS_PER_SEC;
		
		cout << tiempo_total << "," << total_elipses << " elipses. " << NPS << " ps"<<endl;
		
		dibujar_resultados(img_original, total_elipses, params_elipses, tiempo_total);

		/* liberacion de memoria */
		cudaFree(x);
		cudaFree(y);
		cudaFree(lon_arcs);

		string filenameEnd = generate_filename(conglomerado);
		
		//string str_resultado(conglomerado + ".resultado_paralelo_v2.jpg");
		imwrite(filenameEnd, img_original);

		//if (opt_eq.compare("+e"))
		if (opt_eq=="+e")
		{
			for (int i = 0; i < total_elipses; i++) {
				cout << params_elipses.at(5*i)<<",";
				cout << params_elipses.at(5*i+1) << ",";
				cout << params_elipses.at(5*i+2) << ",";
				cout << params_elipses.at(5*i+3) << ",";
				cout << params_elipses.at(5*i+4) << endl;
			}
		}
		cudaDeviceReset();
	}
	else {
		cout << "-1,Parametros incompletos";
	}

	pausa("Terminado");
    return 0;
}

Mat proceso_imagen_base(double a[], vector<Point2f> &cbase, Mat imagen_base, const ushort opt) {
	Mat bin = binarizar(imagen_base, opt);
	Mat contorno = generar_contorno(bin);
	obtener_coordenadas_contorno(contorno, cbase);
	float *cx = (float *)malloc(sizeof(float)*cbase.size());
	float *cy = (float *)malloc(sizeof(float)*cbase.size());
	Point2f p;
	for (int i=0;i<cbase.size();i++)
	{
		p = cbase.at(i);
		cx[i] = p.x;
		cy[i] = p.y;
	}
	FitEllipse(a, cx, cy, cbase.size()>MAX_NPA?MAX_NPA: cbase.size());
	free(cx);
	free(cy);
	return contorno;
}

void copiar_vector_a_memoriac(vector<Point2f> coords, vector<unsigned int> index_arco, vector<unsigned int> longitud_arco, int *x, int *y, int *lon_arcs, int total_puntos_contorno, int NPS) {
	unsigned int index_pa = index_arco.at(0);
	unsigned int index;
	for (unsigned int i = 0; i < total_puntos_contorno; i++) {
		index = (i + index_pa < total_puntos_contorno) ? (i + index_pa) : (i + index_pa - total_puntos_contorno);
		x[i] = coords.at(index).x;
		y[i] = coords.at(index).y;
	}

	for (unsigned int i = 0; i < NPS; i++)
	{
		lon_arcs[2 * i] = index_arco.at(i) - index_pa;
		lon_arcs[2 * i + 1] = longitud_arco.at(i);
	}
}

void dibujar_resultados(Mat img, int elipses, vector<double> params_elipses, double tiempo) {
	ostringstream ne_strs, tiempo_strs;
	int thickness = 2; int lineType = 8;
	ne_strs << elipses;
	tiempo_strs << tiempo;
	string tiempo_str = ne_strs.str() + " elipses en " + tiempo_strs.str() + " segundos (P)";
	putText(img, tiempo_str, cvPoint(50, 50), FONT_HERSHEY_COMPLEX_SMALL, 1.5, cvScalar(0, 0, 255), 1, CV_AA);
	double a[5];
	for (int i = 0; i < elipses; i++) {
		a[0] = params_elipses.at(5*i);
		a[1] = params_elipses.at(5*i+1);
		a[2] = params_elipses.at(5*i+2);
		a[3] = params_elipses.at(5*i+3);
		a[4] = params_elipses.at(5*i+4);
		ellipse(img, Point(floor(a[0]), floor(a[1])),
			Size(a[2], a[3]),
			(180 * a[4]) / PI,
			0, 360,
			Scalar(0, 255, 0),
			thickness,
			lineType);
		ellipse(img, Point(floor(a[0]), floor(a[1])),
			Size(5,5),
			(180 * a[4]) / PI,
			0, 360,
			Scalar(0, 0, 255),
			thickness,
			lineType);
		putText(img, std::to_string((long double)(i+1)), cvPoint(floor(a[0]), floor(a[1])),
			FONT_HERSHEY_COMPLEX_SMALL, 1.5, cvScalar(0, 255, 0), 1, CV_AA);
	}

}

__host__ __device__ void FitEllipse(double a[], float *X, float *Y, long int npuntos){
	if (npuntos<LIM_INF_DAT_COMB) return;
	int i, j, k;
	float mx, my, sx, sy;
	float maxX, maxY, minX, minY;
	mx = X[0]; my = Y[0];
	maxX = X[0]; minX = X[0];
	maxY = Y[0]; minY = Y[0];

	for (i = 1; i<npuntos; i++) {
		mx += X[i]; my += Y[i];
		maxX = (maxX<X[i]) ? X[i] : maxX;
		minX = (minX>X[i]) ? X[i] : minX;
		maxY = (maxY<Y[i]) ? Y[i] : maxY;
		minY = (minY>Y[i]) ? Y[i] : minY;
	}

	mx /= npuntos; my /= npuntos;
	sx = (maxX - minX) / (2.0);
	sy = (maxY - minY) / (2.0);

	for (i = 0; i<npuntos; i++) {
		X[i] = (X[i] - mx) / sx; Y[i] = (Y[i] - my) / sy;
	}

	// Paso 1. crear el diseño de la matriz D = [x^2  xy  y^2  x  y  1]
	float D[MAX_NPA * 6];
	#pragma unroll
	for (i = 0; i<npuntos; i++) {
		D[6 * i] = X[i] * X[i];
		D[6 * i + 1] = X[i] * Y[i];
		D[6 * i + 2] = Y[i] * Y[i];
		D[6 * i + 3] = X[i];
		D[6 * i + 4] = Y[i];
		D[6 * i + 5] = 1.0;
	}

	// Paso 2. crear la matriz dispersa S = D'*D, que representa las sumatorias necesarias

	register float S[36];
	for (i = 0; i<36; i++)S[i] = 0;

	for (i = 0; i<6; i++) 
		for (j = 0; j<6; j++){
			#pragma unroll 
			for (k = 0; k<npuntos; k++) 
				S[6 * i + j] = S[6 * i + j] + (D[6 * k + i] * D[6 * k + j]);
		}
			
		
	// Paso 3. Rompemos D en bloques y resolvemos problemas eig
	float mA[9];
	float mB[9];
	float mC[9];
	float mCI[9];

	for (i = 0; i<3; i++) {
		for (j = 0; j<3; j++) {
			mA[3 * i + j] = S[6 * i + j];
			mB[3 * i + j] = S[6 * i + (j + 3)];
			mC[3 * i + j] = S[6 * (i + 3) + (j + 3)];
		}
	}

	obtener_inversa_matriz3x3(mC, mCI); // obtenemos la inversa de mC
	
	float mE[9];
	for (i = 0; i<9; i++)mE[i] = 0;

	for (i = 0; i<3; i++) { // calculamos inv(mC)*mB' <==> mE= mCI*mB'
		for (j = 0; j<3; j++) {
			for (k = 0; k<3; k++) {
				mE[3 * i + j] = mE[3 * i + j] + (mCI[3 * i + k] * mB[3 * j + k]);
			}
		}
	}

	float tmpMult[9];

	for (i = 0; i<9; i++)tmpMult[i] = 0;

	for (i = 0; i<3; i++) { // Espacio temporal tmpMult=mA-mB*mE; <==> inv(mD)*(mA-mB*mE) <==> mDI * tmpMult
		for (j = 0; j<3; j++) {
			for (k = 0; k<3; k++) {
				tmpMult[3 * i + j] = tmpMult[3 * i + j] + (mB[3 * i + k] * mE[3 * k + j]);
			}
			tmpMult[3 * i + j] = mA[3 * i + j] - tmpMult[3 * i + j];
		}
	}

	double eigVec[9];

	for (i = 0; i<9; i++)eigVec[i] = 0;

	for (i = 0; i<3; i++) {
		for (j = 0; j<3; j++) {
			for (k = 0; k<3; k++) {
				eigVec[3 * i + j] = eigVec[3 * i + j] + (mDI[3 * i + k] * tmpMult[3 * k + j]);
				if (eigVec[3 * i + j] != eigVec[3 * i + j]) {
					//printf("\n<<<Valores NaN en combinacion %d>>>\n",i);
					return;
				}
			}
		}
	}

	double evec_x[9];
	double eval_x[3];

	eig(eigVec, eval_x, evec_x); // resolvemos el problema de valores propios
	
								 // Paso 4. Obtencion del resultado
	int indexVec = -1;
	for (i = 0; i<3; i++) {
		if (eval_x[i]<1E-8 && eval_x[i]<10000) {
			indexVec = i;
			break;
		}
	}

	if (indexVec == -1) {
		return;
	}
	double A[6];

	A[2] = evec_x[6 + indexVec]; A[1] = evec_x[3 + indexVec]; A[0] = evec_x[0 + indexVec];

	for (i = 3; i<6; i++) {
		A[i] = 0;
		for (j = 0; j<3; j++) {
			A[i] += (mE[3 * (i - 3) + j] * A[j]);
		}
		A[i] *= -1;
	}

	// en A tenemos los parametros de la seccion conica [A B C D E F] <===> Ax^2 + Bxy + C y^2 + ... + F = 0
	double par[6];

	par[0] = A[0] * sy*sy;
	par[1] = A[1] * sx*sy;
	par[2] = A[2] * sx*sx;
	par[3] = -2 * A[0] * sy*sy*mx - A[1] * sx*sy*my + A[3] * sx*sy*sy;
	par[4] = -A[1] * sx*sy*mx - 2 * A[2] * sx*sx*my + A[4] * sx*sx*sy;
	par[5] = A[0] * sy*sy*mx*mx + A[1] * sx*sy*mx*my + A[2] * sx*sx*my*my - A[3] * sx*sy*sy*mx - A[4] * sx*sx*sy*my + A[5] * sx*sx*sy*sy;

	// convertir en radios geometricos y centros
	float thetarad = 0.5*atan2(par[1], par[0] - par[2]);
	float cost = cos(thetarad);
	float sint = sin(thetarad);
	float sin_squared = sint*sint;
	float cos_squared = cost*cost;
	float cos_sin = sint*cost;

	float Ao = par[5];
	float Au = par[3] * cost + par[4] * sint;
	float Av = -par[3] * sint + par[4] * cost;
	float Auu = par[0] * cos_squared + par[2] * sin_squared + par[1] * cos_sin;
	float Avv = par[0] * sin_squared + par[2] * cos_squared - par[1] * cos_sin;

	// rotada [Ao Au Av Auu Avv]
	float tuCentre = (-1 * Au) / (2 * Auu);
	float tvCentre = (-1 * Av) / (2 * Avv);
	float wCentre = Ao - Auu*tuCentre*tuCentre - Avv*tvCentre*tvCentre;

	float uCentre = tuCentre*cost - tvCentre*sint;
	float vCentre = tuCentre*sint + tvCentre*cost;

	float Ru = (-1 * wCentre) / Auu;
	float Rv = (-1 * wCentre) / Avv;

	int signou, signov;

	signou = (Ru<0) ? -1 : 1;
	signov = (Rv<0) ? -1 : 1;

	Ru = sqrt(abs(Ru))*signou;
	Rv = sqrt(abs(Rv))*signov;
	a[0] = uCentre;
	a[1] = vCentre;
	a[2] = Ru;
	a[3] = Rv;
	a[4] = thetarad;
}

unsigned int proceso_combinatorio_paralelo(int *man_x, int *man_y, int *man_lon_arcs, int NPSt, int total_puntos_contorno, double a[], vector<double> &params_elipses,int limite_superior_datos_combinacion){
	
	clock_t tIniP2, tFinP2, tIniP3, tFinP3;
	float tTotalP2=0, tTotalP3=0;
	
	// total de iteraciones inicialmente necesarias
	unsigned int NPS = (NPSt > LIM_INF_DAT_COMB) ? LIM_INF_DAT_COMB : NPSt;
	long long maxIter = pow(2.0, (int)NPS);

	// espacio para utilizar en las combinaciones con un tamaño maximo
	float *dev_espacio_x;
	float *dev_espacio_y;
	HANDLE_ERROR( cudaMalloc((void**)&dev_espacio_x, sizeof(float)*MAX_THREADS*limite_superior_datos_combinacion) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_espacio_y, sizeof(float)*MAX_THREADS*limite_superior_datos_combinacion) );

	// espacio para almacenar las cadenas binarias asociadas a cada combinacion
	short *dev_str_bins;
	HANDLE_ERROR( cudaMalloc((void**)&dev_str_bins, sizeof(short)*MAX_THREADS*NPS) );

	// espacios para almacenar las combinaciones que si se analizaran
	int *dev_LISTA_CSEL;
	size_t tam_mem_csel = sizeof(int)*MAX_THREADS;
	HANDLE_ERROR( cudaMalloc((void**)&dev_LISTA_CSEL, tam_mem_csel) );

	// espacios para almacenar las elipses mejor ajustadas
	double *host_eqEllipses;
	double *man_eqEllipses;

	host_eqEllipses = (double *)malloc(sizeof(double) * 6 * MAX_THREADS);
	HANDLE_ERROR( cudaMallocManaged((void**)&man_eqEllipses, sizeof(double)*6*MAX_THREADS) );

	double rMax, rMin;
	rMax = (a[2]>a[3]) ? a[2] : a[3];
	rMin = (a[2]<a[3]) ? a[2] : a[3];

	long long indexItProcesadas = 0;
	int preseleccionadas;
	int total_preseleccionadas = 0;
	int i, j;
	int ipcs = 0; // conteo de combinaciones que pasan el Primer Criterio de Seleccion
	int host_indice_sel;
	int last_indice_sel;
	while(indexItProcesadas<maxIter){
		// inicializamos memoria para proximas combinaciones selectas
		HANDLE_ERROR( cudaMemset(dev_LISTA_CSEL,0,tam_mem_csel) );

		// inicializamos la variable preseleccionadas para contar las proximas combinaciones selectas
		tIniP2 = clock();
		preseleccionadas = 0;
		host_indice_sel = 0; //-1 para que el primer elemento al aumentar 1 sea 0
		last_indice_sel = 0;
		cudaMemcpyToSymbol(dev_indice_csel, &host_indice_sel, sizeof(int));
		while(preseleccionadas < MAX_THREADS){
			// kernel con MAX_THREADS para elegir del rango [indexItProcesadas, indexItProcesadas+MAX_THREADS]
			// a las combinaciones que cumplan con la restriccion del tamaño
			//cout<<"analiza desde comb="<<indexItProcesadas<<" hasta "<<(indexItProcesadas+MAX_THREADS-1)<<endl;
			kernel_obtener_lista_evaluacion<<< MAX_NUM_BLOCK , MAX_THREADS_PER_BLOCK >>>(dev_LISTA_CSEL, NPS, man_lon_arcs, indexItProcesadas, dev_str_bins, limite_superior_datos_combinacion, maxIter);
			cudaDeviceSynchronize();

			// obtenemos el indice del valor en donde quedo el llenado de dev_LISTA_CSEL 
			cudaMemcpyFromSymbol(&host_indice_sel, dev_indice_csel, sizeof(int));
			
			// cuando host_indice_sel no sobrepasa MAX_THREADS
			// significa que todos los hilos (MAX_THREADS) fueron analizados
			i = host_indice_sel<MAX_THREADS?MAX_THREADS:0;
			
			// aumentamos el total de combinaciones ya seleccionadas
			if(last_indice_sel==0){
				preseleccionadas+=(host_indice_sel+1); //mas 1 porque host_indice_sel es base 0
				//ipcs +=(host_indice_sel+1);
				if(i==0){
					i=host_indice_sel+1;
				}
			}else{
				preseleccionadas+=(host_indice_sel-last_indice_sel);
				//ipcs+=(host_indice_sel-last_indice_sel);
				if(i==0){
					i=host_indice_sel-last_indice_sel;
				}
			}

			last_indice_sel = host_indice_sel;
			
			// 
			
			if (i < MAX_THREADS) {
				indexItProcesadas += (i);
				break;
			}
			
			// si condicion anterior NO se cumple entonces podemos analizar la siguiente particion de MAX_THREADS combs del total 2^n-1
			indexItProcesadas += (MAX_THREADS);

			// para el caso de combinaciones menores a MAX_THREADS, sera necesario solo una iteracion
			if (indexItProcesadas - MAX_THREADS >= maxIter)
				break;

		}
		tFinP2 = clock();
		tTotalP2 +=((float)(tFinP2-tIniP2)/CLOCKS_PER_SEC);
		//cout<<"-------------------------------------------"<<endl;
		//cout<<"Hasta ahora combinaciones validas"<<ipcs<<endl;
		tIniP3 = clock();
		// reiniciamos la memoria
		HANDLE_ERROR( cudaMemset(man_eqEllipses,0,sizeof(double)*MAX_THREADS*6) );

		// llamamos al kernel que obtiene los mejores ajustes a dichas combinaciones
		kernel_combinaciones_selectas<<< MAX_NUM_BLOCK, MAX_THREADS_PER_BLOCK >>>(man_x, man_y, man_lon_arcs, NPS, total_puntos_contorno, maxIter, man_eqEllipses, limite_superior_datos_combinacion, dev_espacio_x, dev_espacio_y, dev_str_bins, dev_LISTA_CSEL, rMax, rMin);
		cudaDeviceSynchronize();
		tFinP3 = clock();
		tTotalP3 +=((float)(tFinP3-tIniP3)/CLOCKS_PER_SEC);
		// una vez obtenidos todos los resultados para este conjunto de combinaciones
		// se procede a guardar aquellos que cumplieron con el
		// *********** PRIMER CRITERIO DE SELECCION ************
		for (i = 0; i < MAX_THREADS; i++) {
			if (man_eqEllipses[6 * i] > 0) {
				host_eqEllipses[6 * ipcs] = man_eqEllipses[6 * i];
				host_eqEllipses[6 * ipcs + 1] = man_eqEllipses[6 * i + 1];
				host_eqEllipses[6 * ipcs + 2] = man_eqEllipses[6 * i + 2];
				host_eqEllipses[6 * ipcs + 3] = man_eqEllipses[6 * i + 3];
				host_eqEllipses[6 * ipcs + 4] = man_eqEllipses[6 * i + 4];
				host_eqEllipses[6 * ipcs + 5] = man_eqEllipses[6 * i + 5];
				ipcs++;
			}
		}
	}
	//cout<<endl<<endl;

	// Al finalizar liberamos la memoria que ya no nos sera util

	// limpiar memoria del device
	cudaFree(dev_espacio_x);
	cudaFree(dev_espacio_y);
	cudaFree(dev_str_bins);
	cudaFree(dev_LISTA_CSEL);
	cudaFree(man_eqEllipses);

	// *********** INICIA PROCESO COMPLETAMENTE DEL HOST ***************


	// la variable "ipcs" mantiene el total de elipses que pasaron la primera restriccion

	// procedemos al segundo criterio de seleccion, para lo cual es necesario armar una matriz
	// binaria

	// en caso que no se hayan detectado elipses
	if (ipcs == 0) {
		cout<<"No se detectaron"<<endl;
		free(host_eqEllipses);
		return 0;
	}

	short *str_bin = (short *)malloc(sizeof(short)*NPS);
	short *cadenas_binarias = (short *)malloc(sizeof(short)*NPS*ipcs);
	double *ecuaciones = (double *)malloc(sizeof(double)*5*ipcs);
	int k;
	j = 0;
	for (i = 0; i < MAX_THREADS; i++) {
		if (host_eqEllipses[6 * i + 5] > 0) {
			obtener_cadena_binaria(NPS, (long long)host_eqEllipses[6 * i + 5], str_bin);
			for (k = 0; k<NPS; k++) { cadenas_binarias[NPS*j + k] = str_bin[k]; }
			ecuaciones[5 * j] = host_eqEllipses[6 * i];
			ecuaciones[5 * j + 1] = host_eqEllipses[6 * i + 1];
			ecuaciones[5 * j + 2] = host_eqEllipses[6 * i + 2];
			ecuaciones[5 * j + 3] = host_eqEllipses[6 * i + 3];
			ecuaciones[5 * j + 4] = host_eqEllipses[6 * i + 4];
			j++;
		}
	}

	// liberamos la memoria restante no necesaria
	free(host_eqEllipses);
	free(str_bin);

	// ******************* APLICAMOS EL SEGUNDO CRITERIO DE SELECCION ***********
	int o,p;
	for (i = 0; i<j; i++) {
		for (k = 0; k<NPS; k++) {
			if (cadenas_binarias[NPS*i + k] == 1) {
				for (o = i + 1; o<j; o++) {
					if (cadenas_binarias[NPS*o + k] == 1) {
						cadenas_binarias[NPS*o] = -1;
						for (p = 1; p<NPS; p++) { cadenas_binarias[NPS*o + p] = 0; }
					}
				}
			}
		}
	}

	// contamos el resultado
	int totalEllipses = 0;
	for (i = 0; i<j; i++)
		if (cadenas_binarias[NPS*i] >= 0) {
			totalEllipses++;
			params_elipses.push_back(ecuaciones[5 * i]);
			params_elipses.push_back(ecuaciones[5 * i + 1]);
			params_elipses.push_back(ecuaciones[5 * i + 2]);
			params_elipses.push_back(ecuaciones[5 * i + 3]);
			params_elipses.push_back(ecuaciones[5 * i + 4]);
		}

	free(cadenas_binarias);
	free(ecuaciones);
	cout<<"Tiempos"<<endl<<"Seleccion elipses validas: "<<tTotalP2<<" segundos"<<endl<<"Ajuste de elipses validas: "<<tTotalP3<<" segundo"<<endl;
	return totalEllipses;

}