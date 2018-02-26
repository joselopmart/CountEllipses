#pragma once

#ifndef HELPERS
#define HELPERS
#include <iostream>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

const ushort _BW_GLOBAL_ = 1;
const ushort _BW_LOCAL_ = 2;

/*Agrega un (void)getchar() */
void pausa(string str="");

/* Muestra una imagen en pantalla */
void mostrar(Mat img, int opt_show = CV_WINDOW_NORMAL, string titulo = "Imagen");

/* Binariza una imagen de entrada @img */
Mat binarizar(Mat img, ushort opt = _BW_GLOBAL_);

/* generar contorno de imagen */
Mat generar_contorno(Mat bw);

/* Obtiene las coordenadas de una imagen de contornos */
void obtener_coordenadas_contorno(Mat cont, vector<Point2f> &vec);

/* Obtiene las secciones de arcos a partir de obtencion de ps */
Mat obtener_ps_arco(Mat contorno, vector<Point2f> pxy, vector<unsigned int> &index_arco, vector<unsigned int> &longitud_arco, vector<Point2f> cbase, Mat bw_cont, bool seleccion_automatica);

/* metodo de SAM04 para deteccion de esquinas */
int sam04(vector<Point2f> coords, float D, int L, int R, Mat bw, vector<unsigned int> &index_arco, vector<unsigned int> &longitud_arco, bool dibujar=false);

/* metodo SZ09 para deteccion de esquinas*/
int sarfraz(vector<Point2f> coords, Mat bw, int A, int B, int thr2, int th3, vector<unsigned int> &index_arco, vector<unsigned int> &longitud_arco, bool dibujar=false);

/* define si un punto dado se encuentra dentro de un poligono */
int wn_PnPoly(int x0, int y0, float sx[], float sy[], int n);

/* se utiliza en conjunto con wn_PnPoly(...) */
int isLeft(int x0, int y0, int x1, int y1, int x2, int y2);

#endif