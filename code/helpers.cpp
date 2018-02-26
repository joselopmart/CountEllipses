#include <iostream>
#include <opencv2\opencv.hpp>
#include "helpers.h"
using namespace std;
using namespace cv;

void pausa(string str) {
	if (str.length() > 0)
		cout << str;
	cout << "\nOprime una tecla para continuar..." << endl;
	(void)getchar();
}

void mostrar(Mat img, int opt_show, string titulo) {
	namedWindow(titulo, opt_show);
	imshow(titulo, img);
	waitKey();
}

Mat binarizar(Mat img, ushort opt) {
	Mat gray, intermedio, bin;
	cvtColor(img, gray, CV_BGR2GRAY);

	if (opt == _BW_GLOBAL_) {
		GaussianBlur(gray, intermedio, Size(5, 5), 0);
		threshold(intermedio, intermedio, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	}
	else {
		adaptiveThreshold(gray, intermedio, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);
	}

	//bitwise_not(intermedio, intermedio);
	//Mat cintermedio = intermedio.clone();
	//floodFill(cintermedio, Point(0, 0), 255);
	//bitwise_not(cintermedio, cintermedio);
	//bin = intermedio | cintermedio;

	return intermedio;
	
	//Mat bin2;
	//bitwise_not(bin,bin);
	////mostrar(bin);
	//Mat b1;
	//pyrUp(bin,b1);
	//for(int i=0;i<5;i++){
	//	medianBlur(b1,b1,7);
	//}

	//pyrDown(b1,b1);
	//threshold(b1,b1,200,255,THRESH_BINARY);

	// 
	//bitwise_not(b1,b1);
	//return b1;
}

Mat generar_contorno(Mat bw) {
	Mat ee = Mat(3, 3, CV_8U, Scalar(255));
	ee.at<uchar>(0, 0) = 0; ee.at<uchar>(2, 2) = 0;
	ee.at<uchar>(0, 2) = 0; ee.at<uchar>(2, 0) = 0;
	ee.at<uchar>(1, 1) = 0;
	Mat erosion;
	erode(bw, erosion, ee);
	return abs(bw - erosion)>128;
}

void obtener_coordenadas_contorno(Mat cont, vector<Point2f> &vec) {
	Mat bw = cont.clone();
	/* obtenemos los puntos del contorno */
	vector<vector<Point>> contour;
	vector<Vec4i>hierarchy;
	findContours(bw, contour, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	/* proceso para eliminaciones de puntos repetidos*/
	Point p;
	for (unsigned int i = 0; i < contour.size(); i++)
		for (unsigned int j = 0; j < contour[i].size(); j++) {
			p = contour[i][j];
			if ((int)bw.at<uchar>(p) > 0) {
				bw.at<uchar>(p) = 0;
				vec.push_back(p);
			}
		}
	
}

Mat obtener_ps_arco(Mat contorno, vector<Point2f> pxy, vector<unsigned int> &index_arco, vector<unsigned int> &longitud_arco, vector<Point2f> cbase, Mat bw_cont, bool seleccion_automatica) {
	float sam_D = 1.9f;
	int sam_L = 14, sam_R = 18;
	int sz_A = 15, sz_B = 4, sz_k = 5, sz_thr2 = 4, sz_th3 = 15;
	int opt = 1;
	// En caso se que se solicite, se ajustaran automaticamente los valores
	// en caso contrario, el método preseleccionado es sam04
	if (seleccion_automatica) {
		Mat base_cont = bw_cont.clone();
		unsigned int base_totalpp;
		vector<unsigned int> ia;
		vector<unsigned int> la;
		base_totalpp = sarfraz(cbase, base_cont, sz_A, sz_B, sz_thr2, sz_th3, ia, la);
		if (base_totalpp == 0)
			opt = 2;
		else
		{
			base_totalpp = sam04(cbase, sam_D, sam_L, sam_R, base_cont, ia, la);
			if (base_totalpp == 0)
				opt = 1;
			else {
				unsigned int maxiter = 20;
				bool encontrado = false;
				while (maxiter > 0 && base_totalpp > 0) {
					maxiter--;
					sam_D += 0.1;
					base_totalpp = sam04(cbase, sam_D, sam_L, sam_R, base_cont, ia, la);
					if (base_totalpp == 0)
						encontrado = true;
				}
				sam_D = (encontrado) ? sam_D : 1.9f;
				opt = 1;
			}
		}
	}
		
	Mat ps;
	cvtColor(contorno, ps, CV_GRAY2BGR);
	if (opt == 1)
		sam04(pxy, sam_D, sam_L, sam_R, ps, index_arco, longitud_arco,true);
	else {
		sarfraz(pxy, ps, sz_A,sz_B, sz_thr2, sz_th3,index_arco, longitud_arco,true);
	}
	
	return ps;
}

int sam04(vector<Point2f> coords, float D, int L, int R, Mat bw, vector<unsigned int> &index_arco, vector<unsigned int> &longitud_arco, bool dibujar) {
	int n = coords.size();
	float *Dj = new float[n];
	memset(Dj, 0, n * sizeof(n));
	Point2d pi, pk, pj;
	float mx, my, m, max, valdj, dj;
	int i, li, j, indx_max;

	for (i = 0; i < n; i++) {
		pi = coords.at(i);
		li = i + L;
		li = (li >= n) ? li - n : li;
		pk = coords.at(li);


		my = pk.y - pi.y;
		mx = pk.x - pi.x;

		m = my / mx;

		max = -999;

		indx_max = i;

		valdj = 0;


		for (j = i; j <= i + L; j++) {
			li = j;
			li = (li >= n) ? li - n : li;
			pj = coords.at(li);
			dj = fabs(pj.x - pi.x);

			if (mx != 0)
				dj = fabs(pj.y - m*pj.x + m*pi.x - pi.y) / sqrt(m*m + 1);

			if (max < dj) {
				max = dj;
				indx_max = li;
				valdj = dj;
			}

		}

		if (valdj > D)
			Dj[indx_max] = valdj;

	}

	float maxv;
	int indx;

	for (i = 0; i < n; i++) {
		if (Dj[i] > 0) {
			if (i - R >= 0) {
				maxv = Dj[i - R];
				indx = i - R;
			}
			else {
				maxv = Dj[i - R + n];
				indx = i - R + n;
			}

			for (j = i - R; j < i + R; j++) {
				li = j;

				if (li<0) {
					li = li + n;
				}
				else if (li >= n) {
					li = li - n;
				}

				if (maxv<Dj[li]) {
					maxv = Dj[li];
					indx = li;
				}

				Dj[li] = 0;
			}

			Dj[indx] = maxv;
		}

	}

	int sz = 0;

	for (i = 0; i < n; i++)
		sz += ((Dj[i] > 0) ? 1 : 0);

	if (dibujar) {
		int index = -1;
		for (i = 0; i < n; i++)
		{
			if (Dj[i] > 0) {
				index_arco.push_back(i);
				if (index >= 0)
					longitud_arco.push_back(i - index - 1);

				index = i;
				ellipse(bw, coords.at(i), Size(10, 10), 0, 0, 360, Scalar(0, 255, 0), 3);
			}
		}

		longitud_arco.push_back(i - index + index_arco.at(0)-1);
	}

	
	delete[]Dj;
	return sz;
}


int sarfraz(vector<Point2f> coords, Mat bw, int A, int B, int thr2, int th3, vector<unsigned int> &index_arco, vector<unsigned int> &longitud_arco, bool dibujar) {
	//int A = 15, B = 4, k = 5, thr2 = 4, th3 = 15;
	int k = 5;
	int n = coords.size();
	float *Gp;
	float *Rc;
	Gp = new float[n];
	Rc = new float[n];
	
	memset(Gp, 0, sizeof(int)*n);
	memset(Rc, 0, sizeof(float)*n);

	int i, j, j2, p, q, indx;
	int nE1i, nE2i, nR3i;

	float mxt1, myt1, mxt2, myt2, m, pxr, pyr, px1, px2, px3, px4, py1, py2, py3, py4, mnxr, mnyr, xc, yc, dif, minp;
	float rrx[5], rry[5];

	for (i = 0; i < n; i++) {
		mxt1 = myt1 = mxt2 = myt2 = 0;

		for (j = 0; j < k; j++) {
			j2 = i - j;
			j2 = (j2 >= 0) ? j2 : j2 + n;

			mxt1 += coords.at(j2).x;
			myt1 += coords.at(j2).y;

			j2 = i + j;
			j2 = (j2 < n) ? j2 : j2 - n;

			mxt2 += coords.at(j2).x;
			myt2 += coords.at(j2).y;
		}

		mxt1 /= k;
		myt1 /= k;
		mxt2 /= k;
		myt2 /= k;

		m = atan2(myt1 - myt2, mxt1 - mxt2);

		pxr = coords.at(i).x - A / 2;
		pyr = coords.at(i).y + B / 2;

		rrx[0] = cos(m)*pxr - sin(m)*pyr;
		rry[0] = sin(m)*pxr + cos(m)*pyr;

		rrx[1] = cos(m)*(pxr + A) - sin(m)*pyr;
		rry[1] = sin(m)*(pxr + A) + cos(m)*pyr;

		rrx[2] = cos(m)*(pxr + A) - sin(m)*(pyr - B);
		rry[2] = sin(m)*(pxr + A) + cos(m)*(pyr - B);

		rrx[3] = cos(m)*pxr - sin(m)*(pyr - B);
		rry[3] = sin(m)*pxr + cos(m)*(pyr - B);

		rrx[4] = cos(m)*pxr - sin(m)*pyr;
		rry[4] = sin(m)*pxr + cos(m)*pyr;



		px1 = rrx[0]; px2 = rrx[2];
		py1 = rry[0]; py2 = rry[2];

		px3 = rrx[1]; px4 = rrx[3];
		py3 = rry[1]; py4 = rry[3];

		mnxr = ((px1*py2 - py1*px2)*(px3 - px4) - (px1 - px2)*(px3*py4 - py3*px4)) / ((px1 - px2)*(py3 - py4) - (py1 - py2)*(px3 - px4));
		mnyr = ((px1*py2 - py1*px2)*(py3 - py4) - (py1 - py2)*(px3*py4 - py3*px4)) / ((px1 - px2)*(py3 - py4) - (py1 - py2)*(px3 - px4));

		for (p = 0; p < 5; p++) {
			rrx[p] += (coords.at(i).x - mnxr);
			rry[p] += (coords.at(i).y - mnyr);
		}

		nE1i = nE2i = nR3i = 0;

		Point2f pi = coords.at(i);
		Point2f ptemp;
		int dif_l = A * 2;
		int index_p;
		for (p = i - dif_l; p < i + dif_l; p++) {
			index_p = (p < 0) ? (p + n) : p;
			index_p = (index_p >= n) ? (index_p - n) : index_p;
			ptemp = coords.at(index_p);
			xc = (ptemp.x - pi.x)*cos(m) + (ptemp.y - pi.y)*sin(m);
			yc = -(ptemp.x - pi.x)*sin(m) + (ptemp.y - pi.y)*cos(m);
			dif = (xc*xc) / (A*A) + (yc*yc) / (B*B);
			if (dif <= 1)
				nE1i++;

			dif = (xc*xc) / (0.75*A*0.75*A) + (yc*yc) / (B*B);

			if (dif <= 1) {
				nE2i += 1;
				if (wn_PnPoly(ptemp.x, ptemp.y, rrx, rry, 5) != 0) {
					nR3i++;
				}
			}

		}
		
		dif = abs(nE1i - nE2i);

		if (dif < thr2) {
			Gp[i] = 1;
			Rc[i] = nR3i;
		}

	}

	int ini, fin, cont;
	j = 0;
	while (j < n) {
		if (Gp[j] > 0) {
			ini = j;
			cont = 15;
			p = j + 1;
			while (cont > 0) {
				if (p <= n) {
					cont = cont - ((Gp[p] == 1) ? 0 : 1);
					p = p + 1;
				}
				else {
					p = p - 1;
					break;
				}
			}
			fin = p;

			minp = Rc[j];

			indx = j;

			for (p = ini; p <= fin; p++) {
				if (p<n && Gp[p]>0) {
					if (minp > Rc[p]) {
						indx = p;
						minp = Rc[p];
					}
					Gp[p] = 0;
				}
			}

			if (Rc[indx] <= th3)
				Gp[indx] = 1;

			j = fin;
		}
		else {
			j++;
		}
	}

	cont = 0;

	for (i = 0; i < n; i++)
		if (Gp[i] > 0)
			cont++;
		
	if (dibujar) {
		int index = -1;
		for (i = 0; i < n; i++)
		{
			if (Gp[i] > 0) {
				index_arco.push_back(i);
				if (index >= 0)
					longitud_arco.push_back(i - index - 1);

				index = i;
				ellipse(bw, coords.at(i), Size(10, 10), 0, 0, 360, Scalar(0, 255, 0), 3);
			}
		}

		longitud_arco.push_back(i - index+index_arco.at(0) - 1);
	}


	delete[] Gp;
	delete[] Rc;
	return cont;
}

int wn_PnPoly(int x0, int y0, float sx[], float sy[], int n) {
	int    wn = 0;

	// loop through all edges of the polygon
	for (int i = 0; i<n; i++) {   // edge from V[i] to  V[i+1]
		if (sy[i] <= y0) {          // start y <= P.y
			if (sy[i + 1]  > y0)      // an upward crossing
				if (isLeft((int)sx[i], (int)sy[i], (int)sx[i + 1], (int)sy[i + 1], x0, y0) > 0)  // P left of  edge
					++wn;            // have  a valid up intersect
		}
		else {                        // start y > P.y (no test needed)
			if (sy[i + 1] <= y0)     // a downward crossing
				if (isLeft((int)sx[i], (int)sy[i], (int)sx[i + 1], (int)sy[i + 1], x0, y0) < 0)  // P right of  edge
					--wn;            // have  a valid down intersect
		}
	}
	return wn;
}

int isLeft(int x0, int y0, int x1, int y1, int x2, int y2) {
	return ((x1 - x0)*(y2*y0) - (x2 - x0)*(y1 - y0));
}