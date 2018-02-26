#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <stddef.h>
#include <opencv2\opencv.hpp>
#include <cuComplex.h>

#define SWAP(g,h) {y=(g);(g)=(h);(h)=y;}
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define RADIX 2.0
#define NP 3

using namespace std;
using namespace cv;

__host__ __device__ void obtener_cadena_binaria(unsigned int lbits, unsigned long long n, short* mbits) {
	long int c, k;
	for (c = lbits - 1; c >= 0; c--) {
		k = n >> c;
		if (k & 1) {
			mbits[lbits - 1 - c] = 1;
		}
		else {
			mbits[lbits - 1 - c] = 0;
		}
	}
}

__host__ __device__ void obtener_inversa_matriz3x3(float a[], float ainv[]) {
	float out[9];
	out[0] = a[4] * a[8] - a[5] * a[7];
	out[1] = (a[3] * a[8] - a[5] * a[6])*-1;
	out[2] = a[3] * a[7] - a[4] * a[6];
	out[3] = (a[1] * a[8] - a[2] * a[7])*-1;
	out[4] = a[0] * a[8] - a[2] * a[6];
	out[5] = (a[0] * a[7] - a[1] * a[6])*-1;
	out[6] = a[1] * a[5] - a[2] * a[4];
	out[7] = (a[0] * a[5] - a[2] * a[3])*-1;
	out[8] = a[0] * a[4] - a[1] * a[3];
	float det = (a[0] * out[0]) + (a[1] * out[1]) + (a[2] * out[2]);
	int i, j;
	for (i = 0; i<3; i++)
		for (j = 0; j<3; j++)
			ainv[i * 3 + j] = (1 / det)*out[j * 3 + i];
}


// ##############################################################

// obtiene al maximo entre 2 valores
__host__ __device__ int IMAX(int a, int b){
	return (a>b)?a:b;
}

__host__ __device__ double SQR(float a){
	return (a==0.0)?0.0:a*a;
}

__host__ __device__ double machine_eps_dbl() {
    typedef union {
        long long i64;
        double d64;
    } dbl_64;

    dbl_64 s;

    s.d64 = 1.;
    s.i64++;
    return (s.d64 - 1.);
}

__host__ __device__ cuFloatComplex Complexn(float re, float im){
	cuFloatComplex c = make_cuFloatComplex(re,im);
	return c;
}

__host__ __device__ cuFloatComplex Conjg(cuFloatComplex z){
	cuFloatComplex c = cuConjf(z);
	return c;
}

__host__ __device__ cuFloatComplex Cdiv(cuFloatComplex a, cuFloatComplex b){
	cuFloatComplex c = cuCdivf(a,b);
	return c;
}

__host__ __device__ void balanc(double a[], int n, float scale[]){
	int done=0;
	double sqrdx=RADIX*RADIX;
	while (!done) {
		done=true;
		for (int i=0;i<n;i++) {
			double r=0.0,c=0.0;
			for (int j=0;j<n;j++)
				if (j != i) {
					c += abs(a[j*NP+i]);
					r += abs(a[i*NP+j]);
				}
			if (c != 0.0 && r != 0.0) {
				double g=r/RADIX;
				double f=1.0;
				double s=c+r;
				while (c<g) {
					f *= RADIX;
					c *= sqrdx;
				}
				g=r*RADIX;
				while (c>g) {
					f /= RADIX;
					c /= sqrdx;
				}
				if ((c+r)/f < 0.95*s) {
					done=false;
					g=1.0/f;
					scale[i] *= f;
					for (int j=0;j<n;j++) a[i*NP+j] *= g;
					for (int j=0;j<n;j++) a[j*NP+i] *= f;
				}
			}
		}
	}
}

__host__ __device__ void elmhes(double a[], int n, int perm[]){
	int m,j,i;
	float y,x;

	for (m=1;m<n-1;m++) {
		x=0.0;
		i=m;
		for (j=m;j<n;j++) {
			if (fabs(a[j*NP+(m-1)]) > fabs(x)) {
				x=a[j*NP+(m-1)];
				i=j;
			}
		}

		perm[m]=i;
		if (i != m) {
			for (j=m-1;j<n;j++) SWAP(a[i*NP+j],a[m*NP+j]);
			for (j=0;j<n;j++) SWAP(a[j*NP+i],a[j*NP+m]);
		}
		if (x) {
			for (i=m+1;i<n;i++) {
				if ((y=a[i*NP+(m-1)]) != 0.0) {
					y /= x;
					a[i*NP+(m-1)]=y;
					for (j=m;j<n;j++)
						a[i*NP+j] -= y*a[m*NP+j];
					for (j=0;j<n;j++)
						a[j*NP+m] += y*a[j*NP+i];
				}
			}
		}
	}
}

__host__ __device__ void eltran(int n, double a[], double zz[], int perm[]){
	for (int mp=n-2;mp>0;mp--) {
		for (int k=mp+1;k<n;k++)
			zz[k*NP+mp]=a[k*NP+mp-1];
		int i=perm[mp];
		if (i != mp) {
			for (int j=mp;j<n;j++) {
				zz[mp*NP+j]=zz[i*NP+j];
				zz[i*NP+j]=0.0;
			}
			zz[i*NP+mp]=1.0;
		}
	}
}

__host__ __device__ void hqr2(double a[], int n, cuFloatComplex wri[], double zz[]){
	int nn,m,l,k,j,its,i,mmin,na;
	double z,y,x,w,v,u,t,s,r,q,p,anorm=0.0,ra,sa,vr,vi;

	const double EPS =  machine_eps_dbl();
	
	for (i=0;i<n;i++)
		for (j=IMAX(i-1,0);j<n;j++)
			anorm += abs(a[i*NP+j]);
	
	nn=n-1;
	t=0.0;
	while (nn >= 0) {
		its=0;
		do {
			for (l=nn;l>0;l--) {
				s=abs(a[(l-1)*NP+(l-1)])+abs(a[l*NP+l]);
				if (s == 0.0) s=anorm;
				if (abs(a[l*NP+(l-1)]) <= EPS*s) {
					a[l*NP+(l-1)] = 0.0;
					break;
				}
			}
			x=a[nn*NP+nn];
			if (l == nn) {
				a[nn*NP+nn]=x+t;
				wri[nn] = Complexn(x+t,0);
				nn--;
			} else {
				y=a[(nn-1)*NP+(nn-1)];
				w=a[nn*NP+(nn-1)]*a[(nn-1)*NP+nn];
				if (l == nn-1) {
					p=0.5*(y-x);
					q=p*p+w;
					z=sqrt(abs(q));
					x += t;
					a[nn*NP+nn]=x;
					a[(nn-1)*NP+(nn-1)]=y+t;
					if (q >= 0.0) {
						z=p+SIGN(z,p);
						wri[nn-1]=wri[nn]=Complexn(x+z,0);
						if (z != 0.0) wri[nn]=Complexn(x-w/z,0);
						x=a[nn*NP+(nn-1)];
						s=abs(x)+abs(z);
						p=x/s;
						q=z/s;
						r=sqrt(p*p+q*q);
						p /= r;
						q /= r;
						for (j=nn-1;j<n;j++) {
							z=a[(nn-1)*NP+j];
							a[(nn-1)*NP+j]=q*z+p*a[nn*NP+j];
							a[nn*NP+j]=q*a[nn*NP+j]-p*z;
						}
						for (i=0;i<=nn;i++) {
							z=a[i*NP+(nn-1)];
							a[i*NP+(nn-1)]=q*z+p*a[i*NP+nn];
							a[i*NP+nn]=q*a[i*NP+nn]-p*z;
						}
						for (i=0;i<n;i++) {
							z=zz[i*NP+(nn-1)];
							zz[i*NP+(nn-1)]=q*z+p*zz[i*NP+nn];
							zz[i*NP+nn]=q*zz[i*NP+(nn)]-p*z;
						}
					} else {
						wri[nn]= Complexn(x+p,-z);
						wri[nn-1] = Conjg(wri[nn]);
					}
					nn -= 2;
				} else {
					//if (its == 30) throw("Too many iterations in hqr");
					if(its==30){
						//printf("Muchas iteraciones en hqr");
						return;
					}
					
					
					if (its == 10 || its == 20) {
						t += x;
						for (i=0;i<nn+1;i++) a[i*NP+i] -= x;
						s=abs(a[nn*NP+(nn-1)])+abs(a[(nn-1)*NP+(nn-2)]);
						y=x=0.75*s;
						w = -0.4375*s*s;
					}
					++its;
					for (m=nn-2;m>=l;m--) {
						z=a[m*NP+m];
						r=x-z;
						s=y-z;
						p=(r*s-w)/a[(m+1)*NP+m]+a[m*NP+(m+1)];
						q=a[(m+1)*NP+(m+1)]-z-r-s;
						r=a[(m+2)*NP+(m+1)];
						s=abs(p)+abs(q)+abs(r);
						p /= s;
						q /= s;
						r /= s;
						if (m == l) break;
						u=abs(a[m*NP+(m-1)])*(abs(q)+abs(r));
						v=abs(p)*(abs(a[(m-1)*NP+(m-1)])+abs(z)+abs(a[(m+1)*NP+(m+1)]));
						if (u <= EPS*v) break;
					}
					for (i=m;i<nn-1;i++) {
						a[(i+2)*NP+i]=0.0;
						if (i != m) a[(i+2)*NP+(i-1)]=0.0;
					}
					for (k=m;k<nn;k++) {
						if (k != m) {
							p=a[k*NP+(k-1)];
							q=a[(k+1)*NP+(k-1)];
							r=0.0;
							if (k+1 != nn) r=a[(k+2)*NP+(k-1)];
							if ((x=abs(p)+abs(q)+abs(r)) != 0.0) {
								p /= x;
								q /= x;
								r /= x;
							}
						}
						if ((s=SIGN(sqrt(p*p+q*q+r*r),p)) != 0.0) {
							if (k == m) {
								if (l != m)
								a[k*NP+(k-1)] = -a[k*NP+(k-1)];
							} else
								a[k*NP+(k-1)] = -s*x;
							p += s;
							x=p/s;
							y=q/s;
							z=r/s;
							q /= p;
							r /= p;
							for (j=k;j<n;j++) {
								p=a[k*NP+j]+q*a[(k+1)*NP+j];
								if (k+1 != nn) {
									p += r*a[(k+2)*NP+j];
									a[(k+2)*NP+j] -= p*z;
								}
								a[(k+1)*NP+j] -= p*y;
								a[k*NP+j] -= p*x;
							}
							mmin = nn < k+3 ? nn : k+3;
							for (i=0;i<mmin+1;i++) {
								p=x*a[i*NP+k]+y*a[i*NP+(k+1)];
								if (k+1 != nn) {
									p += z*a[i*NP+(k+2)];
									a[i*NP+(k+2)] -= p*r;
								}
								a[i*NP+(k+1)] -= p*q;
								a[i*NP+k] -= p;
							}
							for (i=0; i<n; i++) {
								p=x*zz[i*NP+k]+y*zz[i*NP+(k+1)];
								if (k+1 != nn) {
									p += z*zz[i*NP+(k+2)];
									zz[i*NP+(k+2)] -= p*r;
								}
								zz[i*NP+(k+1)] -= p*q;
								zz[i*NP+k] -= p;
							}
						}
					}
				}
			}
		} while (l+1 < nn);
	}
	
	
	if (anorm != 0.0) {
		for (nn=n-1;nn>=0;nn--) {
			p = cuCrealf( wri[nn]);
			q = cuCimagf(wri[nn]);
			na=nn-1;
			if (q == 0.0) {
				m=nn;
				a[nn*NP+nn]=1.0;
				for (i=nn-1;i>=0;i--) {
					w=a[i*NP+i]-p;
					r=0.0;
					for (j=m;j<=nn;j++)
						r += a[i*NP+j]*a[j*NP+nn];
					if ( cuCimagf(wri[i]) < 0.0) {
						z=w;
						s=r;
					} else {
						m=i;
						
						if (cuCimagf(wri[i]) == 0.0) {
							t=w;
							if (t == 0.0)
								t=EPS*anorm;
							a[i*NP+nn]=-r/t;
						} else {
							x=a[i*NP+(i+1)];
							y=a[(i+1)*NP+i];
							q=SQR(cuCrealf(wri[i])-p)+SQR(cuCimagf(wri[i]));
							t=(x*s-z*r)/q;
							a[i*NP+nn]=t;
							if (abs(x) > abs(z))
								a[(i+1)*NP+nn]=(-r-w*t)/x;
							else
								a[(i+1)*NP+nn]=(-s-y*t)/z;
						}
						t=abs(a[i*NP+nn]);
						if (EPS*t*t > 1)
							for (j=i;j<=nn;j++)
								a[j*NP+nn] /= t;
					}
				}
			} else if (q < 0.0) {
				m=na;
				if (abs(a[nn*NP+na]) > abs(a[na*NP+nn])) {
					a[na*NP+na]=q/a[nn*NP+na];
					a[na*NP+nn]=-(a[nn*NP+nn]-p)/a[nn*NP+na];
				} else {
					
					cuFloatComplex temp =  Cdiv(Complexn(0.0,-a[na*NP+nn]),Complexn(a[na*NP+na]-p,q)) ;

					a[na*NP+na]=cuCrealf(temp);
					a[na*NP+nn]=cuCimagf(temp);
				}
				a[nn*NP+na]=0.0;
				a[nn*NP+nn]=1.0;
				for (i=nn-2;i>=0;i--) {
					w=a[i*NP+i]-p;
					ra=sa=0.0;
					for (j=m;j<=nn;j++) {
						ra += a[i*NP+j]*a[j*NP+na];
						sa += a[i*NP+j]*a[j*NP+nn];
					}
					if (cuCimagf(wri[i]) < 0.0) {
						z=w;
						r=ra;
						s=sa;
					} else {
						m=i;
						if (cuCimagf(wri[i])== 0.0) {
							cuFloatComplex temp = Cdiv(Complexn(-ra,-sa),Complexn(w,q) );
							a[i*NP+na]=cuCrealf(temp);
							a[i*NP+nn]=cuCimagf(temp);
						} else {
							x=a[i*NP+i+1];
							y=a[(i+1)*NP+i];
							vr=SQR(cuCrealf(wri[i])-p)+SQR(cuCimagf(wri[i]))-q*q;
							vi=2.0*q*(cuCrealf(wri[i])-p);
							if (vr == 0.0 && vi == 0.0)
								vr=EPS*anorm*(abs(w)+abs(q)+abs(x)+abs(y)+abs(z));
							
							cuFloatComplex temp=Cdiv(Complexn(x*r-z*ra+q*sa,x*s-z*sa-q*ra),Complexn(vr,vi));

							a[i*NP+na]=cuCrealf(temp);
							a[i*NP+nn]=cuCimagf(temp);
							if (abs(x) > abs(z)+abs(q)) {
								a[(i+1)*NP+na]=(-ra-w*a[i*NP+na]+q*a[i*NP+nn])/x;
								a[(i+1)*NP+nn]=(-sa-w*a[i*NP+nn]-q*a[i*NP+na])/x;
							} else {
								cuFloatComplex temp=Cdiv(Complexn(-r-y*a[i*NP+na],-s-y*a[i*NP+nn]),Complexn(z,q));
								a[(i+1)*NP+na]=cuCrealf(temp);
								a[(i+1)*NP+nn]=cuCimagf(temp);
							}
						}
					}
					t=IMAX(abs(a[i*NP+na]),abs(a[i*NP+nn]));
					if (EPS*t*t > 1)
						for (j=i;j<=nn;j++) {
							a[j*NP+na] /= t;
							a[j*NP+nn] /= t;
						}
				}
			}
		}
		for (j=n-1;j>=0;j--)
			for (i=0;i<n;i++) {
				z=0.0;
				for (k=0;k<=j;k++)
					z += zz[i*NP+k]*a[k*NP+j];
				zz[i*NP+j]=z;
			}
	}
}

__host__ __device__ void balbak(int n, double zz[], float scale[]){
	for (int i=0;i<n;i++)
		for (int j=0;j<n;j++)
			zz[i*NP+j] *= scale[i];
}

__host__ __device__ void sortvecs( cuFloatComplex wri[], double zz[]){
	int i;
	double temp[NP];
	for (int j=1;j<NP;j++) {
		cuFloatComplex x=wri[j];
		for (int k=0;k<NP;k++)
			temp[k]=zz[k*NP+j];
		for (i=j-1;i>=0;i--) {
			if (cuCrealf(wri[i]) >= cuCrealf(x)) break;
			wri[i+1]=wri[i];
			for (int k=0;k<NP;k++)
				zz[k*NP+i+1]=zz[k*NP+i];
		}
		wri[i+1]=x;
		for (int k=0;k<NP;k++)
			zz[k*NP+i+1]=temp[k];
	}
}

__host__ __device__ void eig(double a[], double eigVal[], double zz[]){
	int i,j;
	cuFloatComplex wri[NP];
	int perm[NP];
	for(i=0;i<NP;i++)perm[i]=0;

	float scale[NP];
	for(i=0;i<NP;i++){
		for(j=0;j<NP;j++){
			zz[i*NP+j]=0.0;
		}
		zz[i*NP+i] = 1.0;
		scale[i] = 1.0;
	}

	balanc(a,NP,scale);
	elmhes(a,NP,perm);
	eltran(NP,a,zz,perm);
	hqr2(a,NP,wri,zz);
	balbak(NP,zz,scale);
	sortvecs(wri,zz);

	for(i=0;i<NP;i++) {eigVal[i] = cuCrealf(wri[i]);}
}