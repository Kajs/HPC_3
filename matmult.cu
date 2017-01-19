#include <stdio.h>
#include <stdlib.h>

__global__
void matmult_gpu1_compute(int m,int n,int k,double *d_A,double *d_B,double *d_C) {
  int i, j, l;

  //INITIALIZE DEVICE
  for(i = 0; i < m; i++) {
    for(j = 0; j < n; j++) {
      d_C[i*n + j] = 0.0;
    }
  }

  //MAIN CODE DEVICE
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      for (l = 0; l < k; l++) {
	d_C[i*n + j] += d_A[i*k + l] * d_B[l*n + j];
      }
    }
  }
}

__global__
void matmult_gpu2_compute(int m,int n,int k,double *d_A,double *d_B,double *d_C) {
  int i, l;
  int x = blockIdx.x * blockDim.x + threadIdx.x; 
  int y = blockIdx.y * blockDim.y + threadIdx.y; 
  i = n * y + x;

  if(x < n && y < m) {
    //printf("x = %i, y=%i, i=%i m=%i n=%i\n", x, y, i, m, n);
    d_C[i] = 0.0;
    for (l = 0; l < k; l++) {
      int A_pos = y*k + l;
      int B_pos = l*n + x;
      d_C[i] += d_A[A_pos] * d_B[B_pos];
    }
  }
}

extern "C" {
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef GPU_DEVICE
#define GPU_DEVICE 3
#endif

void matmult_nat(int m,int n,int k,double *A,double *B,double *C) {
  int i, j, l;
  // C initialization
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      C[i*n + j]=0;
    }
  }
  //main code
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      for (l = 0; l < k; l++) {
	C[i*n + j] += A[i*k + l]*B[l*n + j];
      }
    }
  }
}

void matmult_lib(int m, int n, int k, double *A, double *B, double *C) {
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k, B, n, 0, C, n);
}

void matmult_gpu1(int m,int n,int k,double *h_A,double *h_B,double *h_C) {
  //cudaSetDevice(GPU_DEVICE);
  double* d_A;
  double* d_B;
  double* d_C;
  int size_A = m*k*sizeof(double);
  int size_B = n*k*sizeof(double);
  int size_C = m*n*sizeof(double);

  //ALLOCACE DEVICE
  cudaMalloc ((void**) &d_A, size_A); 
  if (d_A == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return; }

  cudaMalloc ((void**) &d_B, size_B); 
  if (d_B == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return; }

  cudaMalloc ((void**) &d_C, size_C); 
  if (d_C == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return; }

  //TRANSFER HOST->DEVICE
  cudaMemcpy (d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy (d_B, h_B, size_B, cudaMemcpyHostToDevice);

  dim3 dimGrid(1, 1, 1);
  dim3 dimBlock(1, 1, 1);
  matmult_gpu1_compute<<<dimGrid, dimBlock>>>(m, n, k, d_A, d_B, d_C);
  cudaDeviceSynchronize();

  //TRANSFER DEVICE->HOST
  cudaMemcpy (h_C, d_C, size_C, cudaMemcpyDeviceToHost);

  //FREE DEVICE
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void matmult_gpu2(int m,int n,int k,double *h_A,double *h_B,double *h_C) {
  cudaSetDevice(2);
  double* d_A;
  double* d_B;
  double* d_C;
  int size_A = m*k*sizeof(double);
  int size_B = n*k*sizeof(double);
  int size_C = m*n*sizeof(double);

  //ALLOCACE DEVICE
  cudaMalloc ((void**) &d_A, size_A); 
  if (d_A == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return; }

  cudaMalloc ((void**) &d_B, size_B); 
  if (d_B == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return; }

  cudaMalloc ((void**) &d_C, size_C); 
  if (d_C == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return; }

  //TRANSFER HOST->DEVICE
  cudaMemcpy (d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy (d_B, h_B, size_B, cudaMemcpyHostToDevice);
  int factor = 16;

  int grid_x = n/factor + 1;
  int grid_y = m/factor + 1;

  int block_x = factor;
  int block_y = factor;
  
  dim3 dimGrid(grid_x, grid_y, 1);
  dim3 dimBlock(block_x, block_y, 1);
  printf("gpu2: running\n");
  matmult_gpu2_compute<<<dimGrid, dimBlock>>>(m, n, k, d_A, d_B, d_C);
  cudaDeviceSynchronize();

  //TRANSFER DEVICE->HOST
  cudaMemcpy (h_C, d_C, size_C, cudaMemcpyDeviceToHost);

  //FREE DEVICE
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
}
