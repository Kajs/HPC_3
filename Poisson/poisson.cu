//extern "C" {
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>

#ifndef GPU_DEVICE
#define GPU_DEVICE 2
#endif

struct twoPointers {
  double* pointers[2];
  double* rad;
  int lastused;
};

__device__
void kernelPrintMatrix(int r, int c, double* A) {
  int i, j;
  for(i = 0; i < r; i++) {
    for(j = 0; j < c; j++) {
      printf("%lf ", A[i*c + j]);
    }
    printf("\n");
  }
  printf("\n");
}

__global__
void reference(int N, struct twoPointers mydata, int nthreads, int loops) {
  int i, r, c;
  int new_val = mydata.lastused;
  int old;
  //double epsilon = 0.0001;
  //double comparison = 4.0;
  //int count_it = 0;

  for (i = 0; i < loops; i++){ //&& comparison > epsilon ; i++) {
    //count_it +=1;
    old = mydata.lastused;
    if (new_val == 0) { mydata.lastused = 1; }
    else { mydata.lastused = 0; }
    new_val = mydata.lastused;
    //comparison =0.0;

    for(r = 1; r < N-1; r++) {
      for(c = 1; c < N-1; c++) {
        double val;
        val = mydata.pointers[old][(r-1)*N + c];
        val += mydata.pointers[old][(r+1)*N + c];
        val += mydata.pointers[old][r*N + c-1];
        val += mydata.pointers[old][r*N + c+1];
        val += mydata.rad[r*N + c];
        val = val * 0.25;
        mydata.pointers[new_val][r*N + c] = val;
        //double delta = val - mydata.pointers[old][r*N + c];
        //comparison += delta*delta;
      }
    }
  }
}

__global__
void naive(int N, struct twoPointers mydata, int nthreads, int loops) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; 
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= 1 && x < N-1 && y >= 1 && y < N - 1) {
    int i, r, c;
    int new_val;
    int old;


    old = mydata.lastused;
    if (old == 0) { new_val = 1; }
    else { new_val = 0; }

    double val;
    int myPos = y*N + x;
    val  = mydata.pointers[old][myPos - N];
    val += mydata.pointers[old][myPos + N];
    val += mydata.pointers[old][myPos - 1];
    val += mydata.pointers[old][myPos + 1];
    val += mydata.rad[myPos];
    val = val * 0.25;
    mydata.pointers[new_val][myPos] = val;
  }
}

extern "C" {
void printMatrix(int r, int c, double* A) {
  int i, j;
  for(i = 0; i < r; i++) {
    for(j = 0; j < c; j++) {
      printf("%lf ", A[i*c + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void writeMatrix(int N, double* data) {
  FILE *fpout;
  fpout = fopen("data.dat", "wb+");
    if (fpout == NULL) {
      printf("Error opening outfile!\n");
      exit(1);
  }

  int r, c;
  fprintf(fpout, "#x y z\n");
  for(r = 0; r < N; r++) {
    for(c = 0; c < N; c++) {
      double x = r * 2.0/((double) (N-1)) - 1;
      double y = c * 2.0/((double) (N-1)) - 1;
      fprintf(fpout, "%lf %lf %lf\n", x, y, data[r*N + c]);
    }
  }
  fclose(fpout);
}

int main(int argc, char *argv[]) {
  int N = 3;
  //int i;
  int nthreads;
  int completedLoops = -1;

  #pragma omp parallel
  { if(omp_get_thread_num()==0) { nthreads = omp_get_num_threads(); } } //end parallel
  if(argc >= 2) { N = atoi(argv[1]); }
  int mat_size = N*N*sizeof(double);

  int loops = N * N * log(N);
  
  double spacing = 2.0/((double) N);
  char* funcname = "ref";
  if(argc >= 3) { funcname = argv[2]; }
  if (argc >=4){ loops =atoi(argv[3]);}

  //ALLOCATION HOST
  double *h_matrix;
  double *h_matrix2;
  double *h_radiator;

  if ( (h_matrix = (double*)calloc( N*N, sizeof(double) )) == NULL ) 
    { perror("main(__LINE__), allocation failed"); exit(1); }
  if ( (h_matrix2 = (double*)calloc( N*N, sizeof(double) )) == NULL ) 
    { perror("main(__LINE__), allocation failed"); exit(1); }
  if ( (h_radiator = (double*)calloc( N*N, sizeof(double) )) == NULL ) 
    { perror("main(__LINE__), allocation failed"); exit(1); }

  //ALLOCATION DEVICE
  double *d_matrix;
  double *d_matrix2;
  double *d_radiator;

  cudaMalloc ((void**) &d_matrix, mat_size); 
  if (d_matrix == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return 1; }
  cudaMalloc ((void**) &d_matrix2, mat_size); 
  if (d_matrix2 == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return 1; }
  cudaMalloc ((void**) &d_radiator, mat_size); 
  if (d_radiator == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return 1; }

  double hotWall = 20.0;
  double coldWall = 0.0;
  //Initialization Outer
  int r,c;

  printf("main: initialize matrices 1\n");
  #pragma omp parallel for private(c)
  for(c = 0; c < N; c++) {
    h_matrix[0 + c] = hotWall;
    h_matrix[(N-1)*N + c] = hotWall;
    h_matrix2[0 + c] = hotWall;
    h_matrix2[(N-1)*N + c] = hotWall;
  }
  #pragma omp parallel for private(r)
  for(r = 0; r < N; r++) {
    h_matrix[r*N + 0] = coldWall;
    h_matrix2[r*N + 0] = coldWall;
    h_matrix[r*N + (N-1)] = hotWall;
    h_matrix2[r*N + (N-1)] = hotWall;
  }

  //Initialization Inner
  double initialGuess = 5.0; 
  #pragma omp parallel for private(r) private(c)
  for(r = 1; r < N-1; r++) {
    for(c = 1; c < N-1; c++) {
      h_matrix[r*N + c] = initialGuess;
    }
  }

  //Initialization radiator
  #pragma omp parallel for private(r) private(c)
  for(r = 1; r < N-1; r++) {
    for(c = 1; c < N-1; c++) {
      double x = r * 2.0/((double) (N-1)) - 1;
      double y = c * 2.0/((double) (N-1)) - 1;
      if(x >= 0 && x <= 1.0/3.0 && y >= -2.0/3.0 && y <= -1.0/3.0) { h_radiator[r*N + c]= 200.0*spacing*spacing; }
      else {h_radiator[r*N + c]= 0.0;}
    }
  }

  //TRANSFER HOST->DEVICE
  printf("main: beginning transfer\n");
  cudaMemcpy (d_matrix, h_matrix, mat_size, cudaMemcpyHostToDevice);
  cudaMemcpy (d_matrix2, h_matrix2, mat_size, cudaMemcpyHostToDevice);
  cudaMemcpy (d_radiator, h_radiator, mat_size, cudaMemcpyHostToDevice);

  struct twoPointers mydata;
  mydata.pointers[0] = d_matrix;
  mydata.pointers[1] = d_matrix2;
  mydata.rad = d_radiator;
  mydata.lastused = 0;

  //TIME JACOBI
  struct timeval t1, t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);
  if(strcmp(funcname, "ref") == 0) { 
    printf("main: starting jacobi reference\n"); 
    reference<<<1,1>>>(N, mydata, nthreads, loops); 
    cudaDeviceSynchronize();
  }
  else { 
    if(strcmp(funcname, "nai") == 0) {
      int factor = 16;
      int grid_x = (N-1)/factor + 1;
      int grid_y = (N-1)/factor + 1;
      int block_x = factor;
      int block_y = factor;
      int i;
  
      dim3 dimGrid(grid_x, grid_y, 1);
      dim3 dimBlock(block_x, block_y, 1);
      printf("main: starting jacobi naive\n");
      for (i = 0; i < loops; i++){ 
        naive<<<dimGrid, dimBlock>>>(N, mydata, nthreads, loops);
        cudaDeviceSynchronize();
        if(mydata.lastused == 0) { mydata.lastused = 1; }
        else { mydata.lastused = 0; } 
      } 
    }
  }
  //___TIME JACOBI
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000000.0;


  printf("main: kernel function finished\n");
  cudaMemcpy (h_matrix, mydata.pointers[mydata.lastused], mat_size, cudaMemcpyDeviceToHost);
  //printMatrix(N, N, h_matrix);
  writeMatrix(N, h_matrix);

  //FREE HOST
  free(h_matrix);
  free(h_matrix2);
  free(h_radiator);

  //FREE DEVICE
  cudaFree(d_matrix);
  cudaFree(d_matrix2);
  cudaFree(d_radiator);

  //printf("#nthreads, time, N, function, iterations, it/s, size\n");
  //double it_per_second = ((double) completedLoops)/elapsedTime;
  double size = ((double)(N*N*sizeof(double) * 3))/(1024*1024);
  double jacobiMflops = (7.0*(N-2)*(N-2)/1000000.0) / elapsedTime;
  printf("%i %lf %i %s %i %lf %lf %lf\n", nthreads, elapsedTime, N, funcname, completedLoops, size, jacobiMflops);
  return 0;
}
}
//}

