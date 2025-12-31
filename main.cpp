#include <Accelerate/Accelerate.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <dispatch/dispatch.h>
#include <arm_neon.h>

using namespace std;

void generateInput(double* firstMatrix, double* secondMatrix, int n) {
    for (int i = 0; i < n * n; i++) {
        firstMatrix[i] = (double)rand() / RAND_MAX;
        secondMatrix[i] = (double)rand() / RAND_MAX;
    }
}

void resetResult(double* resultMatrix, int n) {
    for (int i = 0; i < n * n; i++) {
        resultMatrix[i] = 0.0;
    }
}

void matmulNaive(double* firstMatrix, double* secondMatrix, double* resultMatrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += firstMatrix[i * n + k] * secondMatrix[k * n + j];
            }
            resultMatrix[i * n + j] = sum;
        }
    }
}

void matmulOptimized(double* firstMatrix, double* secondMatrix, double* resultMatrix, int n) {
    int blockSize = 64;
    int numBlocks = n / blockSize;
    
    dispatch_apply(numBlocks, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t iiIdx) {
        int ii = (int)iiIdx * blockSize;
        for (int kk = 0; kk < n; kk += blockSize) {
            for (int jj = 0; jj < n; jj += blockSize) {
                for (int i = ii; i < min(ii + blockSize, n); i++) {
                    for (int k = kk; k < min(kk + blockSize, n); k++) {
                        double firstVal = firstMatrix[i * n + k];
                        float64x2_t aVec = vdupq_n_f64(firstVal);
                        
                        int j = jj;
                        for (; j + 2 <= min(jj + blockSize, n); j += 2) {
                            float64x2_t bVec = vld1q_f64(&secondMatrix[k * n + j]);
                            float64x2_t cVec = vld1q_f64(&resultMatrix[i * n + j]);
                            cVec = vfmaq_f64(cVec, aVec, bVec);
                            vst1q_f64(&resultMatrix[i * n + j], cVec);
                        }
                        for (; j < min(jj + blockSize, n); j++) {
                            resultMatrix[i * n + j] += firstVal * secondMatrix[k * n + j];
                        }
                    }
                }
            }
        }
    });
}

void matmulBlas(double* firstMatrix, double* secondMatrix, double* resultMatrix, int n) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, firstMatrix, n, secondMatrix, n, 0.0, resultMatrix, n);
}

int main() {
    int n = 2048;
    
    double* firstMatrix = new double[n * n];
    double* secondMatrix = new double[n * n];
    double* resultMatrix = new double[n * n];
    
    generateInput(firstMatrix, secondMatrix, n);
    
    cout << "Matrix size: " << n << " x " << n << endl;
    
    auto startNaive = chrono::high_resolution_clock::now();
    matmulNaive(firstMatrix, secondMatrix, resultMatrix, n);
    auto endNaive = chrono::high_resolution_clock::now();
    double timeNaive = chrono::duration<double, milli>(endNaive - startNaive).count();
    cout << "Naive: " << timeNaive << " ms" << endl;
    
    
    resetResult(resultMatrix, n);
    auto startOpt = chrono::high_resolution_clock::now();
    matmulOptimized(firstMatrix, secondMatrix, resultMatrix, n);
    auto endOpt = chrono::high_resolution_clock::now();
    double timeOpt = chrono::duration<double, milli>(endOpt - startOpt).count();
    cout << "Optimized: " << timeOpt << " ms" << endl;
    
    resetResult(resultMatrix, n);
    auto startBlas = chrono::high_resolution_clock::now();
    matmulBlas(firstMatrix, secondMatrix, resultMatrix, n);
    auto endBlas = chrono::high_resolution_clock::now();
    double timeBlas = chrono::duration<double, milli>(endBlas - startBlas).count();
    cout << "BLAS: " << timeBlas << " ms" << endl;
    
    cout << "Speedup (Naive vs Optimized): " << timeNaive / timeOpt << "x" << endl;
    cout << "Speedup (Naive vs BLAS): " << timeNaive / timeBlas << "x" << endl;
    cout << "Gap (Optimized vs BLAS): " << timeOpt / timeBlas << "x" << endl;
    
    delete[] firstMatrix;
    delete[] secondMatrix;
    delete[] resultMatrix;
    
    return 0;
}
