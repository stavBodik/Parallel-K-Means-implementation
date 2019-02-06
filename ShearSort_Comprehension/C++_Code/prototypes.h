#pragma once

void Mpi_InitValues(int argc, char *argv[], int& myid, int& numprocs);
void SwapIntegersInArray(int index1, int index2, int arr[]);
void OddEvenParallelSort(int arr[], bool isLeftToRight, int arrLength, int nThreads);
void PrintArray(int arr[], int arrLength);
int GenerateRandomInt(int min, int max);
int GenerateRandomInt(int min, int max);
int** GenerateRandomMatrixNN(int sizeN, int min, int max);
void Print2DArray(int** arr, int sizeN);
int* GetColFromMatrix(int** matrix, int sizeN, int colIndex);
void ReplaceMatrixCol(int** matrix, int sizeN, int* arr, int colIndex);
void ShearSortMatrixHorizontalIteration(int** matrix, int sizeN, int nThreads);
void ShearSortMatrixHorizontalIteration(int** matrix, int sizeN, int nThreads);
void ShearSortMatrixVerticallIteration(int** matrix, int sizeN, int nThreads);
void ShearSort(int** matrix, int sizeN, int nThreads);
