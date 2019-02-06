#include <stdio.h>
#include "prototypes.h"
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// helps to find the size of any array type,this macro is taken from http://www.cplusplus.com/forum/general/33669/
#define ARRAY_SIZE(array) (sizeof((array))/sizeof((array[0])))

#define MASTER_ID 0
#define MATRIX_SIZE 8
#define NUMBER_OF_PROCESSORS 4

int main(int argc, char *argv[])
{

	int **matrix;
	int randomIntegerRangeMin = 0;
	int randomIntegerRangeMax = 100;
	matrix=GenerateRandomMatrixNN(MATRIX_SIZE, randomIntegerRangeMin, randomIntegerRangeMax);
	const clock_t begin_time = clock();
	ShearSort(matrix, MATRIX_SIZE, NUMBER_OF_PROCESSORS);	
	printf("time : %f\n", float(clock() - begin_time) / (CLOCKS_PER_SEC/1000));
	Print2DArray(matrix,MATRIX_SIZE);
	
	return 0;
}

// Sorting given matrix with sharesort algorithem , the output matrix will be sorted in "snake" order.
void ShearSort(int** matrix, int sizeN, int nThreads)
{
	for (int i = 0; i<sizeN + 1; i++) {
		if (i % 2 == 0) {
			ShearSortMatrixHorizontalIteration(matrix, MATRIX_SIZE, nThreads);
		}else 
		{
			ShearSortMatrixVerticallIteration(matrix, MATRIX_SIZE, nThreads);
		}
	}
}

// Performs part of vertical iteration from shearsort algorithem on given matrix.
void ShearSortMatrixVerticallIteration(int** matrix, int sizeN, int nThreads)
{
	#pragma omp parallel for 
	for (int i = 0; i < sizeN; i++){
		int* matrixCol = GetColFromMatrix(matrix, sizeN, i);
		OddEvenParallelSort(matrixCol, true, sizeN, nThreads);
		ReplaceMatrixCol(matrix, sizeN, matrixCol,i);
	}
}

// Performs part of horizontal iteration from shearsort algorithem on given matrix.
void ShearSortMatrixHorizontalIteration(int** matrix, int sizeN, int nThreads)
{
	#pragma omp parallel for num_threads(nThreads)
	for (int i = 0; i < sizeN; i++) {
		if (i % 2 == 0) {
			OddEvenParallelSort(matrix[i],true,sizeN, nThreads);
		}
		else {
			OddEvenParallelSort(matrix[i], false, sizeN, nThreads);
		}
	}
}

// Replace matrix column by given integers array 
void ReplaceMatrixCol(int** matrix, int sizeN,int* arr, int colIndex)
{
	for (int i = 0; i<sizeN; i++) {
		matrix[i][colIndex] = arr[i];
	}
}

// get matrix column by given index
int* GetColFromMatrix(int** matrix, int sizeN, int colIndex)
{
	int *matrixCol = new int[sizeN];
	for (int i = 0; i<sizeN; i++) {
		matrixCol[i] = matrix[i][colIndex];
	}
	return matrixCol;
}

// generates random matrix with size of N*N , where numbers are random generated in range of min and max . 
int** GenerateRandomMatrixNN(int sizeN, int min, int max)
{
	//malloc 2 dimentional array , take from http://stackoverflow.com/ written by Kevin Loney
	int **matrix = new int*[sizeN];
	for (int i = 0; i < sizeN; ++i) {
		matrix[i] = new int[sizeN];
	}

	for (int i = 0; i<sizeN; i++) {
		for (int j = 0; j<sizeN; j++) {
			matrix[i][j] = GenerateRandomInt(min, max);
		}
	}

	return matrix;
}

// paraller sort of integers array by given direction, when leftToRight is true |-> 1,2,3,4 else 4,3,2,1
void OddEvenParallelSort(int arr[], bool isLeftToRight, int arrLength, int nThreads)
{
	for (int i = 0; i<arrLength; i++) {
		int startIndex;
		i % 2 == 0 ? startIndex = 0 : startIndex = 1;

		#pragma omp parallel for num_threads(nThreads)
		for (int j = startIndex; j<arrLength - 1; j += 2) {
			if (isLeftToRight && arr[j]>arr[j + 1]) {
				SwapIntegersInArray(j, j + 1, arr);
			}
			else if (!isLeftToRight && arr[j]<arr[j + 1]) {
				SwapIntegersInArray(j, j + 1, arr);
			}
		}
	}
}

// swaps 2 integers in array 
void SwapIntegersInArray(int index1, int index2, int arr[])
{
	int temp = arr[index1];
	arr[index1] = arr[index2];
	arr[index2] = temp;
}

// init MPI ,load process id and number of processes needed for this program to execute.
void Mpi_InitValues(int argc, char *argv[], int& myid, int& numberOfSlaves)
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numberOfSlaves);
}

//generates random integer in range between min and max, 
int GenerateRandomInt(int min, int max)
{
	return min + (rand() % (int)(max - min + 1));
}

// used for debug propose , prints 2 dimentional array with sizeN to console
void Print2DArray(int** arr, int sizeN)
{
	for (int i = 0; i < sizeN; i++)
	{
		PrintArray(arr[i], sizeN);
	}
	printf("\n");
}

// used for debug propose , prints 1 dimentional array to console
void PrintArray(int arr[], int arrLength) {
	for (int i = 0; i<arrLength; i++) { printf("%d,", arr[i]); }
	printf("\n");
}