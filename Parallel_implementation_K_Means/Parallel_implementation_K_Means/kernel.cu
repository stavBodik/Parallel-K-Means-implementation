#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "prototypes.h"
#include "math.h"


// correct for nvidia quadro 600 GPU device, not used needed for debug calculations.
//#define MAX_NUMBER_OF_THREADS_IN_BLOCK 1024  
//#define MAX_NUMBER_OF_BLOCKS 65535     


// for given point p and size clusters, calculates p distance to each one of the clusters centroid
// and assign it to the closest cluster.
__device__ void AssignPointToClosestClusterDevice(POINT_S *p, CLUSTER_S* clusters,int clusters_size) 
{
    //calculate distance from point p to each cluster centroid,clusterID is the id of minimum distane found.
	int clusterID=clusters[0].id;
	double minDistance=GetDistanceBetweenPoints(*p, clusters[0].centroid);
	
 
	for (int i = 1; i<clusters_size; i++) {
		double dist = GetDistanceBetweenPoints(*p, clusters[i].centroid);
		if(dist<minDistance){
			minDistance=dist;
			clusterID=clusters[i].id;
		}
	}

	////assign point p to found cluster.
	p->clusterID=clusterID;
}

__device__ double GetDistanceBetweenPoints(POINT_S p1, POINT_S p2)
{
	double dx = p2.x - p1.x;
	double dy = p2.y - p1.y;
	return sqrt(dx*dx + dy*dy);
}

// kernal function where each thread takes range of points from points array , for each point checks the distance to each cluster
// and assign it to closest cluster.
__global__ void AssignRangeOfPointsToClosestClusters(POINT_S *dev_points,int points_size, CLUSTER_S *dev_clusters,int clusters_size,int pointsRangeForThread,int pointsRangeForBlock)
{
    int tID = threadIdx.x;
	int bID = blockIdx.x;
	int startIndexOffset = bID*pointsRangeForBlock + tID*pointsRangeForThread;
	
	// this case cudaOccupancyMaxPotentialBlockSize gave to much threads for job, some of them not needed
	if(startIndexOffset>points_size-1){
		return;
	}
		
	//assign each point to closest cluster
	for(int i=startIndexOffset; i<(startIndexOffset+pointsRangeForThread); i++){
		AssignPointToClosestClusterDevice(&dev_points[i], dev_clusters, clusters_size);
	}

}

// kernal function where each thread take care of single point from points array, checks its distance to clusters
// and assign it to closest cluster.
__global__ void AssignPointsToClosestClusters(POINT_S *dev_points,CLUSTER_S *dev_clusters,int clusters_size,int nThreadsInBlock,int startIndexOffset)
{
    int tID = threadIdx.x;
	int bID = blockIdx.x;
	int pointIndex = startIndexOffset+((bID * nThreadsInBlock)+tID);
	
	AssignPointToClosestClusterDevice(&dev_points[pointIndex], dev_clusters, clusters_size);
	
}

// For given array , returns the index of minimum number in this array.
__device__ int GetMinimumNumberIndexArray1(double arr[], int arr_size)
{
	double min = arr[0];
	int minIndex = 0;
	for (int i = 1; i<arr_size; i++) {
		if (arr[i]<min) {
			min = arr[i];
			minIndex = i;
		}
	}
	return minIndex;
}


// For given array of points and clusters, calculates the distance for each point to each cluster
// and assign it to the closest cluster.
cudaError_t AssignPointsToClosestClustersCuda(POINT_S* &points,int points_size, CLUSTER_S* &clusters, int clusters_size,int nMaxIterations)
{
	// pointers to start of arrays will be used on the GPU device.
	POINT_S *dev_points = 0;
	CLUSTER_S *dev_clusters = 0;

	cudaError_t cudaStatus; // used to check status error code after each communication with GPU device.

	// Calculate number of blocks and number of threads per block will need
	// to achieve maximum occupancy of work.
	// idia taken from stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
	int nBlockes,nThreadsForBlock,minGridSize;
	
	//  calculates number of threads for block size that achieves the maximum multiprocessor-level occupancy.
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &nThreadsForBlock, AssignRangeOfPointsToClosestClusters, 0, points_size); 
	
	// Round up nBlockes to use for runing kernal function
	nBlockes = (points_size + nThreadsForBlock - 1) / nThreadsForBlock; 


	// each thread will make calculation to range of points from points array
	// calculate the length of range which each thread should work on .
	int pointsRangeForThread;
	if(nBlockes*nThreadsForBlock>points_size){
		pointsRangeForThread=1;
	}else{
		pointsRangeForThread = points_size/(nBlockes*nThreadsForBlock);
	}
	 
	
	// calculate the total range size which each block will work on 
	int pointsRangeForBlock = pointsRangeForThread*nThreadsForBlock;

	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf( "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n%s",cudaGetErrorString(cudaStatus));
		FreeDevBuffers(dev_points,dev_clusters);
    }
	
	// Allocate GPU buffers for points and clusters array
	cudaStatus = cudaMalloc((void**)&dev_points, points_size * sizeof(POINT_S));
    if (cudaStatus != cudaSuccess) {
        printf( "cudaMalloc failed! Allocate GPU buffer for points array \n%s",cudaGetErrorString(cudaStatus));
		FreeDevBuffers(dev_points,dev_clusters);
    }

	
	cudaStatus = cudaMalloc((void**)&dev_clusters, clusters_size * sizeof(CLUSTER_S));
    if (cudaStatus != cudaSuccess) {
        printf( "cudaMalloc failed! Allocate GPU buffer for clusters array \n%s",cudaGetErrorString(cudaStatus));
		FreeDevBuffers(dev_points,dev_clusters);
    }


	// Copy points and clusters array to alocated GPU buffers
    cudaStatus = cudaMemcpy(dev_points, points, points_size * sizeof(POINT_S), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf( "cudaMemcpy failed! Copy points alocated GPU buffers\n%s",cudaGetErrorString(cudaStatus));
		FreeDevBuffers(dev_points,dev_clusters);
    }
	

	
	for(int i=0; i<nMaxIterations; i++){

		cudaStatus = cudaMemcpy(dev_clusters, clusters, clusters_size * sizeof(CLUSTER_S), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf( "cudaMemcpy failed! Copy points alocated GPU buffers\n%s",cudaGetErrorString(cudaStatus));
			FreeDevBuffers(dev_points,dev_clusters);
		}

		////run kernal function which will asign each point to closest clusters
		AssignRangeOfPointsToClosestClusters<<<nBlockes,nThreadsForBlock>>>(dev_points,points_size,dev_clusters, clusters_size, pointsRangeForThread , pointsRangeForBlock);
	
		// wait for the kernel to finish.
	   cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf( "cudaDeviceSynchronize failed! AssignRangeOfPointsToClosestClusters\n%s",cudaGetErrorString(cudaStatus));
			FreeDevBuffers(dev_points,dev_clusters);
		}
	
		//// special case where not all points got assign to clusters due to diviation reminder
		//// in this case number of thrads will used is the number of remind points to calculate
		//// where each thread will take care of single point .
		if(points_size % pointsRangeForThread != 0){
			printf("reminder case\n");
			int nRemindPoints = points_size % pointsRangeForThread;
			int startIndexOffset = points_size-nRemindPoints;
		
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &nThreadsForBlock, AssignPointsToClosestClusters, 0, nRemindPoints); 

			nBlockes = (nRemindPoints + nThreadsForBlock - 1) / nThreadsForBlock; 
		
			AssignPointsToClosestClusters<<<nBlockes,nThreadsForBlock>>>(dev_points,dev_clusters, clusters_size, nThreadsForBlock , startIndexOffset);
				
			// wait for the kernel to finish.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				printf( "cudaDeviceSynchronize failed! AssignRangeOfPointsToClosestClusters\n%s",cudaGetErrorString(cudaStatus));
				FreeDevBuffers(dev_points,dev_clusters);
			}
		}
	

		// Copy results of sorted points per clusters
		cudaStatus = cudaMemcpy(points, dev_points, points_size * sizeof(POINT_S), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			printf( "Copy results of found clusters from device to host failed!\n%s",cudaGetErrorString(cudaStatus));
			FreeDevBuffers(dev_points,dev_clusters);
		}
		
		
		RecalculateClusterCentroids(clusters,clusters_size,points,points_size);

		// stop K Means when all clusters centeroids stays the same
		if(!IsClustersCentroidsHasChanged(clusters,clusters_size)){
			break;
		}

	}


	FreeDevBuffers(dev_points,dev_clusters);

    return cudaStatus;
}

void FreeDevBuffers(POINT_S *dev_points,CLUSTER_S *dev_clusters){

	cudaFree(dev_points);
	cudaFree(dev_clusters);
}

