#pragma once
#include <mpi.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct TIME_INFO_S {
   double tStart;
   double tEnd;
   double deltaT;
   double timeInterval;
   int nMaxIteration;
   int nClusters;
   int nCircles;

} timeInfo;

typedef struct POINT_S {
	double x;
	double y;
	int clusterID;
} point;

typedef struct CIRCLE_S {
	double radius;
	double centerX;
	double centerY;
} CIRCLE_S;

typedef struct CLUSTER_S {
	int id;
	int points_size;
	POINT_S centroid;
	POINT_S oldCentroid;
} cluster;


typedef struct SYSTEM_SNAPSHOT_S {
	double time;
	double minimumDistanceBetweenClusters;
	CLUSTER_S* clusters;
} system_snapshot;





//MPI,OpenMP
MPI_Datatype CreateMPICircleType();
MPI_Datatype CreateMPIPointType();
MPI_Datatype CreateMPISnapShotType();
MPI_Datatype CreateMPIRangeType();
MPI_Datatype CreateMPIClusterType(MPI_Datatype PointMPIType);
int GetBestSnapShotIndex(SYSTEM_SNAPSHOT_S* snapshots, int snapshots_size);
void CreateTimeInformations(TIME_INFO_S* time_informations,double timeIntervalT,int nProcesses,double deltaT,int nMaxIterations,int nCircles,int nClusters);
double GetMinimumDistanceBetweenClusters(CLUSTER_S* clusters, int nClusters);
void K_Means(CLUSTER_S* & clusters,int nClusters,POINT_S *points,int nPoints,double systemAngel,int nMaxIterations,double timeT);
void CreatePointsMap(POINT_S* points, CIRCLE_S* circles, int circles_size, double angel);
void ReadInputFile(char * fullFilePath, CIRCLE_S* &circles, int *nCircles, int *nClusters, double *deltaT, double *timeIntervalT, int *nMaxIterations);
void AddPointToCluster(POINT_S p, CLUSTER_S* cluster);
void FreeClusters(CLUSTER_S* clusters, int clusters_size);
int GetMinimumNumberIndexArray(double arr[], int arr_size);
void LoadPoints(POINT_S* points, int points_size);
double GenerateRandomDouble(int min, int max);
double GetDistanceBetweenCircles(POINT_S p1, POINT_S p2);
void GeneratePointsFile(char* fileFullPath, int numerOfPoints, double maxRaidus, int boundsWidth, int boundsHeight);
void WritePointsToFile(char* fileFullPath, POINT_S* points, int nPoints);
void InitClusters(CLUSTER_S* &clusters,int nClusters,POINT_S* points);
void WriteOutputFile(char* fileFullPath, SYSTEM_SNAPSHOT_S best_system_snapshoot, int nClusters);
void InitiateMPI(int argc, char *argv[], int& processID, int& nProcesses);
void RecalculateClusterCentroids(CLUSTER_S* &clusters,int clusters_size,POINT_S* points,int points_size);
bool IsClustersCentroidsHasChanged(CLUSTER_S* &clusters,int clusters_size);
void mallocCUDA(int psize,int csize);
void RestClusters(CLUSTER_S* &clusters,int nClusters);

//CUDA
cudaError_t AssignPointsToClosestClustersCuda(POINT_S* &points,int points_size, CLUSTER_S* &clusters, int clusters_size,int nMaxIterations);
void FreeDevBuffers(POINT_S *dev_points,CLUSTER_S *dev_clusters);
__device__ void AssignPointToClosestClusterDevice(POINT_S p, CLUSTER_S* clusters, int clusters_size);
__device__ void AddPointToClusterDevice(POINT_S p, CLUSTER_S* cluster);
__device__ double GetDistanceBetweenPoints(POINT_S p1, POINT_S p2);
__device__ int GetMinimumNumberIndexArray1(double arr[], int arr_size);
__device__ POINT_S* myrealloc(int oldsize, int newsize, POINT_S* old);