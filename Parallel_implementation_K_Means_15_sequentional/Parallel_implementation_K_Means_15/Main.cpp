#include <stdlib.h>
#include <stdio.h>
#include "prototypes.h"
#include "math.h"
#include <string.h>
#include <omp.h>
#include <time.h>

char*   inputFilePath = "circles_100_25.txt";
char*   outputFilePath = "output.txt";
const   double PI = 4 * atan(1.0);
#define MAX_DISTANCE_BETWEEN_CENTROIDS 800;

int main(int argc, char *argv[])
{
	double deltaT = 0, timeIntervalT = 0;
	int nPoints = 0, nClusters = 0, nMaxIterations = 0;

	POINT_CIRCLE_S* circles=NULL;
	ReadInputFile(inputFilePath,circles,&nPoints,&nClusters,&deltaT,&timeIntervalT,&nMaxIterations);


	// look for best system snapshoot with minimum distance between clusters with given time interval.
	SYSTEM_SNAPSHOT_S best_system_snapshoot;
	best_system_snapshoot.minimumDistanceBetweenClusters = MAX_DISTANCE_BETWEEN_CENTROIDS;
	best_system_snapshoot.clusters = (CLUSTER_S*)malloc(nClusters * sizeof(CLUSTER_S));
	 
	const clock_t begin_time = clock();
	for(double t=0; t<timeIntervalT; t+=deltaT){
		
		printf("Iteration at time : %lf\n",t);

		// create map of points relative to current time t .
		POINT_S* points = (POINT_S*)malloc(nPoints * sizeof(POINT_S));
		CreatePointsMap(points,circles,nPoints, t,timeIntervalT);
		
		// find clusters using K Means algorithem .
		CLUSTER_S* clusters = NULL;
		K_Means(clusters,points, nClusters, nMaxIterations, nPoints);

		// found minimum distance for this snapshoot
		double minimumDistanceBetweenClusters = GetMinimumDistanceBetweenClusters(clusters, nClusters);

		// check is this snapshoot has better result from the others, if it has save it.
		if (minimumDistanceBetweenClusters < best_system_snapshoot.minimumDistanceBetweenClusters){
			DeepCopyClusters(clusters, best_system_snapshoot.clusters,nClusters);
			best_system_snapshoot.minimumDistanceBetweenClusters = minimumDistanceBetweenClusters;
			best_system_snapshoot.time = t;
		}

		free(points);
		FreeClusters(clusters, nClusters);
	}
	printf("time : %f\n", float(clock() - begin_time) / (CLOCKS_PER_SEC / 1000));

	WriteOutputFile(outputFilePath,best_system_snapshoot,nClusters);
	WriteClustersToFile("C:\\studies\\mpimpi\\final_project\\Parallel_implementation_K_Means_15\\clusters_100.txt", best_system_snapshoot.clusters, nClusters);
	free(circles);
	

	return 0;
}

void WriteOutputFile(char* fileFullPath,SYSTEM_SNAPSHOT_S best_system_snapshoot,int nClusters)
{
	FILE *f;
	errno_t errorCode = fopen_s(&f, fileFullPath, "w");

	if (errorCode != 0)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	
	fprintf(f, "Minimum distance : %lf\n\n", best_system_snapshoot.minimumDistanceBetweenClusters);
	fprintf(f, "Occurred at time : %lf\n\n", best_system_snapshoot.time);
	fprintf(f, "Centers of the clusters:\n\n", best_system_snapshoot.time);
	
	for (int i = 0; i < nClusters; i++) {
		fprintf(f, "%lf\n\n", best_system_snapshoot.clusters[i].centroid);
	}
	
	fclose(f);
}

void DeepCopyClusters(CLUSTER_S* s_clusters, CLUSTER_S* d_clusters,int clusters_size) 
{
	memcpy(d_clusters, s_clusters, clusters_size * sizeof(CLUSTER_S));
	for (int i = 0; i < clusters_size; i++) {
		d_clusters[i].points = (POINT_S*)malloc(s_clusters[i].points_size * sizeof(POINT_S));
		memcpy(d_clusters[i].points, s_clusters[i].points, s_clusters[i].points_size * sizeof(POINT_S));
	}
}

double GetMinimumDistanceBetweenClusters(CLUSTER_S* clusters,int nClusters)
{
	int minDist = GetDistanceBetweenPoints(clusters[0].centroid, clusters[1].centroid);
	#pragma omp parallel for
	for (int i = 0; i < nClusters; i++) {
		#pragma omp parallel for
		for (int j = i+1; j < nClusters; j++) {
			POINT_S p1 = clusters[i].centroid;
			POINT_S p2 = clusters[j].centroid;
			double distanceBetweenPoints = GetDistanceBetweenPoints(p1, p2);
			if (GetDistanceBetweenPoints(p1, p2) < minDist) {
				minDist = distanceBetweenPoints;
			}
		}
	}

	return minDist;
}
// for given array of points perform K-Means algorithem and returns found clusters.
void K_Means(CLUSTER_S* &clusters,POINT_S* points,int nClusters,int nMaxIterations,int points_size)
{
	InitClusters(clusters, nClusters);

	for (int i = 0; i<nMaxIterations; i++) {

		ClearClustersPoints(clusters, nClusters);

		for (int j = 0; j<points_size; j++) {
			AssignPointToClosestCluster(points[j], clusters, nClusters);
		}

		
		for (int j = 0; j<nClusters; j++) {
			if (clusters[j].points_size>0) {
				clusters[j].centroid = GetPointsCentroid(clusters[j].points, clusters[j].points_size);
			}
		}
	}
	
}

// initiate empty clusters with arbitrary centroid points.
void InitClusters(CLUSTER_S* &clusters,int nClusters)
{
	clusters = (CLUSTER_S*)malloc(nClusters * sizeof(CLUSTER_S));

	for (int i = 0; i < nClusters; i++) {
		clusters[i].points = (POINT_S*)malloc(sizeof(POINT_S));
		clusters[i].centroid.x = i;
		clusters[i].centroid.y = i;
		clusters[i].points_size = 0;
		clusters[i].id = i;
	}
}

// clears points from cluster
void ClearClustersPoints(CLUSTER_S* clusters,int clusters_size)
{
	for (int i = 0; i<clusters_size; i++) {
		clusters[i].points_size = 0;
		free(clusters[i].points);
		clusters[i].points = (POINT_S*)malloc(sizeof(POINT_S));
	}
}

// create map of points in given time t relative to circle centers positions.
void CreatePointsMap(POINT_S* points, POINT_CIRCLE_S* circles, int circles_size, double t, double timeIntervalT)
{
	#pragma omp parallel for
	for (int i = 0; i<circles_size; i++) {
		points[i].x = circles[i].centerX + circles[i].radius*cos((2 * PI*t) / timeIntervalT);
		points[i].y = circles[i].centerY + circles[i].radius*sin((2 * PI*t) / timeIntervalT);
	}
}

// free memory alocated for clusters array
void FreeClusters(CLUSTER_S* clusters, int clusters_size)
{
	for (int i = 0; i<clusters_size; i++) {
		free(clusters[i].points);
	}

	free(clusters);
}

void LoadPoints(POINT_S* points, int points_size)
{
	for (int i = 0; i < points_size; i++)
	{
		points[i].x = GenerateRandomDouble(0,10);
		points[i].y = GenerateRandomDouble(0, 10);
	}
}

//adds point p to cluster existing points.
void AddPointToCluster(POINT_S p, CLUSTER_S* cluster)
{
	//realloc existing points to new points with sie +1
	cluster->points = (POINT_S*)realloc(cluster->points, (cluster->points_size + 1) * sizeof(POINT_S));

	//copy information of new point
	cluster->points[cluster->points_size].x = p.x;
	cluster->points[cluster->points_size].y = p.y;

	cluster->points_size += 1;
}

// for given array arr , returns the index of minimum number of this array.
int GetMinimumNumberIndexArray(double arr[], int arr_size)
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

POINT_S GetPointsCentroid(POINT_S* points, int points_size)
{
	POINT_S result;

	double xSum = 0;
	double ySum = 0;

	for (int i = 0; i<points_size; i++) {
		xSum += points[i].x;
		ySum += points[i].y;
	}
	result.x = xSum / points_size;
	result.y = ySum / points_size;

	return result;
}

// for given point p and size clusters, calculates p distance to each one of the clusters centroid
// and assign it to the closest cluster.
void AssignPointToClosestCluster(POINT_S p, CLUSTER_S* clusters, int clusters_size) {

	//calculate distance from point p to each cluster centroid
	double* distances = (double*)malloc(clusters_size * sizeof(double));
	
 
	for (int i = 0; i<clusters_size; i++) {
		distances[i] = GetDistanceBetweenPoints(p, clusters[i].centroid);
	}

	//find the index of closest cluster
	int assingToClusterIndex = GetMinimumNumberIndexArray(distances, clusters_size);

	//assign point p to found cluster.
	AddPointToCluster(p, &clusters[assingToClusterIndex]);

	free(distances);
}

double GetDistanceBetweenPoints(POINT_S p1, POINT_S p2)
{
	double dx = p2.x - p1.x;
	double dy = p2.y - p1.y;
	return sqrt(dx*dx + dy*dy);
}

// loads circles of points,number of clusters to find, time increment deltaT,time interval T, and maximum number of iterations LIMIT. 
void ReadInputFile(char * fullFilePath, POINT_CIRCLE_S* &circles, int *nPoints, int *nClusters, double *deltaT, double *timeIntervalT, int *nMaxIterations)
{
	FILE *f;
		
	errno_t errorCode = fopen_s(&f, fullFilePath, "r");
	
	if (errorCode != 0)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	int row = fscanf_s(f, "%d %d %lf %lf %d", nPoints, nClusters, deltaT, timeIntervalT, nMaxIterations);
	circles = (POINT_CIRCLE_S*)malloc(*nPoints * sizeof(POINT_CIRCLE_S));

	int pointId;
	double circleCenterX, circleCenterY, radius;
	while (row != EOF) {
		row = fscanf_s(f, "%d %lf %lf %lf\n", &pointId, &circleCenterX, &circleCenterY, &radius);
		circles[pointId].radius = radius;
		circles[pointId].centerX = circleCenterX;
		circles[pointId].centerY = circleCenterY;
	}

	fclose(f);
}

void WriteClustersToFile(char* fileFullPath, CLUSTER_S* clusters,int clusters_size)
{

	FILE *f;
	errno_t errorCode = fopen_s(&f, fileFullPath, "w");

	if (errorCode != 0)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	for (int i = 0; i<clusters_size; i++) {
		for (int j = 0; j < clusters[i].points_size; j++) {
			fprintf(f, "%d %lf %lf\n", clusters[i].id, clusters[i].points[j].x, clusters[i].points[j].y);
		}
	}

	fclose(f);
}

void GeneratePointsFile(char* fileFullPath, int numerOfPoints, double maxRaidus, int boundsWidth, int boundsHeight)
{
	FILE *f;
	errno_t errorCode = fopen_s(&f, fileFullPath, "w");

	if (errorCode != 0)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	for (int i = 0; i<numerOfPoints; i++) {
		double circleCenterX = GenerateRandomDouble(0, boundsWidth);
		double circleCenterY = GenerateRandomDouble(0, boundsHeight);
		double radius = GenerateRandomDouble(0, maxRaidus);
		fprintf(f, "%d %lf %lf %lf\n", i, circleCenterX, circleCenterY, radius);
	}

	fclose(f);
}

//generates random double in range between min and max, 
double GenerateRandomDouble(int min, int max)
{
	double f = (double)rand() / RAND_MAX;
	return min + f * (max - min);
}