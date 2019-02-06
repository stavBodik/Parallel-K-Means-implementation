#pragma once

typedef struct POINT_S {
	double x;
	double y;
} point;

typedef struct POINT_CIRCLE_S {
	double radius;
	double centerX;
	double centerY;
} POINT_CIRCLE_S;

typedef struct CLUSTER_S {
	int id;
	POINT_S centroid;
	POINT_S* points;
	int points_size;
} cluster;

typedef struct SYSTEM_SNAPSHOT_S {
	CLUSTER_S* clusters;
	double time;
	double minimumDistanceBetweenClusters;
} system_snapshot;

double GetMinimumDistanceBetweenClusters(CLUSTER_S* clusters, int nClusters);
void K_Means(CLUSTER_S* &clusters, POINT_S* points, int nClusters, int nMaxIterations, int points_size);
void ClearClustersPoints(CLUSTER_S* clusters, int clusters_size);
void CreatePointsMap(POINT_S* points, POINT_CIRCLE_S* circles, int circles_size, double t, double timeIntervalT);
void ReadInputFile(char * fullFilePath, POINT_CIRCLE_S* &circles, int *nPoints, int *nClusters, double *deltaT, double *timeIntervalT, int *nMaxIterations);
void AddPointToCluster(POINT_S p, CLUSTER_S* cluster);
void FreeClusters(CLUSTER_S* clusters, int clusters_size);
int GetMinimumNumberIndexArray(double arr[], int arr_size);
POINT_S GetPointsCentroid(POINT_S* points, int points_size);
void LoadPoints(POINT_S* points, int points_size);
double GenerateRandomDouble(int min, int max);
double GetDistanceBetweenPoints(POINT_S p1, POINT_S p2);
void AssignPointToClosestCluster(POINT_S p, CLUSTER_S* clusters, int clusters_size);
void GeneratePointsFile(char* fileFullPath, int numerOfPoints, double maxRaidus, int boundsWidth, int boundsHeight);
void WriteClustersToFile(char* fileFullPath, CLUSTER_S* clusters, int clusters_size);
void InitClusters(CLUSTER_S* &clusters, int nClusters);
void DeepCopyClusters(CLUSTER_S* s_clusters, CLUSTER_S* d_clusters, int clusters_size);
void WriteOutputFile(char* fileFullPath, SYSTEM_SNAPSHOT_S best_system_snapshoot, int nClusters);