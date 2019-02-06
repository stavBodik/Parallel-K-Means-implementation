// State of work : 
// The goal is to find the minimum distance between clusters occurred in all t times of system work.

// Master process performs K Means algorithem with help of slave process using CUDA	and OpenMP.
		
// Master initial send to each slave his interval T from total timeIntervalT for performing K Means algorithem (stored in TIME_INFO_S).

// Eech Slave performs K Means algorithem T times in parrallel .
// Eeach T time calculated by seperate thread using OpenMP and CUDA.
// When all threads finished work, minimum distance between clusters and time occurnece 
// is saved inside SYSTEM_SNAPSHOT_S struct and then sent back to Master process

// Master final work is to find best system snapshot between all system snapshots found by slaves . 



#include <stdlib.h>
#include <stdio.h>
#include "prototypes.h"
#include "math.h"
#include <string.h>
#include <omp.h>
#include <time.h>
#include <mpi.h>
                         
char*   inputFilePath;
char*   outputFilePath = "C:\\PK\\output.txt";
const   double PI = 4 * atan(1.0);
#define MASTER_PROCESS_ID 0
#define MAX_DISTANCE_BETWEEN_CENTROIDS 800;
#define STOP_SLAVE_WORK_FLAG 1


// optional, if true writes best system snapshoot points to file.
const bool isWritePointsToFile = true;


int main(int argc, char *argv[])
{
	MPI_Status status;
	int nProcesses, processID;
	double startTime, finishTime;

	// initiate MPI ,load this process id and number of processes needed for this program to execute.
	InitiateMPI(argc, argv, processID, nProcesses);

	// Create and commit POINT_S,CIRCLE_S struct types in MPI system.
	MPI_Datatype CircleMPIType = CreateMPICircleType();
	MPI_Datatype RangeMPIType =  CreateMPIRangeType();
	MPI_Datatype PointMPIType = CreateMPIPointType();
	MPI_Datatype ClusterMPIType =  CreateMPIClusterType(PointMPIType);
	MPI_Datatype SnapShotMPIType =  CreateMPISnapShotType();

	double deltaT = 0, timeIntervalT = 0;
	int nCircles = 0, nClusters = 0, nMaxIterations = 0;
	bool isMasterFoundMinimum=false;

	double global_minimum_distance;
	TIME_INFO_S slaveTimeInfo;
	CIRCLE_S* circles=NULL;
	TIME_INFO_S* time_informations = (TIME_INFO_S*)malloc((nProcesses) * sizeof(TIME_INFO_S));
	

	//---------------------------------Initial Master work------------------------------------------------------
	// Read input file includes :  
	// circles (data for creating 2D points (centerX,centerY,radiuis))
	// nCircles (number of points in system to create)
	// nClusters (number of clusters to find)
	// timeIntervalT (total time system work)
	// deltaT (delta time to check system within timeIntervalT)
	// nMaxIterations (max of iterations for K Means algorithem limit)
		
	if (processID == MASTER_PROCESS_ID) {
		// read input file , argv[1] holds file path.
		inputFilePath = argv[1];
		ReadInputFile(inputFilePath,circles,&nCircles,&nClusters,&deltaT,&timeIntervalT,&nMaxIterations);
		// Creates Time information array which is used by each slave for calculating K Means.
		CreateTimeInformations(time_informations,timeIntervalT,nProcesses,deltaT,nMaxIterations,nCircles,nClusters);
	}

	// record work start time
	startTime = MPI_Wtime();

	// send to each slave his time information struct (contains input file data)
	MPI_Scatter(time_informations,1, RangeMPIType,&slaveTimeInfo,1, RangeMPIType, MASTER_PROCESS_ID, MPI_COMM_WORLD);
	// send to each slave circles from input file (in case this is not master malloc new circles array)
	if(processID!=MASTER_PROCESS_ID){circles = (CIRCLE_S*)malloc(slaveTimeInfo.nCircles * sizeof(CIRCLE_S));}
	MPI_Bcast(circles,slaveTimeInfo.nCircles,CircleMPIType,MASTER_PROCESS_ID,MPI_COMM_WORLD);
		
	//---------------------------------Initial Master work Finish------------------------------------------------------

	//---------------------------------Slave work------------------------------------------------------

	//perform K Means algorithem  for time range given to this slave.
	// system snapshoot saved for each t time .
	int number_snapshoots= (slaveTimeInfo.tEnd-slaveTimeInfo.tStart)/slaveTimeInfo.deltaT;
	SYSTEM_SNAPSHOT_S* systemSnapShots = (SYSTEM_SNAPSHOT_S*)malloc(number_snapshoots*sizeof(SYSTEM_SNAPSHOT_S));
	for(int i=0; i<number_snapshoots; i++){systemSnapShots[i].clusters = (CLUSTER_S*)malloc(slaveTimeInfo.nClusters * sizeof(CLUSTER_S));}
	
		
	#pragma omp parallel for
	for (int tIterarionIndex = 0; tIterarionIndex < number_snapshoots; tIterarionIndex ++) {
		
			//current t for check	
			double current_t = slaveTimeInfo.tStart+tIterarionIndex*slaveTimeInfo.deltaT;

			// current system angel
			double systemAngel = (2 * PI * current_t / slaveTimeInfo.timeInterval);

			//Create points map relative to current system angel
			POINT_S* points_slave = (POINT_S*)malloc(slaveTimeInfo.nCircles * sizeof(POINT_S));
			CreatePointsMap(points_slave,circles,slaveTimeInfo.nCircles, systemAngel);
				
			//init clusters
			CLUSTER_S* clusters = NULL;
			InitClusters(clusters, slaveTimeInfo.nClusters,points_slave);

			double singleTstartTime;
			if(processID==MASTER_PROCESS_ID && tIterarionIndex==0){
				 singleTstartTime= MPI_Wtime();
			}

			// perform KMean algorithem with limit of nMaxIterations using CUDA.
			K_Means(clusters,slaveTimeInfo.nClusters,points_slave,slaveTimeInfo.nCircles,systemAngel,slaveTimeInfo.nMaxIteration,current_t);
			

			if(processID==MASTER_PROCESS_ID && tIterarionIndex==0){
				 printf("single t time : %lf\n",MPI_Wtime()-singleTstartTime);
			}

			//save minimum distance for this snapshoot
			double minimumDistance = GetMinimumDistanceBetweenClusters(clusters,slaveTimeInfo.nClusters);
				
			systemSnapShots[tIterarionIndex].minimumDistanceBetweenClusters = minimumDistance;
			systemSnapShots[tIterarionIndex].time=current_t;
			memcpy(systemSnapShots[tIterarionIndex].clusters, clusters, slaveTimeInfo.nClusters * sizeof(CLUSTER_S));
				
			

			free(points_slave);
			free(clusters);
	}


	//find best snapshot which is the one with minimum distance between clusters for this slave
	SYSTEM_SNAPSHOT_S best_snapshoot;
	int bestSnapshootIndex = GetBestSnapShotIndex(systemSnapShots,number_snapshoots);

	// each slave notify all slaves with his minimum distance found using MPI_Allreduce which also finally saves the minimum in global_minimum_distance
	MPI_Allreduce(&systemSnapShots[bestSnapshootIndex].minimumDistanceBetweenClusters, &global_minimum_distance, 1, MPI_DOUBLE, MPI_MIN,MPI_COMM_WORLD);

	// each slave checks if his minimum is best minimum of all minimum's ^^
	if(global_minimum_distance==systemSnapShots[bestSnapshootIndex].minimumDistanceBetweenClusters){
		
		// if minimum found by none master process , send best snapshoot to master
		if(processID!=MASTER_PROCESS_ID){
			MPI_Send(&systemSnapShots[bestSnapshootIndex], 1,SnapShotMPIType, MASTER_PROCESS_ID, 0, MPI_COMM_WORLD);
			MPI_Send(systemSnapShots[bestSnapshootIndex].clusters,slaveTimeInfo.nClusters*sizeof(CLUSTER_S),MPI_BYTE, MASTER_PROCESS_ID, 0, MPI_COMM_WORLD);
		}else{
			isMasterFoundMinimum=true;
		}

	}


	//---------------------------------Slave work Finish------------------------------------------------------


	//---------------------------------Final Master work------------------------------------------------------

	// master recives from slave which found the best minimum, best system snapshoot and write it to file
	if(processID==MASTER_PROCESS_ID){
		SYSTEM_SNAPSHOT_S best_system_snapshoot;
		best_system_snapshoot.clusters = (CLUSTER_S*)malloc(nClusters * sizeof(CLUSTER_S));
		
		if(!isMasterFoundMinimum){
			MPI_Recv(&best_system_snapshoot,1, SnapShotMPIType, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(best_system_snapshoot.clusters,slaveTimeInfo.nClusters*sizeof(CLUSTER_S), MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		}else{
			best_system_snapshoot=systemSnapShots[bestSnapshootIndex];
		}

		finishTime = MPI_Wtime();
		printf("total time : %lf\n  ",finishTime-startTime);
		WriteOutputFile(outputFilePath,best_system_snapshoot,nClusters);


		// used for debug only , writes best snapshoot points to file for viewing later on graphic application.
		if(isWritePointsToFile){

			//current t for check	
			double current_t = best_system_snapshoot.time;

			// current system angel
			double systemAngel = (2 * PI * current_t / slaveTimeInfo.timeInterval);

			//Create points map relative to current system angel
			POINT_S* points_slave = (POINT_S*)malloc(slaveTimeInfo.nCircles * sizeof(POINT_S));
			CreatePointsMap(points_slave,circles,slaveTimeInfo.nCircles, systemAngel);
				
			//init clusters
			CLUSTER_S* clusters = NULL;
			InitClusters(clusters, nClusters,points_slave);

		    systemAngel = (2 * PI * best_system_snapshoot.time / timeIntervalT);

			POINT_S* points = (POINT_S*)malloc(nCircles * sizeof(POINT_S));
			CreatePointsMap(points,circles,nCircles, systemAngel);

			// perform KMean algorithem with limit of nMaxIterations using CUDA.
			K_Means(clusters,nClusters,points,nCircles,systemAngel,nMaxIterations,best_system_snapshoot.time);
			WritePointsToFile("C:\\PK\\clusters_best.txt",points,nCircles);
		}

	}

	//---------------------------------Final Master work Finish------------------------------------------------------

	//final slaves work, each slave clear his snapshots memory
	for(int i=0; i<number_snapshoots; i++){free(systemSnapShots[i].clusters);}

	MPI_Finalize();

	return 0;
}


// Creates Time information array which is used by each slave for calculating K Means.
void CreateTimeInformations(TIME_INFO_S* time_informations,double timeIntervalT,int nProcesses,double deltaT,int nMaxIterations,int nCircles,int nClusters)
{
	
	int totalNumberOfSystemSnapShots = timeIntervalT/deltaT; 
	int nSystemSnapShootsForProcess = totalNumberOfSystemSnapShots/nProcesses; 
	int reminder = totalNumberOfSystemSnapShots % nProcesses;
	double tRangeForProcess=nSystemSnapShootsForProcess*deltaT; 


	int nSplites=nProcesses;

	if (reminder != 0) {
		nSplites-=1;
	}

	double startT=0;

	for(int i=0; i<nSplites; i++){
		time_informations[i].tStart=startT;
		time_informations[i].tEnd=startT+tRangeForProcess;

		time_informations[i].deltaT=deltaT;
		time_informations[i].timeInterval=timeIntervalT;
		time_informations[i].nMaxIteration=nMaxIterations;
		time_informations[i].nCircles=nCircles;
		time_informations[i].nClusters=nClusters;

		startT+=tRangeForProcess;
	}

	if (reminder != 0) {
		time_informations[nSplites].tStart=startT;
		time_informations[nSplites].tEnd=startT+(reminder+nSystemSnapShootsForProcess)*deltaT;
		time_informations[nSplites].deltaT=deltaT;
		time_informations[nSplites].timeInterval=timeIntervalT;
		time_informations[nSplites].nMaxIteration=nMaxIterations;
		time_informations[nSplites].nCircles=nCircles;
		time_informations[nSplites].nClusters=nClusters;
	}


}


MPI_Datatype CreateMPIPointType()
{
	MPI_Datatype PointType;
	POINT_S point;
	MPI_Datatype type[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
	int blocklen[3] = {1,1,1};
	MPI_Aint disp[3];

	// tell to MPI about offsets of each type inside the struct from begining of struct adresss
	disp[0] = (char *)&point.x - (char *)&point;
	disp[1] = (char *)&point.y - (char *)&point;
	disp[2] = (char *)&point.clusterID - (char *)&point;


	// Create MPI struct and commit.
	MPI_Type_create_struct(3, blocklen, disp, type, &PointType);
	MPI_Type_commit(&PointType);
	return PointType;
}

// Create and commit type SYSTEM_SNAPSHOT_S to MPI for sending the struct of SYSTEM_SNAPSHOT_S type.
MPI_Datatype CreateMPISnapShotType()
{
	MPI_Datatype SnapShotMPIType;
	SYSTEM_SNAPSHOT_S snapShoot;
	MPI_Datatype type[2] = { MPI_DOUBLE, MPI_DOUBLE};
	int blocklen[2] = {1,1};
	MPI_Aint disp[2];

	// tell to MPI about offsets of each type inside the struct from begining of struct adresss
	disp[0] = (char *) &snapShoot.minimumDistanceBetweenClusters -	 (char *) &snapShoot;
	disp[1] = (char *) &snapShoot.time -	 (char *) &snapShoot;
	//disp[2] = (char *) &snapShoot.clusters -	 (char *) &snapShoot;
	
	// Create MPI user data type for partical
    MPI_Type_create_struct(2, blocklen, disp, type, &SnapShotMPIType);
    MPI_Type_commit(&SnapShotMPIType);
	return SnapShotMPIType;
}

// Create and commit type CLUSTER_S to MPI for sending the struct of CLUSTER_S type.
MPI_Datatype CreateMPIClusterType(MPI_Datatype PointMPIType)
{
	MPI_Datatype ClusterMPIType;
	CLUSTER_S cluster;
	MPI_Datatype type[4] = { MPI_INT, MPI_INT,PointMPIType,PointMPIType};
	int blocklen[4] = {1,1,2,2};
	MPI_Aint disp[4];

	// tell to MPI about offsets of each type inside the struct from begining of struct adresss
	disp[0] = (char *) &cluster.id -	 (char *) &cluster;
	disp[1] = (char *) &cluster.points_size -	 (char *) &cluster;
	disp[2] = (char *) &cluster.centroid -	 (char *) &cluster;
	disp[3] = (char *) &cluster.oldCentroid -	 (char *) &cluster;

	// Create MPI user data type for partical
    MPI_Type_create_struct(4, blocklen, disp, type, &ClusterMPIType);
    MPI_Type_commit(&ClusterMPIType);
	return ClusterMPIType;
}

// Create and commit type TIME_INFO_S to MPI for sending the struct of TIME_INFO_S type.
MPI_Datatype CreateMPIRangeType()
{
	MPI_Datatype RangeMPIType;
	TIME_INFO_S range;
	MPI_Datatype type[7] = { MPI_DOUBLE, MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_INT,MPI_INT,MPI_INT};
	int blocklen[7] = {1,1,1,1,1,1,1};
	MPI_Aint disp[7];

	// tell to MPI about offsets of each type inside the struct from begining of struct adresss
	disp[0] = (char *) &range.tStart -	 (char *) &range;
    disp[1] = (char *) &range.tEnd -	 (char *) &range;
	disp[2] = (char *) &range.deltaT -	 (char *) &range;
	disp[3] = (char *) &range.timeInterval -	 (char *) &range;
	disp[4] = (char *) &range.nMaxIteration -	 (char *) &range;
	disp[5] = (char *) &range.nClusters -	 (char *) &range;
	disp[6] = (char *) &range.nCircles -	 (char *) &range;

	// Create MPI user data type for partical
    MPI_Type_create_struct(7, blocklen, disp, type, &RangeMPIType);
    MPI_Type_commit(&RangeMPIType);
	return RangeMPIType;
}

// each K-Means iteration recalculate new clusters centroids.
void RecalculateClusterCentroids(CLUSTER_S* &clusters,int clusters_size,POINT_S* points,int points_size)
{
	// arr -> [sumX,sumY,pointsCount] * number of clusters 
	double* arr = (double*)calloc(clusters_size*3,sizeof(double));
	
	for(int i=0; i<points_size; i++){
		arr[points[i].clusterID*3]+=points[i].x;
		arr[points[i].clusterID*3+1]+=points[i].y;
		arr[points[i].clusterID*3+2]+=1;
	}
 
   // #pragma parallel for
	for(int i=0; i<clusters_size; i++){
		if(arr[i*3+2]!=0){

			double newCentroidX = arr[i*3]/arr[i*3+2];
			double newCentroidY = arr[i*3+1]/arr[i*3+2];

			clusters[i].centroid.x=newCentroidX;
			clusters[i].centroid.y=newCentroidY;
		}
	}

	free(arr);

}

// each K Means iteration checks whether any of clusters centroids has changed
bool IsClustersCentroidsHasChanged(CLUSTER_S* &clusters,int clusters_size)
{
	bool isClustersHasChanged=false;
	
	for(int i=0; i<clusters_size; i++){
		
		double newCentroidX = clusters[i].centroid.x;
		double newCentroidY = clusters[i].centroid.y;
		
		if(newCentroidX != clusters[i].oldCentroid.x || newCentroidY!=clusters[i].oldCentroid.y){
			isClustersHasChanged=true;
		}
		
		//remmber this new centroids for compare with next iteration
		clusters[i].oldCentroid.x=newCentroidX;
		clusters[i].oldCentroid.y=newCentroidY;
	}
		
	
	return isClustersHasChanged;
	
}

// init MPI ,load process id and number of processes needed for this program to execute.
void InitiateMPI(int argc, char *argv[], int& processID, int& nProcesses)
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &processID);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);
}

// Create and commit type CIRCLE_S in MPI system.
MPI_Datatype CreateMPICircleType()
{
	MPI_Datatype CircleType;
	CIRCLE_S circle;
	MPI_Datatype type[3] = { MPI_DOUBLE, MPI_DOUBLE ,MPI_DOUBLE };
	int blocklen[3] = { 1, 1, 1 };
	MPI_Aint disp[3];

	// tell to MPI about offsets of each type inside the struct from begining of struct adresss
	disp[0] = (char *)&circle.centerX - (char *)&circle;
	disp[1] = (char *)&circle.centerY - (char *)&circle;
	disp[2] = (char *)&circle.radius - (char *)&circle;
	
	// Create MPI struct and commit.
	MPI_Type_create_struct(3, blocklen, disp, type, &CircleType);
	MPI_Type_commit(&CircleType);
	return CircleType;
}

void WriteOutputFile(char* fileFullPath,SYSTEM_SNAPSHOT_S best_system_snapshoot,int nClusters)
{
	FILE *f;
	errno_t errorCode = fopen_s(&f, fileFullPath, "w");

	if (errorCode != 0)
	{
		printf("Error writing file!\n");
		exit(1);
	}

	
	fprintf(f, "Minimum distance : %lf\n\n", best_system_snapshoot.minimumDistanceBetweenClusters);
	fprintf(f, "Occurred at time : %lf\n\n", best_system_snapshoot.time);
	fprintf(f, "Centers of the clusters:\n\n", best_system_snapshoot.time);
	
	for (int i = 0; i < nClusters; i++) {
		fprintf(f, "%lf,%lf\n\n", best_system_snapshoot.clusters[i].centroid.x, best_system_snapshoot.clusters[i].centroid.y);
	}
	
	fclose(f);
}

double GetMinimumDistanceBetweenClusters(CLUSTER_S* clusters,int nClusters)
{
	double minDist = GetDistanceBetweenCircles(clusters[0].centroid, clusters[1].centroid);
	
	#pragma omp parallel for 
	for (int i = 0; i < nClusters; i++) {
		#pragma omp parallel for
		for (int j = i+1; j < nClusters; j++) {
			POINT_S p1 = clusters[i].centroid;
			POINT_S p2 = clusters[j].centroid;
			double distanceBetweenCircles = GetDistanceBetweenCircles(p1, p2);
			
			if (distanceBetweenCircles < minDist) {
				#pragma omp critical
				{
					minDist = distanceBetweenCircles;
				}
			}
		}
	}

	return minDist;
}

// For given array of points performs K-Means algorithem and returns found clusters centroids.
void K_Means(CLUSTER_S* & clusters,int nClusters,POINT_S *points,int nPoints,double systemAngel,int nMaxIterations,double timeT)
{
	AssignPointsToClosestClustersCuda(points,nPoints,clusters, nClusters,nMaxIterations);
}

// Initiate empty clusters with arbitrary centroid points.
void RestClusters(CLUSTER_S* &clusters,int nClusters)
{
	for (int i = 0; i < nClusters; i++) {
		clusters[i].centroid.x = i;
		clusters[i].centroid.y = i;
		clusters[i].points_size = 0;
	}
}

// Initiate empty clusters with arbitrary centroid points.
void InitClusters(CLUSTER_S* &clusters,int nClusters,POINT_S* points)
{
	clusters = (CLUSTER_S*)malloc(nClusters * sizeof(CLUSTER_S));

	for (int i = 0; i < nClusters; i++) {
		clusters[i].centroid.x = points[i].x;
		clusters[i].centroid.y = points[i].y;
		clusters[i].points_size = 0;
		clusters[i].id = i;
	}
}

// Create map of points in given time t relative to circle centers positions.
void CreatePointsMap(POINT_S* points, CIRCLE_S* circles, int circles_size, double angel)
{
	#pragma omp parallel for
	for (int i = 0; i<circles_size; i++) {
		points[i].x = circles[i].centerX + circles[i].radius*cos(angel);
		points[i].y = circles[i].centerY + circles[i].radius*sin(angel);
	}
}

void LoadPoints(POINT_S* points, int points_size)
{
	for (int i = 0; i < points_size; i++)
	{
		points[i].x = GenerateRandomDouble(0,10);
		points[i].y = GenerateRandomDouble(0, 10);
		points[i].clusterID=-1;
	}
}

// For given array of snapshoots return index of best snapshot which is the one with minimum distance between clusters;
int GetBestSnapShotIndex(SYSTEM_SNAPSHOT_S* snapshots, int snapshots_size)
{
	double min = snapshots[0].minimumDistanceBetweenClusters;
	int minIndex = 0;
	for (int i = 1; i<snapshots_size; i++) {
		if (snapshots[i].minimumDistanceBetweenClusters<min) {
			min = snapshots[i].minimumDistanceBetweenClusters;
			minIndex = i;
		}
	}
	return minIndex;
}

// For given array arr , returns the index of minimum number of this array.
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

double GetDistanceBetweenCircles(POINT_S p1, POINT_S p2)
{
	double dx = p2.x - p1.x;
	double dy = p2.y - p1.y;
	return sqrt(dx*dx + dy*dy);
}

// Reads circles of points,number of clusters to find, time increment deltaT,time interval T, and maximum number of iterations LIMIT. 
void ReadInputFile(char * fullFilePath, CIRCLE_S* &circles, int *nCircles, int *nClusters, double *deltaT, double *timeIntervalT, int *nMaxIterations)
{
	FILE *f;
		
	errno_t errorCode = fopen_s(&f, fullFilePath, "r");
	
	if (errorCode != 0)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	int row = fscanf_s(f, "%d %d %lf %lf %d", nCircles, nClusters, deltaT, timeIntervalT, nMaxIterations);
	circles = (CIRCLE_S*)malloc(*nCircles * sizeof(CIRCLE_S));

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

void WritePointsToFile(char* fileFullPath, POINT_S* points, int nPoints)
{

	FILE *f;
	errno_t errorCode = fopen_s(&f, fileFullPath, "w");

	if (errorCode != 0)
	{
		printf("Error writing file!\n");
		exit(1);
	}

	
	for (int i = 0; i < nPoints; i++) {
		fprintf(f, "%d %lf %lf\n", points[i].clusterID, points[i].x, points[i].y);
	}
	

	fclose(f);
}

//Generates random double in range between min and max, 
double GenerateRandomDouble(int min, int max)
{
	double f = (double)rand() / RAND_MAX;
	return min + f * (max - min);
}